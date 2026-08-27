# MoeDistributeDispatchV2 Token 预取特性及实现分析

> 分析对象：[cann/ops-transformer PR #9347：Dispatch 算子支持 Token 预取](https://gitcode.com/cann/ops-transformer/pull/9347)
> PR Head：`2ec502d50198fc6fb378129cf3e668f986bb23fb`
> 分析日期：2026-08-27

## 1. 结论概览

PR #9347 为 `MoeDistributeDispatchV2` 增加了一种面向小 active BS 的性能优化：在真正按照 expert 分散读取 token 之前，先执行一次 `GM → UB scratch` 搬运，利用这次真实读取附带产生的 L2 Cache 填充，让后续多个 AIV 对同一个 token 的读取更容易命中共享 L2。

在 Full Mesh 实现中，这次搬运不是由负责 All-to-All token 发送的 AIV 执行，而是由后半组 Cumsum AIV 在 `CalAndSendCntByRank()` 中发起。预取使用 MTE2（`GM → UB`），随后的计数发送使用 MTE3（`UB → GM`）；两条搬运流水相互独立，源码也没有在两者之间建立 `MTE2 → MTE3` 等待，因此预取可以与计数计算、计数发送重叠，而不会把 MTE3 计数发送串行阻塞在预取之后。

它不是一条专用的 L2 prefetch 指令，而是通过 `DataCopyPad` 主动触发内存访问：

```text
预取阶段：GM x[token, :] ──DataCopyPad──> 独立 UB scratch
                              │
                              └─ 副作用：对应数据进入 L2 Cache

消费阶段：GM x[token, :] ──L2 hit──> 输入 UB ──量化/封装──> 通信窗口
```

UB scratch 中的数据不会参与后续计算，也不会写回 GM。真正有价值的是内存访问在 L2 中留下的数据，因此代码将这一行为描述为 `fire-and-forget`。

该 PR：

- 不修改 Host API 或算子输入输出；
- 只修改两个 kernel 头文件；
- 共新增 45 行代码，没有删除代码；
- 覆盖普通 Dispatch 和 Full Mesh 两种实现；
- 主要优化连续、小 active BS 的 token 访问；
- 不改变功能正确性，预取是否及时完成只影响性能。

## 2. 为什么 Dispatch 需要 Token 预取

### 2.1 输入布局

Dispatch 输入 `x` 的逻辑布局为：

```text
x: [BS, H]

token 0: x[0, 0:H]
token 1: x[1, 0:H]
...
token t 的起始地址：xGMTensor_[t * axisH_]
```

单个 token 的实际搬运长度为：

```cpp
copyInAxisH_ * sizeof(XInType)
```

其中 `copyInAxisH_` 通常等于 `axisH_`；对于 A5 上的 FP4 打包输入，它会按两个 FP4 元素占一个字节进行换算。

### 2.2 原有消费模式

每个 token 会被路由到 `K` 个 MoE expert，并可能额外发送给 shared expert。原有实现会在 expert 分发循环中多次读取同一个 token：

```text
同一个 token
  ├─→ expert 0：GM → UB → 量化/封装 → 远端窗口
  ├─→ expert 1：GM → UB → 量化/封装 → 远端窗口
  ├─→ ...
  ├─→ expert K-1：GM → UB → 量化/封装 → 远端窗口
  └─→ shared expert：GM → UB → 远端窗口
```

这些读取可能由不同 AIV 执行。虽然硬件缓存会在第一次真实读取之后自然升温，但第一次 miss 仍处在关键发送路径上，而且多个核可能近似同时发起相同 token 的访问。

Token 预取把第一次访问提前，并尝试与 mask、计数或其他核上的计算重叠：

```text
连续预取：x[0], x[1], ..., x[activeBS-1]
                         ↓
                    共享 L2 Cache
                         ↓
          按 expert 分散、跨 AIV 的后续读取
```

这对 decode、小 batch 场景尤其有价值：active token 数较少，但一个 token 可能被 `K + sharedExpertNum` 次消费，一次预取的成本可以被多次读取摊销。

## 3. 核心代码实现

两个 kernel 都增加了独立的预取缓冲区和成员函数：

```cpp
TBuf<> prefetchScratch_;

__aicore__ inline void TokenPrefetchToL2Cache();
```

核心实现可以概括为：

```cpp
tpipe_->InitBuffer(
    prefetchScratch_,
    Ceil(copyInAxisH_ * sizeof(XInType), UB_ALIGN) * UB_ALIGN);

LocalTensor<XInType> scratch = prefetchScratch_.Get<XInType>();

for (uint32_t t = pfStart; t < pfEnd; ++t) {
    DataCopyPad(
        scratch,
        xGMTensor_[t * axisH_],
        copyParams,
        prefetchPadParams);
}
```

实现具有以下特点：

1. 源地址是真实输入 `xGMTensor_`。
2. 每次读取一个完整 token 行。
3. scratch 按 `UB_ALIGN`，即 32 字节对齐。
4. scratch 数据不会被读取、量化或写回。
5. 不使用 `SetL2CacheHint` 或专用预取指令，依赖 GM→UB 访问的默认缓存行为。
6. 当前版本使用独立 `TBuf`，不与真正发送 token 使用的 `inQueue` 共用 UB。
7. 预取之后没有立即等待 MTE2 完成，以保留异步发射和访存重叠能力。

## 4. 普通 Dispatch 实现

源码：[`moe_distribute_dispatch_v2.h`](https://gitcode.com/cann/ops-transformer/blob/2ec502d50198fc6fb378129cf3e668f986bb23fb/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h)

### 4.1 使能条件

普通实现使用以下门控：

```cpp
if (isExpertMaskFlag_ ||
    activeMaskBsCnt_ == 0 ||
    activeMaskBsCnt_ > aivNum_) {
    return;
}
```

也就是只有在以下条件全部满足时才预取：

- token 源索引为连续区间；
- active token 数不为 0；
- `activeMaskBsCnt_ <= aivNum_`。

随后用全部 AIV 对 token 区间分核：

```cpp
SplitToCore(
    static_cast<uint32_t>(activeMaskBsCnt_),
    aivNum_,
    pfStart,
    pfEnd,
    pfPerCore,
    true);
```

由于 active token 数不超过 AIV 数，每个 AIV 最多预取一个 token。这样既控制了额外读取量，也避免了单核为了预取而连续搬运大量 token。

### 4.2 调用位置

普通 Dispatch 的调用链为：

```text
Process
  └─ AlltoAllDispatch
       ├─ 初始化 activeMaskBsCnt_ = axisBS_
       ├─ TokenActiveMaskCal（可选）
       ├─ ExpertActiveMaskCal（可选）
       ├─ ZeroComputeExpertMaskCal（可选）
       │
       ├─ shared 发送核
       │    ├─ TokenPrefetchToL2Cache
       │    └─ SendToSharedExpert
       │          └─ ProcessToken
       │
       └─ MoE 发送核
            ├─ TokenPrefetchToL2Cache
            └─ SendToMoeExpert
                  └─ ProcessToken
```

真正消费预取数据的是 `ProcessToken()`。它仍然从 `xGMTensor_` 发起正常的 `DataCopyPad`：

```cpp
DataCopyPad(
    xInTensor_,
    xGMTensor_[tokenIndex * axisH_],
    xCopyParams_,
    padParams);
```

预取不会把数据直接交给 `ProcessToken()`，而是让这次正常读取更可能命中 L2。

### 4.3 为什么 Expert Mask 场景关闭预取

非 expert-mask 场景按照连续 token 区间读取：

```text
0, 1, 2, ..., activeMaskBsCnt_ - 1
```

二维 expert mask 生效后，真正源 token 需要经过索引映射：

```cpp
srcTokenIndex = validBsIndexTensor_.GetValue(tokenIndex);
```

此时有效 token 可能是稀疏、不连续的。如果仍然预取 `[0, activeMaskBsCnt_)`，可能读入大量不会被消费的 token，却漏掉真正需要的离散 token，因此代码直接通过 `isExpertMaskFlag_` 关闭预取。

## 5. Full Mesh 实现

源码：[`moe_distribute_dispatch_v2_full_mesh.h`](https://gitcode.com/cann/ops-transformer/blob/2ec502d50198fc6fb378129cf3e668f986bb23fb/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h)

### 5.1 使能条件

Full Mesh 使用固定上限：

```cpp
constexpr uint32_t TOKEN_PREFETCH_MAX_ACTIVE_BS = 16U;
```

门控条件为：

```cpp
if (isExpertMaskFlag_ ||
    activeMaskBsCnt_ == 0 ||
    activeMaskBsCnt_ > TOKEN_PREFETCH_MAX_ACTIVE_BS) {
    return;
}
```

因此当前代码中的 Full Mesh 预取上限是 **16 个 active token**，不是早期机器人评论中提到的 256。机器人摘要对应的是 PR 的早期版本，不能代表当前 head。

### 5.2 利用 Cumsum 核为 Dispatch 核预热 L2

Full Mesh 会把 AIV 分成两组：

```text
全部 AIV
  │
  ├─ 前 aivUsedAllToAll_ 个核
  │      └─ AllToAllDispatch：负责真正发送 token
  │
  └─ 后 aivUsedCumSum_ 个核
         └─ CalCumSum：负责 token count、recv count 等计算
```

Token 预取放在后半部分 Cumsum 核的 `CalAndSendCntByRank()` 中：

```text
Cumsum 核：
  ExpIdsCopyAndMaskCal
      ↓
  初始化状态数据
      ↓
  TokenPrefetchToL2Cache
      ↓
  继续计算并发送 token count

Dispatch 核（并行）：
  AllToAllDispatch
      ↓
  SendToSharedExpert / SendToMoeExpert
      ↓
  从相同 x 地址读取 token
```

从单个 Cumsum 核内部看，关键顺序更具体地是：

```text
Vector：初始化 statusTensor
   ↓ V_S
Scalar：发起 TokenPrefetchToL2Cache
   └─ DataCopyPad(GM → UB scratch)，进入 MTE2 流水，不立即等待
   ↓
Scalar/Vector：计算各 rank / expert 的 token count
   ↓ S_MTE3
MTE3：DataCopy(statusTensor → 远端状态窗口)
```

这里预取走 MTE2，计数发送走 MTE3。`CalAndSendCntByRank()` 在预取之后没有执行 `MTE2_MTE3`、`MTE2_S` 或 `PIPE_ALL`，因此计数计算不需要先等预取完成，计数的 MTE3 搬运也可以与预取的 MTE2 搬运并行推进。这正是把预取放到 Cumsum 核而不是 All-to-All 发送核上的主要价值：利用 Cumsum 核计数路径中的另一条搬运流水，为并行运行的 Dispatch 核预热共享 L2。

这形成了跨核的软件流水：

```text
时间 ───────────────────────────────────────>

Cumsum 核：   mask/count ── GM→scratch 预取 ── count/status
                               │
                               └── 填充共享 L2
                                      │
Dispatch 核：发送准备 ─────────────────┴── GM→UB→量化→窗口
```

这里没有跨核同步。预取与发送之间是性能依赖，而不是正确性依赖：

- 预取先完成：发送读取更可能 L2 hit；
- 两者部分重叠：可能只隐藏部分访存延迟；
- 发送先到达：仍按原逻辑从 GM 读取，结果不受影响。

需要限定的是，“MTE2 与 MTE3 不共享流水”只说明预取不会直接占住计数发送所使用的 MTE3 流水，并不表示它对整个 Cumsum 路径绝对零开销：

- 调用 `TokenPrefetchToL2Cache()`、分核和发射 `DataCopyPad` 仍需要少量标量指令；
- 预取仍占用 Cumsum 核的 MTE2 队列、UB scratch，以及芯片共享的 GM/L2 带宽；
- `CalAndSendCntByRank()` 之后的 `WaitDispatch()` 会用 MTE2 从状态窗口轮询读回计数状态。如果预取尚未排空，后续 MTE2 请求可能受到同流水顺序或带宽竞争影响；
- 因而代码能证明的是“预取与当前计数 MTE3 发送没有显式流水依赖、具备重叠条件”，实际是否完全隐藏仍需性能数据验证。

### 5.3 实际覆盖范围

当前 Full Mesh 的调用还有架构与分支限制：

- 调用位于 A5，即 `__NPU_ARCH__ == 3510` 的 `CalAndSendCntByRank()` 路径；
- 存在 `hasElasticInfoFlag_` 时改走 `CalAndSendCntByExp()`，不会执行此处预取；
- A3 的 Cumsum 路径也不会执行 Full Mesh 版本的预取；
- A3 的 BS 模式要求 `activeBS > 16`，而预取仅在 `activeBS <= 16` 时启用。

因此 Full Mesh 预取主要针对 A5、小 active BS、连续 token、非弹性缩容的按 expert 分散读取路径，而不是已经做到“一个 token 只读取一次”的大 BS 模式。

## 6. 同步与 UB 生命周期

### 6.1 为什么没有立即等待 MTE2

预取代码在 `DataCopyPad` 后没有执行显式的 `MTE2_S`、`MTE2_V` 等等待。这是为了保留以下能力：

- 尽早发射 GM 请求；
- 与标量、向量或其他核上的计算重叠；
- 不把预取退化为关键路径上的同步搬运。

当前版本能够这样做的前提是：

1. scratch 数据不会被任何计算读取；
2. scratch 不与 dispatch 的输入队列共用；
3. scratch 的 `TBuf` 生命周期覆盖预取搬运；
4. 正确性不依赖搬运完成时刻。

### 6.2 与 Cumsum 的 MTE3 计数发送如何重叠

Full Mesh 的 `CalAndSendCntByRank()` 在 `TokenPrefetchToL2Cache()` 之后继续计算计数，随后通过以下路径发送状态：

```cpp
SyncFunc<AscendC::HardEvent::S_MTE3>();
DataCopy<int32_t>(rankGMTensor, statusTensor_[...], ...);
```

其中 token 预取是 `DataCopyPad(GM → UB)`，属于 MTE2；计数发送是 `DataCopy(UB → GM)`，属于 MTE3。两者之间没有 `MTE2_MTE3` 同步，也没有用 `PIPE_ALL` 将预取排空，所以 MTE3 不必等待 MTE2 预取完成。

这让 Cumsum 核同时承担两个互补角色：MTE3 向各 rank 的状态窗口发送 token count，MTE2 则提前读取 `x` 为 All-to-All 核预热 L2。这里的“不影响”应理解为“不直接串行阻塞计数发送的 MTE3 流水”，而不能外推为对 Cumsum 核执行时间、后续 MTE2 轮询或全芯片带宽完全无影响。

### 6.3 早期 Review 问题及当前状态

PR 早期版本曾复用 `inQueue` 的 UB slot，并对同一个 scratch 连续发起异步搬运。Review 指出：如果预取尚未完成，slot 就被释放并重新用于真实 token，可能产生 UB 生命周期重叠及 WAW/RAW 风险。

当前 head 已改为：

```cpp
TBuf<> prefetchScratch_;
```

也就是使用独立的预取缓冲区，并把 Full Mesh 的 active BS 上限收紧到 16。这样消除了与真实 dispatch 输入队列直接复用同一 UB slot 的问题。

## 7. 性能收益模型

设：

- active token 数为 `B`；
- 每个 token 字节数为 `T`；
- 每个 token 被发送给 `K` 个 MoE expert；
- shared expert 数为 `S`。

预取增加的读取量近似为：

```text
额外读取量 = B × T
```

后续潜在复用次数近似为：

```text
潜在消费次数 = B × (K + S)
```

预取值得执行的条件是：

```text
提前隐藏的 miss 延迟 + 多核 L2 复用收益
    >
额外 GM/MTE2/L2 带宽 + UB 占用 + cache 污染
```

Full Mesh 还多了一层隐藏条件：这笔额外 MTE2 搬运由 Cumsum 核发起，而计数发送主要使用 MTE3，因此预取延迟可以被计数计算和 MTE3 发送覆盖。性能评估既要观察 All-to-All 核的 L2 命中收益，也要确认 Cumsum 核后续 `WaitDispatch()` 的 MTE2 轮询没有被明显推迟。

PR 通过以下门控控制成本：

| 实现          | active BS 上限 | 分核方式                     | 主要目标场景        |
| ------------- | -------------: | ---------------------------- | ------------------- |
| 普通 Dispatch |      `aivNum_` | 全部 AIV，每核最多一个 token | 小 BS、连续 token   |
| Full Mesh     |             16 | Cumsum 核合作预取            | A5、小 BS、跨核重叠 |

## 8. 限制与风险点

### 8.1 不是强制 L2 驻留

代码只通过 GM→UB 读取促成 L2 填充，并没有锁定 cache line。预取数据可能在真正消费前被其他流量淘汰，因此实际收益依赖：

- token 大小；
- active BS；
- L2 容量和并发流量；
- 预取与消费之间的时间距离；
- MTE2 带宽竞争；
- 多 AIV 对同一 token 的真实复用程度。

### 8.2 额外 UB 未纳入部分预算逻辑

普通实现已有 `totalUsedUB_`，用于决定输入输出队列采用单缓冲还是双缓冲，但新增加的一个 token 大小 `prefetchScratch_` 没有计入该预算。

这不等于当前代码一定发生 UB 越界，但在最大 H、复杂量化和多个 mask buffer 同时启用的组合下，应该补充 UB 边界检查，确认新增 scratch 不会压缩已有缓冲空间。

### 8.3 地址步长需要覆盖打包 dtype

预取缓冲大小和搬运长度使用 `copyInAxisH_`，源偏移使用 `t * axisH_`。这与多数现有 token 读取保持一致，但 FP4 打包输入还存在使用 `tokenIndex * copyInAxisH_` 的分支。

因此建议针对 FP4、smooth scale、量化/非量化组合检查预取地址是否和实际消费者完全一致，避免“读入了一个合法地址，但没有预热真正消费地址”的情况。

### 8.4 缺少性能数据

PR 的编译、UT、预冒烟和静态检查已有通过记录，但 PR 描述和评论没有给出以下数据：

- Dispatch 单算子耗时对比；
- 端到端模型时延；
- L2 hit-rate；
- GM 读取带宽；
- 不同 BS、H、K 下的收益曲线；
- 预取开启后的性能回退区间。

因此当前能够确认的是实现机制和功能测试状态，不能根据 PR 现有证据量化实际性能提升。

## 9. 建议验证矩阵

建议至少覆盖：

| 维度          | 建议取值                                              |
| ------------- | ----------------------------------------------------- |
| active BS     | 1、2、4、8、16、17、`aivNum_`、`aivNum_ + 1`          |
| H             | 常用 H、最大 H、非 32B 整除 H                         |
| K             | 1、2、4、8                                            |
| dtype         | FP16、BF16、INT8/HIFP8、FP4                           |
| quant         | 非量化、per-token、per-group、MX                      |
| mask          | 无 mask、token mask、expert mask、zero-compute expert |
| shared expert | 0、1、多 shared expert                                |
| 实现          | 普通 Dispatch、Full Mesh                              |
| 架构          | A3、A5，按实际支持范围区分                            |
| elastic info  | 开启、关闭                                            |

性能指标建议包括：

```text
1. Kernel 总耗时
2. GM→L2 / L2→Core 相关带宽
3. L2 hit-rate
4. MTE2 busy ratio
5. 不同 active BS 下的收益拐点
6. 预取开启前后的 UB 占用
7. 多轮 Dispatch 下 cache 稳态表现
8. Full Mesh 中 `CalAndSendCntByRank`、`WaitDispatch` 的分段耗时，以及 Cumsum 核的 MTE2/MTE3 利用率
```

## 10. 总结

PR #9347 的 Token 预取本质是一个软件控制的 cache warm-up：

```text
提前读取一次连续 token
        ↓
利用 GM→UB 搬运填充共享 L2
        ↓
后续按 expert 分散读取复用 L2
        ↓
降低首次 miss 位于关键发送路径上的概率
```

普通 Dispatch 通过 `activeMaskBsCnt_ <= aivNum_` 保证每个 AIV 最多预取一个 token；Full Mesh 则利用原本负责 Cumsum 的后半部分 AIV，在发送核并行工作的同时为其预热 L2，并将 active BS 上限限制为 16。Full Mesh 中预取使用 MTE2、计数发送使用 MTE3，且两者之间没有显式同步，因此不会把 MTE3 计数发送串行阻塞在预取之后；但预取仍可能通过 MTE2 队列、后续状态轮询和共享 GM/L2 带宽产生间接影响，是否完全隐藏需要实测确认。

它的优点是代码侵入小、不改变接口、不建立新的正确性依赖；限制是收益高度依赖硬件缓存行为和调度时序，同时增加一次 token 读取和一个 token 大小的 UB。要判断其是否适合合入主线，仍需要以 A5 小 BS 场景为核心补充性能曲线、L2 指标和 UB 边界验证。

## 参考链接

- [PR #9347：Dispatch 算子支持 Token 预取](https://gitcode.com/cann/ops-transformer/pull/9347)
- [提交 2ec502d5](https://gitcode.com/song-xinyi-001/ops-transformer_prefetch/commit/2ec502d50198fc6fb378129cf3e668f986bb23fb)
- [普通 Dispatch 源码](https://gitcode.com/cann/ops-transformer/blob/2ec502d50198fc6fb378129cf3e668f986bb23fb/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h)
- [Full Mesh 源码](https://gitcode.com/cann/ops-transformer/blob/2ec502d50198fc6fb378129cf3e668f986bb23fb/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h)
