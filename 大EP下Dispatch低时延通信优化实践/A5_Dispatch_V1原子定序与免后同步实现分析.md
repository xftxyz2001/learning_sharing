# A5 Dispatch V1 原子定序与“免后同步”实现分析

> 分析对象：[cann/ops-transformer PR #9097](https://gitcode.com/cann/ops-transformer/pull/9097)
> PR 标题：`[WIP]A5 Dispatch-v1版本原子定序，免后同步`
> PR Head：`43842e894849c97a238bcfe4a0bc72580c0f18c1`
> 分析日期：2026-08-27

## 1. 结论概览

PR #9097 对 `MoeDistributeDispatchV2` 的发送、状态通知、接收窗口连续化和 `expertTokenNumsOut` 计算进行了核心路径重构。目标是：

> 用“每专家 GM 原子计数器”代替原来的“跨核接收计数前缀和”，使每个 AIV 在自己负责的一组远端状态全部到达后即可独立开始 `LocalWindowCopy()`，不必先等待本卡所有 AIV 会合。

这里的关键粒度是“AIV 分配到的 status 段”，而不是“任意单个窗口”，也不是“整个 expert”。

PR 的主体包含四部分：

1. **按专家分核发送**：从按 `BS × K` flat 索引分核，改成按 expert 分核并建立 per-expert bitmask。
2. **同核发送、同核写状态**：纯 MoE 场景中，负责某个 expert 数据发送的核同时负责该 expert 的状态通知。
3. **原子槽位分配**：接收侧使用 `AtomicAdd(counter[expert], count)` 为每个窗口块分配互不重叠的专家内偏移。
4. **分段暂存和最终密排**：专家 0 直接写最终输出；其他专家先写 workspace staging，末尾根据每专家总数重排到连续输出。

总体链路为：

```text
按 expert 发送并写状态
        ↓
接收核分别等待自己负责的远端状态
        ↓
AtomicAdd 为各 expert 分配唯一槽位
        ↓
expert 0 直写；expert > 0 写 workspace staging
        ↓
末尾一次 SyncAll
        ↓
读取每 expert 计数器并计算 expert 前缀和
        ↓
workspace → expandX/scale/expandIdx 最终密排
```

“免后同步”并不代表 kernel 中不再存在 `SyncAll`。它主要删除了 `WaitDispatch()` 后、`LocalWindowCopy()` 前的全核同步和跨核前缀和；当前实现仍在最终 staging 重排前执行一次全核同步，存在 shared expert 时发送状态前也仍需同步。

## 2. PR 变更范围

PR 当前包含 1 个提交，修改 3 个文件：

| 文件 | 作用 | 变更规模 |
|---|---|---:|
| `moe_distribute_dispatch_v2.h` | Kernel 主流程、发送、原子定序、暂存与重排 | `+495/-337` |
| `moe_distribute_dispatch_v2_tiling.cpp` | 扩大 staging workspace | `+18/-5` |
| `moe_distribute_v2_constant.h` | 原子计数器地址和步长 | `+5/-2` |

合计 `+518/-344`，属于执行模型重构，而不是局部性能 patch。

源码：

- [Kernel 主文件](https://gitcode.com/cann/ops-transformer/blob/43842e894849c97a238bcfe4a0bc72580c0f18c1/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h)
- [Host Tiling](https://gitcode.com/cann/ops-transformer/blob/43842e894849c97a238bcfe4a0bc72580c0f18c1/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp)
- [常量定义](https://gitcode.com/cann/ops-transformer/blob/43842e894849c97a238bcfe4a0bc72580c0f18c1/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h)

## 3. 原实现为什么需要同步

原执行链为：

```text
AlltoAllDispatch
    ↓
SetStatus
    ↓
WaitDispatch
    ├─ 等待本核负责的远端窗口状态
    ├─ 清状态
    ├─ A3：SyncCntOnCore
    └─ SyncAll
         ↓
LocalWindowCopy
    ├─ GetCumSumA5 / GetCumSum
    ├─ 计算本核 beginIdx
    ├─ 窗口 → 最终输出
    └─ 写 sendCounts
         ↓
SetExpertTokenNumsA5 / SetExpertTokenNums
```

多个 AIV 分别处理不同来源 rank、不同本地 expert 的窗口块。假设：

```text
core 0 收到 3 个 token
core 1 收到 2 个 token
core 2 收到 4 个 token
```

为了写入连续输出，原实现必须先得到：

```text
core 0 beginIdx = 0
core 1 beginIdx = 3
core 2 beginIdx = 3 + 2 = 5
```

每个核必须知道排在自己之前的核总共收到多少 token，因此需要先保证所有 count 可见，再计算跨核前缀和。即使某个 AIV 的远端数据已经全部到达，它也不能立即开始最终输出。

### 3.1 Status 的含义与真实等待粒度

在普通 MoE 专家卡上，一个 status block 对应：

```text
(一个源 Rank → 一个目标本地 expert) 的一个窗口块
```

发送端先将该窗口块的 token 全部写入远端 window，再写入 `count + flag`。因此 flag 只能证明“这一个源 Rank 到这一个目标 expert 的窗口块完成”，并不代表该 expert 来自所有源 Rank 的 token 都已到齐。

接收侧的 status 索引为 expert-major 布局：

```text
statusIndex = localExpertId × epWorldSize + sourceRankId
```

`Init()` 将 `rscvStatusNum_` 个 status 连续均分给全部 AIV：

```cpp
recStatusNumPerCore_ = rscvStatusNum_ / aivNum_;
startStatusIndex_ = recStatusNumPerCore_ * aivId_;
```

`WaitStatusA5()` 每次从 `startStatusIndex_` 开始读取本核的 `recStatusNumPerCore_` 个 flag，只有它们全部为 1 才退出轮询：

```text
sumOfFlag == recStatusNumPerCore_
```

所以应当区分三种说法：

- “单个窗口一到就搬”：只有 `recStatusNumPerCore_ == 1` 时成立。
- “整个 expert 的所有源 Rank 都到齐才搬”：一般不成立。
- 实际行为：一个 AIV 等自己的 status 段全部到齐，然后搬这一段对应的所有窗口。

### 3.2 快慢卡场景中新旧策略的差异

假设两个 AIV 各自负责两个 status（括号中表示该status块的到达时间）：

```text
AIV0：A(10 μs)、B(12 μs)
AIV1：C(40 μs，慢卡)、D(11 μs)
```

原策略：

```text
12 μs：AIV0 的 status 段已就绪，但阻塞在 SyncAll
40 μs：AIV1 就绪，两个 AIV 一起开始 LocalWindowCopy
```

原子定序后：

```text
12 μs：AIV0 直接为 A、B 申请槽位并开始搬运
40 μs：AIV1 再为 C、D 申请槽位并搬运
```

这项优化将快组的 `LocalWindowCopy` 与慢组的通信等待重叠。它不会消除慢卡：如果慢 status 与快 status 被分到同一 AIV，该 AIV 仍被阻塞；最终密排也仍需等待所有 AIV 完成。

#### 3.2.1 交互式时延对比

拖动参数可以观察两种方案的关键路径。原子方案仍然受最慢核约束；只有提前完成的搬运足以覆盖末尾密排成本时，总时延才会下降。

<iframe
  src="./assets/dispatch-overlap-timeline.html"
  title="Dispatch 原子定序时延对比"
  width="100%"
  height="980"
  sandbox="allow-scripts"
  referrerpolicy="no-referrer"
></iframe>

如果当前 Markdown 预览器不支持 iframe，请[单独打开交互演示](./assets/dispatch-overlap-timeline.html)。

## 4. 新方案：按专家原子定序

PR 为每个本地 expert 增加独立 GM 原子计数器：

```cpp
uint32_t expertOffset = AtomicAdd<uint32_t>(
    reinterpret_cast<__gm__ uint32_t *>(
        expertCountersGMTensor_.GetPhyAddr(targetExpertId * COUNTER_STRIDE)),
    count);
```

`AtomicAdd` 不负责检测窗口是否到达；到达检测由 `WaitStatusA5()`/`WaitStatus()` 完成。它只在 status 段已就绪后负责槽位分配和定序：

1. 将当前窗口块的 `count` 加入该 expert 总数；
2. 返回加法前旧值，作为该窗口块在 expert 内部的唯一开始偏移。

例如：

```text
counter 初始为 0

窗口 A：AtomicAdd(+3) → 返回 0，负责 expert 内 [0, 3)
窗口 B：AtomicAdd(+2) → 返回 3，负责 expert 内 [3, 5)
窗口 C：AtomicAdd(+4) → 返回 5，负责 expert 内 [5, 9)
```

无论三个 AIV 的执行顺序如何，返回区间都不重叠。每个 AIV 不再依赖其他核的 count，也不需要计算前序核前缀和。

| 项目 | 原方案 | PR #9097 |
|---|---|---|
| 唯一输出偏移 | 跨核前缀和 | `AtomicAdd` 返回旧值 |
| LocalWindowCopy 前同步 | 需要 | 删除 |
| 输出块顺序 | 相对固定的核/rank 顺序 | 原子到达顺序 |
| 中间空间 | 小型 count workspace | 完整 token staging workspace |
| 最终密排 | LocalWindowCopy 直接完成 | expert 0 直写，其他 expert 末尾重排 |

## 5. 原子计数器布局和 Ping-Pong

新增常量为：

```cpp
constexpr uint64_t EXPERT_COUNTERS_OFFSET = 880UL * 1024UL;
constexpr uint64_t EXPERT_COUNTERS_PINGPONG_OFFSET = 890UL * 1024UL;
constexpr uint32_t COUNTER_STRIDE = 512 / sizeof(uint32_t);
```

布局为：

```text
statusDataSpaceGm_
  ├─ dataState=0：+880 KiB
  │    ├─ expert 0 counter：+0×512B
  │    ├─ expert 1 counter：+1×512B
  │    └─ ...
  └─ dataState=1：+890 KiB
       ├─ expert 0 counter
       ├─ expert 1 counter
       └─ ...
```

当前轮使用 `ExpertCountersRegionOffset(dataState_)`，末核通过 `ResetNextRoundCounters()` 清零下一轮区域。

计数器 GlobalTensor 显式关闭 L2：

```cpp
expertCountersGMTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
```

最终读取前还调用 `DataCacheCleanAndInvalid`，设计意图是保证 GM atomic 与标量读取之间的 cache 一致性。

## 6. 发送侧：按 Expert 分核

新路径由文件内宏直接启用：

```cpp
#define EXPERT_SPLIT_SEND
```

这不是运行时开关，当前源码默认编译按 expert 发送。

### 6.1 步进分核

`SplitExpertNumToCore()` 使用：

```text
stride = moeUsedAivNum_

core 0：expert 0, stride, 2×stride, ...
core 1：expert 1, stride+1, 2×stride+1, ...
...
```

步进分核把相邻热点 expert 分散到不同核，降低连续 expert 负载偏斜造成的单核过载。

### 6.2 Per-expert bitmask

`CalExpertSendNum()` 对本核负责的每个 expert：

1. `CompareScalar` 比较完整 `expertIdsTensor_`；
2. 生成该 expert bitmask；
3. `GatherMask` 计算 token 数；
4. 将结果写入 `tokenNumToExpertTensor_`。

`SendToMoeExpertByExpert()` 再使用 `ScalarGetSFFValue<1>` 每次扫描 64 bit mask，精准找到属于该 expert 的下一个 `BS × K` 索引，删除原路径中重复扫描前序 expert ID 的 `CalTokenSendExpertCnt()`。

二维 expert mask 下，压缩索引通过 `validExpertIndexTensor_` 映射回原始空间：

```text
origIdx       = validExpertIndexTensor_[compressedIdx]
srcTokenIndex = origIdx / K
topKIndex     = origIdx % K
```

### 6.3 错峰发送

本核内部用：

```cpp
expertIndex = (index + epRankId_ + aivId_) % sendNum_;
```

它只改变遍历起点，不改变 expert 集合，用于让不同源 rank、不同核错峰访问目标窗口。

## 7. 状态通知：同核发送、同核写 Flag

原 `SetStatus()` 会重新按全部 expert 分核，负责发送某 expert token 的核未必是负责写该 expert 状态的核，因此必须先全核同步。

新 `SetStatusByExpert()` 复用发送阶段的 expert 所有权：

```text
front MoE 核：
  SendToMoeExpertByExpert(expert e)
       ↓
  SetStatusByExpert(expert e)
```

纯 MoE 场景中，同一个 AIV 先写完 token，再写同一 expert 的 count/flag，不再需要为了对齐发送核和状态核执行一次 `SyncAll`。

shared expert 仍按 token 切分，一个 shared expert 的 token 可能由多个 rear 核共同发送，单个状态核无法确认其他 rear 核是否完成，因此仍保留：

```cpp
if (sharedExpertRankNum_ != 0) {
    SyncAll<true>();
}
```

## 8. 接收侧：LocalWindowCopy 原子分配槽位

PR 删除了 `WaitDispatch()` 末尾原有的 `SyncAll`，并删除：

- `SyncCntOnCore()`；
- `GetCumSum()`；
- `GetCumSumA5()`；
- `sumCoreBuf_`、`sumLocalBuf_`、`sumContinueBuf_` 等前缀和缓冲。

每个接收 AIV 在自己负责的整个 status 段全部到达后直接执行：

```cpp
expertOffset = AtomicAdd(counter[targetExpertId], count);
baseSlot = targetExpertId * globalBS_ + expertOffset;
```

中间 staging 布局为：

```text
workspace staging

expert 0 segment：[0×globalBS, 1×globalBS)
expert 1 segment：[1×globalBS, 2×globalBS)
expert 2 segment：[2×globalBS, 3×globalBS)
...
```

每个 expert segment 最大容量为 `globalBS_`，不同 expert 天然隔离；同一 expert 内由 `AtomicAdd` 分配不重叠区间。

## 9. Expert 0 直写与其他 Expert 暂存

最终输出要求按 expert 连续排列：

```text
[expert0 全部 token][expert1 全部 token][expert2 全部 token]...
```

expert 0 的最终起点恒为 0，所以 `expertOffset` 可以直接作为最终位置。它直接写：

- `expandXOut`；
- `dynamicScalesOut`；
- `expandIdxOut`；
- 可选 `expandScalesOut`。

expert 1 的最终起点依赖 expert 0 的实际总数；expert 2 又依赖 expert 0 和 1。其他 AIV 尚未完成时，这些起点未知，因此 expert > 0 先写固定分段：

```text
stagingOffset(e) = e × globalBS + expertOffset
```

待计数稳定后再执行最终密排。

## 10. Staging Token 与 Workspace

暂存的是完整 token 记录：

```text
staging token
  ├─ data / quantized data
  ├─ dynamic quant scales
  ├─ expandIdx：rankId、tokenIndex、topKIndex
  └─ 可选 expert scale
```

Kernel 使用：

```cpp
stagingTokenBytes_ = hOutAlignUbSize_;
```

写入地址为：

```text
workspace
+ (expertId × globalBS + expertOffset) × stagingTokenBytes
```

PR 早期 review 指出原 workspace 仅用于 count scratch，不能容纳完整 token staging。当前 head 已修改 Host `SetWorkSpace()`：

```text
stagingSize = localMoeExpertNum × globalBs × fullTokenBytes
```

其中 `fullTokenBytes` 按 512B 对齐的数据和最大 scale 需求保守预留。总体分配通常大于 Kernel 的真实 stride，防越界上偏安全，但可能显著增加 workspace 占用。

## 11. 最终密排

所有核完成 `LocalWindowCopy()` 后：

```cpp
ResetNextRoundCounters();
PipeBarrier<PIPE_ALL>();
SyncAll<true>();
SetExpertTokenNums();
```

这次同步确保：

- 所有 `AtomicAdd` 完成；
- expert 0 直写完成；
- expert > 0 staging 写完成；
- 后续可以读取稳定计数并重排。

`SetExpertTokenNums()` 将本地 expert 连续分给全部 AIV。每核先计算自己首个 expert 之前的计数和，再依次得到：

```text
globalOffset      = sum(counter[0:e])
expertTokenCount  = counter[e]
dstTokenIdx       = globalOffset + tokenIndexWithinExpert
```

随后完成：

1. `expertTokenNumsOut` 的 count/cumsum；
2. 最后一个 expert 所在核写 `sendCountsGlobal_` 总数；
3. expert > 0 的 staging → `expandXOut`/scale/`expandIdxOut` 重排。

## 12. 完整生产—布局—同步—消费链路

```text
发送侧：
x + expertIds
  → per-expert bitmask
  → ScalarGetSFFValue 枚举 token
  → ProcessToken 量化/封装
  → 远端 expert window
  → 同核 count/flag（纯 MoE）

接收侧：
WaitDispatch
  → AtomicAdd(counter[expert], count)
  → 得到 expertOffset
  → expert0 直写最终输出
  → expert>0 写 workspace segment

收尾：
PipeBarrier + SyncAll
  → 读取 counter[e]
  → expert 前缀和
  → staging 最终密排
  → expertTokenNumsOut / sendCountsOut
```

## 13. “免后同步”的准确边界

| 位置 | 原实现 | 当前 PR |
|---|---|---|
| 纯 MoE 发送 → 状态通知 | 全核同步后写状态 | 同核发送/写状态，可免同步 |
| Shared expert 发送 → 状态通知 | 需要同步 | 仍需要 `SyncAll` |
| WaitDispatch → LocalWindowCopy | `SyncAll` | 删除 |
| LocalWindowCopy 偏移 | 跨核前缀和 | `AtomicAdd` |
| LocalWindowCopy → 最终密排 | 无 staging 重排 | 新增 `PipeBarrier + SyncAll` |

更准确的描述是：

> 将 LocalWindowCopy 前的全核同步和跨核前缀和替换为原子槽位分配，并把同步后移到 staging 完成后的最终重排边界。

收益不仅来自 barrier 数量变化，更来自快核可以提前消费已经到达的窗口，不再等待最慢核。

## 14. 性能收益与代价

预期收益：

- 删除 `CalTokenSendExpertCnt()` 的重复前序扫描；
- 通过 bitmask 精准枚举属于某 expert 的 token；
- 纯 MoE 状态通知免除一次全核会合；
- 删除跨核 count workspace 收集和 ReduceSum；
- 不同 AIV 可按自己的远端到达进度提前 LocalWindowCopy；
- 步进分核和旋转遍历降低热点 expert/rank 冲突。

新增代价：

- 每个接收窗口块一次 GM AtomicAdd；
- 512B 间隔的 per-expert counter；
- expert > 0 一次额外 staging 写和读；
- 更大的 workspace；
- 末尾重排；
- 最终重排前仍需全核同步。

因此是否更快取决于：

```text
节省的早期 barrier、前缀和和重复扫描
    >
AtomicAdd + staging 往返 + 最终重排
```

负载偏斜和远端到达抖动越明显，越可能受益；token 很大且原同步不是瓶颈时，staging 往返可能抵消收益。

对“快慢卡”的收益边界还应补充：

- 改善的是快 AIV 等待慢 AIV 造成的空泡，不是慢卡本身的通信时间；
- status 静态分段决定了遮蔽效果，慢窗口与多少快窗口落在同一段会直接影响收益；
- 末尾 `SyncAll + SetExpertTokenNums` 仍使 kernel 总完成时刻受最慢分段约束，收益本质上是将已就绪数据的搬运前移并与长尾重叠。

## 15. 重点风险与审查项

### 15.1 两块计数器区域仅相隔 10 KiB

基址分别是 880 KiB 和 890 KiB，每个 expert 占 512B，因此不重叠容量只有：

```text
10 KiB / 512B = 20 个 expert
```

Host 当前只看到 `localMoeExpertNum × epWorldSize <= 2048` 的检查，没有显式限制 `localMoeExpertNum <= 20`。

如果本地 expert 超过 20，两块 ping-pong 区域会重叠。当前顺序又是：

```text
LocalWindowCopy 写当前区
  → ResetNextRoundCounters 清下一轮区
  → SyncAll
  → SetExpertTokenNums 读取当前区
```

区域重叠时，清下一轮区可能在最终读取前破坏当前计数。这是当前代码最需要优先确认的边界。

### 15.2 Shared expert 同步注释与代码不一致

注释说明 shared 路径应先 `PipeBarrier<PIPE_ALL>` 排空本核 MTE3，再 `SyncAll`，最后写远端 flag；但对应 `PipeBarrier` 当前被注释，仅保留 `SyncAll<true>()`。

需要确认 A5 的 `SyncAll<true>` 是否自身保证此前 MTE3 token 写完成。如果不保证，远端可能先看到 flag，再读取尚未完全落盘的窗口。

### 15.3 Host 与 Kernel 没有共享同一个 staging stride

Host 用保守 `fullTokenBytes` 分配空间，Kernel 用 `hOutAlignUbSize_` 寻址。当前更像过量分配而不是越界，但两侧没有单一契约。建议将 `stagingTokenBytes` 写入 tiling data，Host/Kernel 共用。

### 15.4 Workspace 可能大幅膨胀

```text
workspace ≈ localExpertNum × globalBs × fullTokenBytes
```

大 H、大 global BS、多本地 expert 时可能达到很大规模。应验证内存池申请、多层并发峰值和 staging GM 往返带宽。

### 15.5 同一 Expert 内 Token 顺序变化

同一 expert 内不同来源窗口块的位置由 AtomicAdd 到达顺序决定，可能不再保持固定 rank/AIV 顺序。`expandIdx` 随 token 一起重排，数学语义可能仍正确，但稳定顺序、bitwise 可复现性需要单独验证。

### 15.6 原子计数器 Cache 一致性

需要通过 AscendC 定义或实机确认：

- `SINGLE_CACHE_LINE` DCCI 是否只处理第一个 counter；
- 间隔 512B 的其他 counter 标量读取是否都能看到最终 AtomicAdd；
- `PipeBarrier + SyncAll` 与 GM atomic 全局可见性的精确关系。

编译通过不能证明这部分运行时语义。

### 15.7 `countPerDataSize` 必须非零

```cpp
countPerDataSize = tripleDataSize_ / hOutAlignUbSize_;
loopTimes = Ceil(count, countPerDataSize);
```

必须保证剩余 UB 至少容纳一个完整 token。建议在 tiling 或 kernel 中显式检查 `tripleDataSize_ >= hOutAlignUbSize_`，覆盖最大 H、量化和 mask 组合。

### 15.8 PR 混入 Token 预取

当前 diff 同时包含 `TokenPrefetchToL2Cache()`，与 PR #9347 目标重叠，不是原子定序闭环所必需。性能测试应拆分按专家发送、原子定序/staging 和 Token 预取，避免收益归因混淆。

## 16. 建议验证矩阵

| 维度 | 建议取值 |
|---|---|
| 本地 expert 数 | 1、2、8、16、20、21、最大支持值 |
| EP world size | 2、8、16、64、较大规模 |
| BS | 1、8、16、64、256、512 |
| K | 1、2、4、8 |
| shared expert | 0、1、多 shared expert |
| mask | 无、1D token mask、2D expert mask、zero-compute |
| quant | 非量化、静态、per-token、per-group、MX |
| dtype | FP16、BF16、INT8/HIFP8、FP4 |
| expertTokenNumsType | count、cumsum |
| dataState | 连续多轮 ping-pong |

功能校验重点：

- `expandXOut` 与参考实现逐 token 对应；
- `expandIdxOut` 三元组和 token 同步；
- dynamic/expert scale 同步重排；
- `expertTokenNumsOut` count/cumsum；
- `sendCountsOut`；
- 多轮计数器无残留；
- local expert 20/21 边界；
- 多次运行输出顺序和可复现性。

同步压力测试应制造来源 rank 到达速度差异和热点 expert，观察 flag 是否早于数据、AtomicAdd 区间是否重叠、staging 与重排是否竞争。

性能建议分别打点：

```text
AlltoAllDispatch
SetStatusByExpert
WaitDispatch
LocalWindowCopy
AtomicAdd 等待
末尾 SyncAll
SetExpertTokenNums 重排
```

同时统计 Kernel 总耗时、AIV 完成时间离散度、GM Atomic 冲突、staging 带宽、workspace 峰值和最终重排占比。

## 17. 当前状态与证据边界

截至 2026-08-27：

- PR 为 open、Draft/WIP；
- 当前 head 为 `43842e89`；
- GitCode 显示 CI state passed；
- 编译、UT、预冒烟和 API Check 有成功记录；
- 早期静态检查多次失败，后续记录通过但有 pre-commit warning；
- PR 与当前目标分支存在冲突；
- PR 描述没有给出性能数据；
- 没有看到 A5 实机同步时序、Atomic 冲突或 workspace 开销数据。

因此可以确认源码设计和静态执行链，但不能仅根据 CI 断言目标 A5 上的原子/cache 同步语义已经验证，也不能量化实际性能提升。

## 18. 总结

PR #9097 的核心变化是重新定义接收侧输出槽位：

```text
原方案：
全核完成 → 跨核前缀和 → 每核得到 beginIdx → 直接密排

新方案：
每核独立到达 → AtomicAdd 获得 expert 内 offset
             → expert0 直写 / 其他 expert 分段暂存
             → 末尾读取计数器 → expert 前缀和 → 最终密排
```

它把同步从 `LocalWindowCopy` 前后移到 staging 完成后的最终重排边界，使快核可以提前消费已到达的窗口；按 expert 分核也消除了重复前序扫描和纯 MoE 状态通知 barrier。

代价是 GM Atomic、完整 token workspace 往返、末尾重排，以及新的 cache 一致性和 ping-pong 地址约束。当前优先级最高的审查项是本地 expert 超过 20 时两块计数器区域的重叠问题，其次是 shared 路径 MTE3 排空语义、workspace/stride 契约和 A5 实机性能数据。

在这些问题得到验证前，该方案应视为 WIP 性能原型，而不是已经完成硬件验证的通用优化。

## 参考链接

- [PR #9097](https://gitcode.com/cann/ops-transformer/pull/9097)
- [提交 43842e89](https://gitcode.com/song-xinyi-001/ops-transformer_atomic/commit/43842e894849c97a238bcfe4a0bc72580c0f18c1)
- [Kernel 主文件](https://gitcode.com/cann/ops-transformer/blob/43842e894849c97a238bcfe4a0bc72580c0f18c1/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h)
- [Host Tiling](https://gitcode.com/cann/ops-transformer/blob/43842e894849c97a238bcfe4a0bc72580c0f18c1/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp)
- [常量定义](https://gitcode.com/cann/ops-transformer/blob/43842e894849c97a238bcfe4a0bc72580c0f18c1/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h)
