# 大 EP 下 Dispatch 低时延通信优化实践

本文梳理 `MoeDistributeDispatchV2` 的三项低时延优化。其中，BS 分核已在《通算融合之 `moe_distribute_dispatch_v2` 分解与调优》中展开分析，这里不再单独新建文档。

| 主题 | PR | 优化位置 | 核心手段 |
|---|---|---|---|
| [BS 分核](../通算融合之moe_distribute_dispatch_v2调优/通算融合之moe_distribute_dispatch_v2调优.md#62-token-整块优先的-bs-分核优化2) | [#6901](https://gitcode.com/cann/ops-transformer/pull/6901) | 发送侧分核与 Token 量化 | 按 BS 分配发送核，使同一 Token 尽量只读取、量化一次，再批量发送到 Top-K 专家 |
| [Token 预取](./Token预取特性及实现分析.md) | [#9347](https://gitcode.com/cann/ops-transformer/pull/9347) | 发送侧 GM 读取 | 提前执行 `GM → UB scratch`，利用共享 L2 预热 token |
| [原子定序](./A5_Dispatch_V1原子定序与免后同步实现分析.md) | [#9097](https://gitcode.com/cann/ops-transformer/pull/9097) | 发送分工与接收侧连续化 | 按 expert 发送，用 `AtomicAdd` 取代搬运前的跨核前缀和和全核会合 |

## 1. 大 EP 为什么容易出现长尾

在普通 MoE 专家卡上，接收侧 status block 数量为：

```text
rscvStatusNum = epWorldSize × localMoeExpertNum
```

每个 status block 对应“一个源 Rank 发往一个目标本地 expert”的窗口块。EP 扩大后：

- 目标卡需要观察的远端 status 更多；
- status 静态分给有限数量的 AIV，一个 AIV 可能负责多个远端窗口；
- 任意一个源 Rank 或热点 expert 的长尾都可能拉长本核的 `WaitStatus`；
- 旧实现还在 `LocalWindowCopy` 前执行全核 `SyncAll`，使已就绪的 AIV 继续等待最慢 AIV。

前三点是从代码布局得到的扩展性推断；长尾实际增长幅度仍取决于集群拓扑、路由分布、通信抖动和硬件调度，需要实测。

## 2. 三项优化分别解决什么

```text
x[BS, H]
  │
  ├─ BS 分核：按 Token 分配发送任务，复用一次读取和量化的结果
  │
  ├─ Token 预取：提前读取 x，尝试为后续多 AIV 读取预热 L2
  │
  ├─ 按 expert/token 发送到远端 window
  │
  ├─ 写 count + flag
  │
  ├─ WaitStatus：每个 AIV 等自己的 status 段全部就绪
  │
  ├─ 原子定序：AtomicAdd 为窗口块分配 expert 内槽位
  │                    快 AIV 可以提前 LocalWindowCopy
  │
  └─ 末尾 SyncAll + 多 expert 密排
```

BS 分核减少同一 Token 面向 Top-K 专家的重复读取和量化；Token 预取尝试降低后续 GM 读取开销；原子定序减少接收侧快 AIV 等待慢 AIV 的空泡。三者作用阶段不同，可以叠加，但性能收益应分开归因。

## 3. 原子定序的精确语义

```text
错误简化：一个窗口一到 → 立即搬运

实际语义：一个 AIV 负责的 status 段全部到达
          → 该 AIV 为这一段窗口执行 AtomicAdd
          → 搬运这一段数据
```

`AtomicAdd` 不是到达通知。它的作用是在不依赖其他 AIV count 的情况下，为同一 expert 的并发窗口块分配互不重叠的区间。

## 4. 收益与边界

| 能力 | Token 预取 | 原子定序 |
|---|---|---|
| 降低发送侧首次 GM miss | 可能 | 否 |
| 允许快 AIV 跳过慢 AIV 提前搬运 | 否 | 是 |
| 消除慢卡 | 否 | 否 |
| 保证 L2 hit | 否，best-effort | 不涉及 |
| 需要末尾全核会合 | 不新增 | 仍需要 |
| 主要新成本 | 多一次 GM 读取和一个 token UB | GM Atomic、staging 往返、最终重排 |

原子定序改善的是“快 AIV 等慢 AIV”的空泡，不是慢卡本身。最终 kernel 仍需等到最慢分段完成，其收益来自将已就绪数据的搬运与长尾等待重叠。

## 5. 性能归因建议

PR #9097 的单个提交同时包含普通 Dispatch Token 预取、按 expert 分核发送和原子定序/staging。因此不应将该分支相对 baseline 的所有收益归因于 AtomicAdd。

建议使用以下拆分对比：

```text
baseline
  + Token 预取
  + 按 expert 分核发送
  + 原子定序/staging
  + 上述组合
```

目标 A5 实测至少应分段记录 `AlltoAllDispatch`、`WaitDispatch`、`LocalWindowCopy`、末尾 `SyncAll` 和 `SetExpertTokenNums`，并同时观察 L2 命中、MTE2 利用率、GM Atomic 冲突、staging 带宽与 AIV 完成时间离散度。

## 6. 证据边界

当前文档基于 PR head 静态源码、PR 信息和已有 CI 记录，可以确认调用链、地址布局和显式同步关系。尚未在目标 A5 集群上重现性能数据，因此不将缓存驻留、GM Atomic 可见性或快慢卡收益幅度表述为已完成实机验证的结论。
