# CANN 9.1.0：`CrossCoreSetFlag` / `CrossCoreWaitFlag` 与 `SyncAll` 对比

> 适用基线：CANN 9.1.0（本文按华为昇腾社区当前公开的 CANN 9.1.0-beta.3 Ascend C API 文档整理）。
> 相关产品：Atlas A2、Atlas A3、Atlas 350 等分离模式产品；具体接口支持情况必须以目标产品文档和实际安装的 CANN 头文件为准。

## 1. 结论先行

- `SyncAll` 是高层的**集合式全核屏障（barrier）**：所有参与核都到达同步点后，所有核才能继续。
- `CrossCoreSetFlag` / `CrossCoreWaitFlag` 是更底层的**发信号—等待信号（signal/wait）**机制：可以实现全核屏障，也可以只同步某些核，例如 AIC → AIV。
- `modeId` 决定“**谁和谁同步**”，即同步拓扑。
- `flagId` 决定“**使用哪一个同步通道/计数器**”，它不是核 ID，也不是 block ID。
- CANN 9.1.0 明确说明：硬同步版 `SyncAll` 内部使用 `CrossCoreSetFlag`，占用内部 `flagId` 范围 `[11, 14]`。官方不建议在同一实现中混用硬同步 `SyncAll` 和手工 CrossCore flag。

可将二者概括为：

```text
SyncAll
  = 封装好的全体到齐屏障

CrossCoreSetFlag / CrossCoreWaitFlag
  = 可自行编排同步对象、方向、流水和同步阶段的底层原语
```

## 2. 接口原型

### 2.1 CrossCore flag

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

两者必须配合使用：

- `CrossCoreSetFlag`：发送同步信号；
- `CrossCoreWaitFlag`：等待对应同步信号，未收到信号时阻塞后续指令。

### 2.2 SyncAll

软同步：

```cpp
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(
    const GlobalTensor<int32_t>& gmWorkspace,
    const LocalTensor<int32_t>& ubWorkspace,
    const int32_t usedCores = 0);
```

硬同步：

```cpp
template <
    bool isAIVOnly = true,
    const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>
__aicore__ inline void SyncAll();
```

## 3. 对比

| 对比项 | `CrossCoreSetFlag` / `CrossCoreWaitFlag` | `SyncAll` |
|---|---|---|
| 抽象层次 | 底层同步原语 | 高层全核屏障 |
| 基本语义 | 指定对象发送信号，另一侧等待信号 | 所有参与核都到齐后一起继续 |
| 同步范围 | 由 `modeId` 精确控制 | 全部核或软同步的 `usedCores` 个核 |
| 同步方向 | 可做单向 AIC → AIV，也可组成双向屏障 | 集合式、对称式 barrier |
| 流水控制 | 通过模板参数 `pipe` 指定指令所在流水 | Atlas 350 的硬同步可通过 `SyncAllConfig` 配置触发/等待流水 |
| 是否管理 `flagId` | 开发者需要管理 | API 自动管理；硬同步内部占用 `[11,14]` |
| Workspace | 不需要用户提供 GM/UB 同步空间 | 软同步需要 GM/UB workspace；硬同步不需要 |
| 灵活性 | 高 | 较低，语义固定但使用简单 |
| 典型用途 | AIC 完成后通知 AIV；两个 AIV 局部同步；定制流水同步 | 多核写完共享 GM 后，统一进入下一阶段 |
| 主要风险 | Set/Wait 不成对、模式或 ID 错误、与高阶 API 冲突 | 参与核未全部调度、`numBlocks` 配置错误、与手工 flag 冲突 |

## 4. `modeId` 的含义

`modeId` 是编译期模板参数，用来选择同步拓扑，而不是同步资源编号。

### 4.1 `modeId = 0`：不同 AI Core 之间同步

对同类计算核做全核同步：

- AIC 场景：等待所有参与的 AIC；
- AIV 场景：等待所有参与的 AIV。

所有参与核都执行到对应的 `CrossCoreSetFlag` 后，`CrossCoreWaitFlag` 后面的指令才继续。因此，模式 0 的一组 Set + Wait 最接近 `SyncAll`。

```text
AIV(core 0) ── Set ── Wait ──┐
AIV(core 1) ── Set ── Wait ──┼── 全部到齐后继续
AIV(core 2) ── Set ── Wait ──┘
```

### 4.2 `modeId = 1`：同一个 AI Core 内两个 AIV 同步

只同步当前 AI Core 内的 AIV0 与 AIV1，不等待其他 AI Core：

```text
AI Core N
  AIV0 ──┐
         ├── 局部同步
  AIV1 ──┘
```

### 4.3 `modeId = 2`：同一个 AI Core 内 AIC 与两个 AIV 同步

这是 AIC 与 AIV 的 1:2 同步关系：

```text
          ┌── AIV0
AIC ──────┤
          └── AIV1
```

- AIC 执行 Set 后，两个 AIV 的等待可以解除；
- 反方向同步时，两个 AIV 都执行 Set 后，AIC 的等待才能解除。

### 4.4 `modeId = 4`：AIC 与单个 AIV 的 1:1 同步

该模式仅支持 **Atlas 350 加速卡**。

```text
AIC ↔ AIV0
AIC ↔ AIV1
```

AIV0、AIV1 可以分别触发 AIC 的等待，适合不希望把两个 AIV 绑定成 1:2 同步的场景。

## 5. `flagId` 的含义

`flagId` 是同步标记/同步通道编号。每个同步标记对应一个用于控制同步的计数器。

可用下面的伪代码理解其行为：

```text
初始：counter[flagId] = 0

CrossCoreSetFlag(flagId)：
    向对应同步通道发送信号

CrossCoreWaitFlag(flagId)：
    如果没有对应信号，则阻塞后续指令
    收到对应信号后，继续执行
```

因此 `flagId` 不是：

- 物理核编号；
- `GetBlockIdx()` 返回的逻辑 block 编号；
- `modeId`；
- 流水编号。

### 5.1 Atlas A2 / A3

手工调用 CrossCore flag 时，文档规定的 `flagId` 范围为：

```text
0 ～ 10
```

不同的独立同步阶段应避免同时复用同一个 `flagId`。同一 `flagId` 的计数器最多设置 15 次。

### 5.2 Atlas 350 的方向映射

Atlas 350 为支持 AIC 与 AIV0/AIV1 分别同步，发送端和等待端看到的数字不一定相同：

| 发送方 | `CrossCoreSetFlag` 的 `flagId` | 接收方 | `CrossCoreWaitFlag` 的 `flagId` |
|---|---:|---|---:|
| AIV0 | 0～10 | AIC | 0～10 |
| AIV1 | 0～10 | AIC | 16～26 |
| AIC | 0～10 | AIV0 | 0～10 |
| AIC | 16～26 | AIV1 | 0～10 |

这说明 `flagId` 是硬件同步通道的寻址规则，不能简单理解成“发送方和接收方永远填写相同数字”。

## 6. CrossCore flag 示例

以下示例沿用 CANN 9.1.0 API 文档的调用形式。实际代码必须结合目标芯片、Kernel 类型和流水安排验证。

### 6.1 模式 0：同步所有 AIV

```cpp
if (g_coreType == AscendC::AIV) {
    // 当前 AIV 的 MTE3 阶段完成后发送同步信号
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);

    // 等待所有参与的 AIV 到达该同步阶段
    AscendC::CrossCoreWaitFlag(0x8);

    // 所有 AIV 到齐后才能执行这里
}
```

其整体效果接近一个 AIV 全核 barrier，但同步协议和 `flagId` 由开发者维护。

### 6.2 模式 1：同步当前 AI Core 内的两个 AIV

```cpp
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);

    // 当前 AI Core 内的两个 AIV 都到达后继续
}
```

该同步不会等待其他 AI Core 内的 AIV。

### 6.3 模式 2：AIC 完成后通知 AIV

典型场景：AIC 完成 Matmul/Fixpipe，把结果写入 GM；AIV 等待结果就绪后再搬入 UB 做后处理。

```cpp
// AIC 侧
if (g_coreType == AscendC::AIC) {
    // Matmul / Fixpipe 工作
    // ...

    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
}

// AIV 侧
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreWaitFlag(0x8);

    // AIC 结果已经就绪，开始 Vector 后处理
    // ...
}
```

这是定向的生产者—消费者同步，不要求所有不同 AI Core 都到达一个全局 barrier。

> 注意：Matmul 高阶 API 自身会使用 CrossCore flag。若实际使用 Matmul 高阶 API，通常不应再手工插入上述同步；应先确认高阶 API 已经完成哪些 AIC/AIV 同步。

### 6.4 两个独立同步阶段使用不同 `flagId`

```cpp
// 阶段一
AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(8);
AscendC::CrossCoreWaitFlag(8);

// 阶段二：使用另一个 ID，避免与尚未完全消费的阶段一信号混淆
AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(9);
AscendC::CrossCoreWaitFlag(9);
```

这只是资源规划示例；是否可以复用某个 ID，取决于上一轮 Set/Wait 是否已经严格配平，以及是否与 Matmul、硬同步 `SyncAll` 等内部实现冲突。

## 7. SyncAll 示例

### 7.1 软同步

软同步需要：

- 所有核共享的 `GlobalTensor<int32_t>`；
- 每个核独享的 `LocalTensor<int32_t>`；
- GM workspace 预先初始化为 0；
- GM 和 UB workspace 大小均不小于 `核数 × 32 Bytes`。

```cpp
// 每个核先把自己的结果写入共享工作区
AscendC::DataCopy(
    workGlobal[blockIdx * perBlockSize],
    localResult,
    perBlockSize);

// workLocal 来自本核的 LocalTensor
AscendC::SyncAll(syncGlobal, workLocal);

// 此处所有参与核均已完成前面的写入
ReadOrAccumulateResultsFromAllCores();
```

指定部分核参与软同步：

```cpp
constexpr int32_t usedCores = 8;
AscendC::SyncAll(syncGlobal, workLocal, usedCores);
```

`usedCores` 不能超过 Kernel 启动时指定的逻辑 `numBlocks`。不传或传 0 表示全核软同步。

### 7.2 纯 Vector 场景的硬同步

```cpp
// 每个 AIV 完成前一阶段工作
// ...

AscendC::SyncAll<true>();

// 所有参与的 Vector 核完成同步后继续
// ...
```

硬同步不需要 GM/UB 同步 workspace，分离模式下性能通常优于软同步。

### 7.3 Cube + Vector 融合场景的硬同步

```cpp
AscendC::SyncAll<false>();
```

`isAIVOnly=false` 的含义是：

1. 分别完成 Vector 核之间、Cube 核之间的全核同步；
2. 再完成 Cube 与 Vector 之间的同步。

该能力不适用于软同步原型；软同步仅适用于纯 Vector 场景。

### 7.4 Atlas 350 的流水配置

CANN 9.1.0 的硬同步原型允许通过 `SyncAllConfig` 指定：

- `triggerPipe`：在哪些流水完成后触发同步；
- `waitPipe`：哪些流水等待同步结果。

默认配置为：

```cpp
DEFAULT_SYNC_ALL_CONFIG = {PIPE_ALL, PIPE_ALL};
```

自定义 `SyncAllConfig` 当前仅 Atlas 350 支持。使用时应以目标 CANN 9.1.0 安装包中 `SyncAllConfig` 的实际声明为准，不应把 Atlas 350 的配置方式移植到 A2/A3。

## 8. 如何选择

### 选择 `SyncAll`

适合以下场景：

- 所有参与核都必须完成阶段 A，才能进入阶段 B；
- 多个核先写共享 GM，然后所有核统一读取；
- 不需要定向同步，只需要标准 barrier；
- 希望由 API 管理底层 flag。

例如：

```text
所有核写 workGm
       ↓
    SyncAll
       ↓
所有核读取/归约 workGm
```

### 选择 CrossCore flag

适合以下场景：

- AIC 只需要通知本 AI Core 内的 AIV；
- 只同步 AIV0 与 AIV1；
- Atlas 350 上需要 AIC↔AIV0、AIC↔AIV1 的独立 1:1 同步；
- 需要明确指定产生信号的硬件流水；
- 全核 barrier 会造成不必要的等待。

## 9. 重要约束与常见错误

### 9.1 Set/Wait 不配对导致永久等待

如果对应的 Set 没有执行，`CrossCoreWaitFlag` 后续指令会一直阻塞，最终表现为 Kernel 超时或卡死。

需要重点检查：

- 条件分支是否导致部分核跳过 Set；
- Set 和 Wait 是否使用了正确的模式与 ID 映射；
- 某些逻辑 block 是否根本没有被调度；
- AIC、AIV 的 Kernel 类型配置是否符合实际场景。

### 9.2 与 Matmul 高阶 API 的 flag 冲突

Matmul 高阶 API 内部会使用 CrossCore flag。若定义了 `N` 个 Matmul 对象，其内部占用范围为：

```text
[0, 2 × N - 1]
```

Matmul 最多支持 4 个对象，此时可能占用：

```text
[0, 7]
```

所以手工选择 `flagId=8` 并不是任意示例：它通常是为了避开常见的 Matmul 内部范围，但仍必须结合整个 Kernel 中所有高阶 API 的实际占用统一规划。

### 9.3 与硬同步 SyncAll 冲突

硬同步版 `SyncAll` 内部占用：

```text
flagId [11, 14]
```

官方明确不建议同时手工使用 `CrossCoreSetFlag` 和硬同步 `SyncAll`。即使目标产品公开给手工调用的范围看起来不同，也不应据此忽略该约束。

### 9.4 全核同步的调度死锁

模式 0 和 `SyncAll` 都需要全部参与核能够到达同步点。如果出现以下组合，可能发生调度死锁：

- 多流并发；
- 两个或更多同步算子并发；
- 所有并发算子的请求核数总和超过物理核数；
- 多个并发算子都执行全核同步。

典型死锁过程：

```text
算子 A 的部分核已运行并在 barrier 等待
                 ↓
算子 A 的剩余核尚未获得物理核
                 ↓
物理核被算子 B 的部分同步核占用
                 ↓
算子 A、B 都等不到各自剩余核
```

CANN 9.1.0 文档建议开启 batchmode，让此类同步算子独占所需核资源：

- Kernel 直调：使用 `__schedmode__(mode)`；
- 工程化算子：使用 `TilingContext::SetScheduleMode`。

### 9.5 `numBlocks` 不能超过实际可同时运行的核数

使用 `SyncAll` 时，如果逻辑 `numBlocks` 超过实际运行该算子的处理器核数，框架可能进行多轮调度。第一轮核进入 barrier 后会等待尚未调度的核，造成 Kernel 卡死。

### 9.6 `pipe` 不是参与核范围

`pipe` 指定同步指令所在的硬件流水，用于保证相关流水上的操作与同步点之间具有正确顺序；它不决定参与多少核。

需要分别回答两个问题：

```text
modeId：哪些核参与、同步拓扑是什么？
pipe：哪个硬件流水发出或等待同步？
```

CANN 9.1.0 API 文档特别注明：显式使用 `PIPE_S` 作为该模板流水类型仅 Atlas 350 支持。官方 A2/A3 示例中的 `CrossCoreWaitFlag(flagId)` 使用接口默认形式；跨产品移植时，应以目标产品文档及实际 CANN 头文件为准。

## 10. 推荐检查清单

使用 CrossCore flag 前：

- [ ] 确认目标芯片支持该接口和所选 `modeId`；
- [ ] 明确同步拓扑是全核、AIV↔AIV、AIC↔双 AIV，还是 Atlas 350 的 1:1；
- [ ] 为每个独立同步阶段规划 `flagId`；
- [ ] 排查 Matmul 等高阶 API 的 flag 占用；
- [ ] 不与硬同步 `SyncAll` 混用；
- [ ] 确保所有等待路径都一定存在对应 Set；
- [ ] 模式 0 下评估并发调度和 batchmode。

使用 `SyncAll` 前：

- [ ] 确认需要的确实是全核 barrier；
- [ ] 确保每个参与核都会执行到同一个同步点；
- [ ] 确保逻辑 `numBlocks` 不超过可同时运行的核数；
- [ ] 软同步正确申请并清零 GM workspace；
- [ ] 软同步正确申请每核 UB workspace；
- [ ] 分离模式优先评估硬同步；
- [ ] 评估多流并发死锁风险并配置 batchmode。

## 11. 官方参考

- [CANN 9.1.0-beta.3：CrossCoreSetFlag](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_0273.html)
- [CANN 9.1.0-beta.3：CrossCoreWaitFlag](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_0274.html)
- [CANN 9.1.0-beta.3：SyncAll](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_0204.html)
