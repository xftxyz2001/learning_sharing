# MoeDistributeDispatch 系列算子的 BS 限制说明

## 1. 结论

`ops-transformer` 中的 MoE Dispatch 算子存在明确的 BS（Batch Sequence Size）限制。BS 对应输入 `x` 的第 0 维：

```text
x.shape = (BS, H)
expertIds.shape = (BS, K)
```

BS 必须大于 0，上限随算子版本、芯片型号和通信算法变化。对当前常用的 `MoeDistributeDispatchV2`，限制如下。

| 产品 | `commAlg` | BS 取值范围 | 备注 |
| --- | --- | ---: | --- |
| Atlas A2 | 默认、空字符串、`fullmesh` | `[1, 256]` | A2 Full Mesh 路径 |
| Atlas A2 | `hierarchy` | `[1, 512]` | A2 分层通信路径 |
| Atlas A3 | 默认、空字符串、`fullmesh_v1` | `[1, 512]` | 默认进入 Full Mesh V1 |
| Atlas A3 | `fullmesh_v2` | `[1, 256]` | Full Mesh V2 有额外上限 |
| Atlas A3 | `hierarchy` | `[1, 256]` | 分层通信路径 |
| Ascend 950DT | 默认、空字符串、`fullmesh_v1` | `[1, 512]` | 仅支持文档列出的通信方式 |
| Ascend 950DT | `fullmesh_v2` | `[1, 512]` | 与 A3 的 256 上限不同 |
| Ascend 950DT | `hierarchy` | 不支持 | 当前接口文档未将其列为合法输入 |

因此，不能笼统地说“Dispatch 的最大 BS 是 256”或“最大 BS 是 512”。必须同时说明算子版本、芯片和 `commAlg`。

## 2. `MoeDistributeDispatchV2` 的源码校验

### 2.1 通用路径

源码在 `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp` 中定义：

```cpp
constexpr int64_t BS_UPPER_BOUND = 512;
constexpr int64_t BS_UPPER_BOUND_LAYERED = 256;
constexpr int64_t FULLMESH_BS_UPPER_BOUND_A3 = 256;
constexpr int64_t FULLMESH_BS_UPPER_BOUND_A5 = BS_UPPER_BOUND;
```

`CheckAttrs()` 根据是否为 `hierarchy` 选择 BS 上限：

```cpp
int64_t bsUpperBound = isLayered ? BS_UPPER_BOUND_LAYERED : BS_UPPER_BOUND;
OP_TILING_CHECK((xDim0 > bsUpperBound) || (xDim0 <= 0), ...);
```

这意味着通用路径的基础约束是：

- 非 `hierarchy`：`1 <= BS <= 512`；
- `hierarchy`：`1 <= BS <= 256`。

`CheckCommAlgAttrs()` 又对 `fullmesh_v2` 增加了芯片相关限制：

```cpp
int64_t fullMeshBsUpperBound =
    GetSocVersion(context) == "Ascend950" ? 512 : 256;
```

所以：

- Atlas A3 的 `fullmesh_v2` 最大 BS 为 256；
- Ascend 950DT 的 `fullmesh_v2` 最大 BS 为 512。

### 2.2 Atlas A2 路径

Atlas A2 使用 `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch22/moe_distribute_dispatch_v2_tiling_arch22.cpp` 中的专用检查：

```cpp
constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
constexpr uint32_t LAYERED_MAX_BATCH_SIZE_A2 = 512;

uint32_t maxBatchSizeA2 = isLayered ? LAYERED_MAX_BATCH_SIZE_A2 : MAX_BATCH_SIZE_A2;
if (bs == 0 || bs > maxBatchSizeA2) {
    return GRAPH_FAILED;
}
```

因此 A2 与通用 A3 路径的 `hierarchy` 上限正好不同：

- A2 Full Mesh：最大 256；
- A2 Hierarchy：最大 512。

## 3. 其他 Dispatch 版本

### 3.1 `MoeDistributeDispatch`

| 产品 | BS 取值范围 |
| --- | ---: |
| Atlas A2 | `[1, 256]` |
| Atlas A3 | `[1, 512]` |
| Ascend 950DT | `[1, 512]` |

对应源码入口：

- 通用校验：`mc2/moe_distribute_dispatch/op_host/op_tiling/moe_distribute_dispatch_tiling.cpp`；
- A2 专用校验：`mc2/moe_distribute_dispatch/op_host/op_tiling/arch22/moe_distribute_dispatch_tiling_a2a3.cpp`；
- 接口约束：`mc2/moe_distribute_dispatch/README.md`。

### 3.2 `MoeDistributeDispatchV3`

`MoeDistributeDispatchV3` 不支持 Atlas A2，支持 Atlas A3 和 Ascend 950DT。V3 的 Host Tiling 复用 V2 的实现：

```text
MoeDistributeDispatchV3TilingFunc
  -> MoeDistributeDispatchV3TilingFuncA2A3 / A5
  -> MoeDistributeDispatchV2TilingFuncA2A3 / A5
```

由此得到：

| 产品 | `comm_alg` | BS 取值范围 |
| --- | --- | ---: |
| Atlas A3 | 默认、空字符串、`fullmesh_v1` | `[1, 512]` |
| Atlas A3 | `fullmesh_v2` | `[1, 256]` |
| Ascend 950DT | 默认、空字符串、`fullmesh_v1`、`fullmesh_v2` | `[1, 512]` |

V3 的复用关系可从以下文件确认：

- `mc2/moe_distribute_dispatch_v3/op_host/op_tiling/arch22/moe_distribute_dispatch_v3_tiling_a2a3.cpp`；
- `mc2/moe_distribute_dispatch_v3/op_host/op_tiling/arch35/moe_distribute_dispatch_v3_tiling_a5.cpp`；
- `mc2/moe_distribute_dispatch_v3/README.md`。

## 4. `globalBS` 与可变 BS 限制

BS 是当前 Rank 的实际 token 数，`globalBS` 用于描述整个 EP 通信域的容量上界。各 Rank BS 相同时，可以传：

```text
globalBS = 0
```

或者：

```text
globalBS = BS * epWorldSize
```

各 Rank BS 不同时，需要满足：

```text
globalBS = maxBS * epWorldSize
```

Host Tiling 对非零 `globalBS` 执行以下检查：

```text
globalBS >= localBS * epWorldSize
globalBS % epWorldSize == 0
```

因此，`globalBS` 不能随意填写，也不能用它绕过本卡 BS 上限：每个 Rank 的实际 `x.shape[0]` 仍然必须通过对应路径的 BS 校验。

以下组合还存在额外约束：

- 输入 `xActiveMask` 时，不支持不同 Rank 使用不同 BS；
- Atlas A3 的 `fullmesh_v2` 不支持不同 Rank 使用不同 BS；
- Atlas A3 的 `hierarchy` 不支持可变 BS；
- `fullmesh_v2` 同时还受 active mask、特殊专家等功能组合限制。

这些限制应与单纯的 `[1, maxBS]` 数值范围分开判断。

## 5. BS 合法不等于一定能够执行

即使 BS 没有超过接口上限，仍可能因为通信窗口不足而在 Tiling 阶段失败。所需空间会随以下参数共同增长：

```text
BS、maxBS、H、K、epWorldSize、localExpertNum、sharedExpertNum
```

主要检查位于：

- `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_check_win_size.h`；
- `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch22/moe_distribute_dispatch_v2_tiling_arch22.cpp`；
- `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_v2_tiling_arch35.cpp`。

因此，排查较大 BS 失败时，应依次确认：

1. `x.shape[0]` 是否在当前芯片和 `commAlg` 的硬上限内；
2. `globalBS` 是否正确表达 EP 域中的 `maxBS`；
3. 是否启用了不支持可变 BS 的功能组合；
4. `HCCL_BUFFSIZE` 或显式 CCL Buffer 是否满足当前形状需要；
5. Dispatch 与对应 Combine 算子的属性是否保持一致。

## 6. 与“BS 分核”和优化门槛的区别

接口级 BS 上限由 Host Tiling 的参数校验决定。Kernel 内部的 BS 分核、预取启用条件或其他优化阈值，只决定采用哪条实现或优化路径，不应被当成算子允许输入的最大 BS。

例如，历史实现中某个预取优化可能只在较小 active BS 下开启，但超过该优化门槛通常表示不启用该优化，并不等价于整个 Dispatch 算子拒绝执行。判断接口是否支持某个 BS，应以当前版本 Host Tiling 的失败条件为准。

## 7. 结论速查

对于 `MoeDistributeDispatchV2`：

```text
A2 + fullmesh      -> BS <= 256
A2 + hierarchy     -> BS <= 512
A3 + fullmesh_v1   -> BS <= 512
A3 + fullmesh_v2   -> BS <= 256
A3 + hierarchy     -> BS <= 256
A5 + fullmesh_v1   -> BS <= 512
A5 + fullmesh_v2   -> BS <= 512
```

同时必须保证 `BS >= 1`，并继续检查 `globalBS`、可变 BS 功能组合和通信 Buffer 容量。

---

本文结论来自 `ops-transformer` 当前源码的静态检查。它能够说明 Host Tiling 的显式限制，但不代表已经在目标 NPU、目标驱动和目标 CANN 安装包上完成运行验证。
