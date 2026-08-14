# ProcessGroupHCCL 创建 HCCL 通信域的两条路径

[TOC]

[pytorch:基于昇腾NPU的PyTorch框架适配插件项目 - AtomGit](https://gitcode.com/Ascend/pytorch)

## 1. 文档范围与结论

本文依据以下当前源码静态整理：
- 核心文件：`torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`

本文讨论普通 HCCL collective 和 P2P 在首次需要 communicator 时的创建过程，不把 `ProcessGroupHCCL` C++ 对象的构造等同于 HCCL communicator 的创建。正常情况下，communicator 是懒创建的：首次 collective/P2P 查询缓存未命中时才创建。`HCCL_ZERO_COPY` 构造期创建的专用 communicator 不属于本文两条主路径。

两条路径不是随机并列选择，而是严格的“先 RankTable，失败再 RootInfo”关系：

```text
collective / P2P 首次需要 HCCL communicator
                    │
                    ▼
       getHCCLComm(devicesKey, devices, ...)
                    │
                    ├── devHCCLCommMap_ 命中 ──► 直接复用
                    │
                    └── 缓存未命中
                            │
                            ▼
                    createHCCLComm(...)
                            │
                            ▼
              createHCCLCommEx(...)      RankTable 路径
                            │
                   ┌────────┴────────┐
                   │ true            │ false
                   ▼                 ▼
              创建完成       createHCCLCommOrigin(...)  RootInfo 路径
                   │                 │
                   └────────┬────────┘
                            ▼
       保存到 devHCCLCommMap_[devicesKey]，后续复用
```

```cpp
// 摘录并添加注释：非完整可编译代码
std::vector<std::shared_ptr<HCCLComm>>& ProcessGroupHCCL::createHCCLComm(...)
{
    // 为每个 device 预留一个 communicator 包装对象。
    std::vector<std::shared_ptr<HCCLComm>> hcclComms;
    hcclComms.resize(devices.size());
    std::vector<c10_npu::NPUStream> streamVal;

    // [第一选择] 尝试 RankTable/父通信域派生路径。
    // true 表示已经创建成功，不再进入 RootInfo。
    if (!createHCCLCommEx(
            devicesKey, devices, commType, commConfig,
            hcclComms, streamVal, p2pRank)) {
        // [兜底] RankTable 不可用或创建失败时，走 RootInfo。
        createHCCLCommOrigin(
            devicesKey, devices, commType, commConfig,
            hcclComms, streamVal, p2pRank);
    }

    // 两条路径最终汇合：保存 stream/event/communicator 缓存。
    hcclStreams_.emplace(devicesKey, std::move(streamVal));
    devHCCLCommMap_.emplace(devicesKey, std::move(hcclComms));
    return devHCCLCommMap_[devicesKey];
}
```

### 1.1 文中“最终 HCCL 接口”的层级说明

`HCCLUtils.cpp` 中实际写的是小写包装函数，随后 `HcclCompile.h` 通过 `GET_FUNC(...)` 从当前加载的 HCCL 动态库解析大写接口符号：

```text
HCCLComm::createGlobalHcclComm
  └─ hcclCommInitClusterInfoConfig(...)       torch_npu 包装函数
       └─ GET_FUNC(HcclCommInitClusterInfoConfig)
            └─ HcclCommInitClusterInfoConfig  HCCL 动态库接口

HCCLComm::createSubHcclComm
  └─ hcclCreateSubCommConfig(...)             torch_npu 包装函数
       └─ GET_FUNC(HcclCreateSubCommConfig)
            └─ HcclCreateSubCommConfig        HCCL 动态库接口

HCCLComm::create_config
  └─ hcclCommInitRootInfoConfig(...)          torch_npu 包装函数
       └─ GET_FUNC(HcclCommInitRootInfoConfig)
            └─ HcclCommInitRootInfoConfig     HCCL 动态库接口
```

包装层没有重新计算这些参数，只把收到的 `nRanks/rootInfo/rank/config/comm` 等参数原样传给动态库函数指针。因此本文的参数追溯一直追到大写 HCCL 动态库接口。

---

## 2. 创建通信域所需的公共元数据来自哪里

### 2.1 `rank_`、`size_`、rank 列表和 group id

Python HCCL backend creator 位于 `torch_npu/__init__.py`：

```python
# 摘录并添加注释
def _new_process_group_hccl_helper(dist_backend_opts, pg_options):
    store = dist_backend_opts.store

    # 当前进程在“当前组”内的 rank，不一定是 global rank。
    group_rank = dist_backend_opts.group_rank

    # 当前组的大小；默认组通常等于 WORLD_SIZE，子组等于 len(ranks)。
    group_size = dist_backend_opts.group_size

    # 子组保存其成员的 global rank 列表；默认组为空列表。
    pg_options.global_ranks_in_group = dist_backend_opts.global_ranks_in_group

    # PyTorch 为当前 PG 生成的 group_name/group_id。
    pg_options.group_id = dist_backend_opts.group_id

    return torch_npu._C._distributed_c10d.ProcessGroupHCCL(
        store, group_rank, group_size, pg_options
    )
```

进入 C++ 构造函数后，`rank` 和 `size` 传给父类 `c10d::Backend`：

```cpp
ProcessGroupHCCL::ProcessGroupHCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,   // Python dist_backend_opts.group_rank
    int size,   // Python dist_backend_opts.group_size
    c10::intrusive_ptr<Options> options)
    : c10d::Backend(rank, size),  // 保存为父类 rank_、size_
      store_(store),
      options_(c10::make_intrusive<Options>(*options.get()))
{}
```

因此后面的：

```cpp
getRank()  -> rank_ -> 当前进程在当前 PG 内的 rank
getSize()  -> size_ -> 当前 PG 的 rank 数量
```

默认组和子组的来源不同：

| 场景 | `group_rank` | `group_size` | `global_ranks_in_group` |
|---|---:|---:|---|
| `init_process_group` 默认组 | 当前 global rank | `world_size`，通常来自 `WORLD_SIZE` | `[]` |
| `new_group(ranks=[...])` 子组 | 当前 global rank 在 `ranks` 中的下标 | `len(ranks)` | Python `ranks` 的 global rank 列表 |

例如默认组为 `[0,1,2,3]`，子组为 `[0,2]`：

```text
global rank 0 ──► 子组 group_rank=0
global rank 2 ──► 子组 group_rank=1

子组 group_size=2
global_ranks_in_group=[0,2]
```

### 2.2 communicator 创建发生在缓存未命中时

```cpp
std::vector<std::shared_ptr<HCCLComm>>& ProcessGroupHCCL::getHCCLComm(...)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (devHCCLCommMap_.find(devicesKey) != devHCCLCommMap_.end()) {
        // 已存在：直接复用，不再调用任何 HCCL init 接口。
        return devHCCLCommMap_[devicesKey];
    }

    // 未命中：此处才真正进入 communicator 创建流程。
    return createHCCLComm(devicesKey, devices, commType, commConfig, p2pRank);
}
```

---

## 3. 路径一：RankTable

入口为 `ProcessGroupHCCL::createHCCLCommEx()`。

### 3.1 RankTable 路径的前置条件

```text
读取环境变量 RANK_TABLE_FILE
          │
          ├── 为空或文件不可读 ─────────────► return false，转 RootInfo
          │
          ▼
检查当前 HCCL 动态库是否导出
HcclCommInitClusterInfoConfig
          │
          ├── 不存在 ──────────────────────► return false，转 RootInfo
          │
          ▼
按默认组/子组/P2P 分支创建 communicator
```

关键源码：

```cpp
// GetRankTableFilePath() 最终读取 getenv("RANK_TABLE_FILE")。
std::string rankTableFile =
    c10_npu::option::OptionsManager::GetRankTableFilePath();

// 未配置或文件不可读：RankTable 路径不可用。
if (rankTableFile.empty() || !checkFilePathReadable(rankTableFile)) {
    return false;
}

// 通过 dlsym/GET_FUNC 检查当前 HCCL 库是否包含目标接口。
if (!hcclCommInitClusterInfoConfigExist()) {
    return false;
}
```

`OptionsManager.cpp` 给出了文件路径的最终来源：

```cpp
std::string OptionsManager::GetRankTableFilePath()
{
    // 最终来源：进程环境变量 RANK_TABLE_FILE。
    char* rank_table_file = get_and_log_env("RANK_TABLE_FILE");
    return rank_table_file != nullptr ? std::string(rank_table_file) : "";
}
```

### 3.2 默认组：最终调用 `HcclCommInitClusterInfoConfig`

默认组普通 collective 的判断条件：

```cpp
options_->global_ranks_in_group.empty() &&
commType == HcclCommType::DEFAULT
```

流程图：

```text
RANK_TABLE_FILE
      │
      ▼
createHCCLCommEx()
      │
      ├── global_ranks_in_group == []
      └── commType == DEFAULT
               │
               ▼
对 devices[i] 设置当前 NPU device
               │
               ├── rank = getRank() * devices.size() + i
               ├── config = 外部 commConfig 或 createHcclCommConfigWithOptions()
               └── clusterInfo = rankTableFile.c_str()
               │
               ▼
HCCLComm::createGlobalHcclComm(...)
               │
               ▼
hcclCommInitClusterInfoConfig(
    clusterInfo, rank, config, &hcclComm_)
```

源码来自 `ProcessGroupHCCL.cpp` 和 `HCCLUtils.cpp`：

```cpp
// ProcessGroupHCCL.cpp
if (options_->global_ranks_in_group.empty() &&
    commType == HcclCommType::DEFAULT) {
    for (size_t i = 0; i < devices.size(); ++i) {
        // 当前实现为组内 rank 与 device 下标组合出的 HCCL rank。
        // 常见的一进程一 NPU 情况 devices.size()==1，因此等于 getRank()。
        int rank = getRank() * static_cast<int>(devices.size())
                 + static_cast<int>(i);

        npuGuard.set_index(devices[i].index());

        HcclCommConfig config;
        if (commConfig == nullptr) {
            // 没有调用方专用配置时，由 ProcessGroupHCCL options 和环境变量生成。
            config = createHcclCommConfigWithOptions();
            commConfig = &config;
        }

        // rankTableFile 是文件路径，不是 torch_npu 预先读取后的 JSON 内容。
        auto comm = HCCLComm::createGlobalHcclComm(
            rankTableFile.c_str(), rank, commConfig);
        hcclComms[i] = comm;
    }
}

// HCCLUtils.cpp
std::shared_ptr<HCCLComm> HCCLComm::createGlobalHcclComm(
    const char* clusterInfo, // 实际值：RANK_TABLE_FILE 指向的路径
    uint32_t rank,           // 实际值：上面计算的 rank
    HcclCommConfig* config)  // 实际值：外部配置或 PG 生成配置
{
    auto comm = std::make_shared<HCCLComm>();

    // 最终 HCCL 接口。
    if (hcclCommInitClusterInfoConfig(
            clusterInfo,
            rank,
            config,
            &(comm->hcclComm_)) != HCCL_SUCCESS) {
        return nullptr;
    }
    return comm;
}
```

#### `HcclCommInitClusterInfoConfig` 参数最终来源

最终调用原型：

```cpp
HcclCommInitClusterInfoConfig(
    const char* clusterInfo,
    uint32_t rank,
    HcclCommConfig* config,
    HcclComm* comm);
```

| 最终形参 | ProcessGroupHCCL 中的实参 | 最终来源 |
|---|---|---|
| `clusterInfo` | `rankTableFile.c_str()` | 环境变量 `RANK_TABLE_FILE` 的字符串值；torch_npu 先检查路径可读，然后把路径交给 HCCL |
| `rank` | `getRank() * devices.size() + i` | `getRank()` 来自创建当前 PG 时传入的 `group_rank`；`i` 来自当前 `devices` 遍历下标 |
| `config` | `commConfig` | 如果调用方传入则直接使用；否则来自 `createHcclCommConfigWithOptions()` |
| `comm` | `&(comm->hcclComm_)` | `HCCLComm::createGlobalHcclComm()` 新建的 `HCCLComm` 包装对象内部输出槽位，由 HCCL 写入原生句柄 |

> 名称虽然叫 `clusterInfo`，但当前 torch_npu 实际传入的是 RankTable 文件路径 `rankTableFile.c_str()`。

### 3.3 子组：最终调用 `HcclCreateSubCommConfig`

当以下默认组普通 collective 条件不成立时，`createHCCLCommEx()` 进入“从全局 communicator 派生子 communicator”分支：

```cpp
!(options_->global_ranks_in_group.empty() &&
  commType == HcclCommType::DEFAULT)
```

对普通 `dist.new_group(ranks=[...])`，其 `global_ranks_in_group` 非空；对 P2P，即使在默认 PG 上也会进入该分支。

```text
createHCCLCommEx()
        │
        ├── 检查 HcclCreateSubCommConfig 是否存在
        ├── 检查 global_ 默认 ProcessGroupHCCL 是否存在
        │
        ▼
global_->getHcclCommByDevices(devices)
        │
        ├── 默认组 communicator 已缓存：直接取
        └── 未缓存：递归触发默认组 RankTable communicator 创建
        │
        ▼
准备子组参数
        ├── rankNum       = getSize()；P2P 时为 2
        ├── rankIds       = global_ranks_in_group；P2P 时为两端 global rank
        ├── subCommId     = hash(group_id)；P2P 时为 hash(devicesKey)
        ├── subCommRankId = 当前组内 rank；P2P 时为 p2pRank
        └── config        = PG 配置；P2P 使用 P2P buffer 配置
        │
        ▼
HcclCreateSubCommConfig(parentComm, ...)
        │
        ▼
从默认组 communicator 派生子 communicator
```

带注释源码，来自 `ProcessGroupHCCL.cpp`：

```cpp
// 必须由当前 HCCL 动态库提供子通信域派生接口。
if (!hcclCreateSubCommConfigExist()) {
    return false; // 随后转 RootInfo
}

// global_ 在默认 PG 构造时被设置为 this。
if (global_ == nullptr) {
    return false;
}

// 父通信域：默认 PG 对相同 devices 的 communicator。
// 如果尚未创建，这个调用会触发默认 PG 的懒创建。
std::shared_ptr<HCCLComm> globalHcclComm =
    global_->getHcclCommByDevices(devices);

// 普通子组的 subCommId：对 PyTorch group_id 字符串求 std::hash。
uint64_t hcclid = std::hash<std::string>{}(options_->group_id);

for (size_t i = 0; i < devices.size(); ++i) {
    // 普通子组：当前子组的 rank 数。
    int numRanks = getSize();

    // 普通子组：当前进程在当前子组内的 rank。
    int rank = getRank() * static_cast<int>(devices.size())
             + static_cast<int>(i);

    HcclCommConfig config;
    if (commConfig == nullptr) {
        config = createHcclCommConfigWithOptions();
        if (commType == HcclCommType::P2P) {
            // P2P 特例覆盖普通子组参数。
            numRanks = 2;
            rank = p2pRank;
            config.hcclBufferSize = OptionsManager::GetP2PBufferSize();
        }
        commConfig = &config;
    }

    std::shared_ptr<HCCLComm> subComm;
    if (commType == HcclCommType::P2P) {
        // P2P 特例：rankIds 只包含通信两端；subCommId 改为 devicesKey 的 hash。
        std::vector<uint32_t> p2pRanks = /* 两端的 global rank */;
        hcclid = std::hash<std::string>{}(devicesKey);
        subComm = HCCLComm::createSubHcclComm(
            globalHcclComm, numRanks, p2pRanks.data(),
            hcclid, rank, commConfig);
    } else {
        // 普通 new_group 子组：rankIds 来自 Python ranks 列表。
        subComm = HCCLComm::createSubHcclComm(
            globalHcclComm,
            numRanks,
            options_->global_ranks_in_group.data(),
            hcclid,
            rank,
            commConfig);
    }
    hcclComms[i] = subComm;
}
```

最终封装位于 `HCCLUtils.cpp`：

```cpp
std::shared_ptr<HCCLComm> HCCLComm::createSubHcclComm(
    std::shared_ptr<HCCLComm> comm, // 父/默认组 communicator
    uint32_t rankNum,
    uint32_t* rankIds,
    uint64_t subCommId,
    uint32_t subCommRankId,
    HcclCommConfig* config)
{
    auto subComm = std::make_shared<HCCLComm>();

    // 最终 HCCL 接口。
    if (hcclCreateSubCommConfig(
            &(comm->hcclComm_),
            rankNum,
            rankIds,
            subCommId,
            subCommRankId,
            config,
            &(subComm->hcclComm_)) != HCCL_SUCCESS) {
        return nullptr;
    }
    return subComm;
}
```

#### `HcclCreateSubCommConfig` 参数最终来源

| 最终形参 | 普通 `new_group` 子组来源 | P2P 特例来源 |
|---|---|---|
| 父 `comm` | `global_->getHcclCommByDevices(devices)` 返回的默认 PG communicator | 相同 |
| `rankNum` | `getSize()`，即当前子组 `len(ranks)` | 固定为 `2` |
| `rankIds` | `options_->global_ranks_in_group.data()`，最终来自 Python `new_group(ranks=...)` | `{lowRank, highRank}`；若当前 PG 本身是子组，再映射到该子组成员对应的 global rank |
| `subCommId` | `std::hash<std::string>{}(options_->group_id)` | `std::hash<std::string>{}(devicesKey)` |
| `subCommRankId` | `getRank() * devices.size() + i` | `p2pRank`，即两端通信域中的 0/1 |
| `config` | 外部 `commConfig` 或 `createHcclCommConfigWithOptions()` | 同左，但内部生成配置时把 buffer size 改为 `P2P_HCCL_BUFFSIZE` |
| 输出 `subComm` | `&(subComm->hcclComm_)` | 相同，由 HCCL 写入原生子 communicator 句柄 |

### 3.4 RankTable 路径何时返回 false

以下任一情况会回退到 RootInfo：

1. `RANK_TABLE_FILE` 为空。
2. RankTable 文件不可读。
3. 当前 HCCL 库没有 `HcclCommInitClusterInfoConfig`。
4. 默认组调用 `HcclCommInitClusterInfoConfig` 失败并返回空 communicator。
5. 子组路径没有 `HcclCreateSubCommConfig`。
6. 默认 `ProcessGroupHCCL` 的静态指针 `global_` 为空。
7. 获取默认组 communicator 抛异常或返回空。
8. `HcclCreateSubCommConfig` 调用失败。

---

## 4. 路径二：RootInfo

入口为 `ProcessGroupHCCL::createHCCLCommOrigin()`，位于 `ProcessGroupHCCL.cpp`。

当前分支中，这个函数直接使用 RootInfo 创建通信域；它不在函数内部继续尝试 `HcclCreateSubCommConfig`。

### 4.1 完整流程

```text
createHCCLCommEx() 返回 false
                │
                ▼
createHCCLCommOrigin()
                │
                ▼
当前组内 rank 0 调用 HcclGetRootInfo(&hcclID)
P2P 则由 p2pRank == 0 的一端调用
                │
                ▼
broadcastMasterID()
                │
                ├── 生成方：store_->set(storeKey, RootInfo bytes)
                └── 其他方：store_->get(storeKey)，复制到本地 hcclID
                │
                ▼
当前组所有 rank 得到相同 HcclRootInfo
                │
                ├── numRanks = getSize()；P2P 时为 2
                ├── rank     = getRank() * devices.size() + i；P2P 时为 p2pRank
                ├── rootInfo = hcclID
                └── config   = 外部配置或 PG 生成配置
                │
                ▼
HcclCommInitRootInfoConfig(numRanks, &hcclID, rank, config, &comm)
                │
                ▼
得到当前 PG 独立创建的 communicator
```

与 RankTable 子组路径不同，RootInfo 子组不依赖默认组 communicator，而是当前子组自己生成和分发一份 RootInfo，再独立初始化通信域。

### 4.2 RootInfo 的生成和 Store 分发

来自 `ProcessGroupHCCL.cpp`：

```cpp
HcclRootInfo hcclID;
bool isSingleP2POp = (commType == HcclCommType::P2P);

// 普通 PG：当前组内 rank_ == 0 的进程生成 RootInfo。
// P2P：两端小通信域里 p2pRank == 0 的一端生成。
if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    HCCL_CHECK_ERROR(HcclGetRootInfo(&hcclID));
}

// 使用当前 PG 的 c10d Store 将同一份 RootInfo 发给组内其他进程。
broadcastMasterID(&hcclID, isSingleP2POp, devicesKey, p2pRank);
```

`broadcastMasterID()` 位于 `ProcessGroupHCCL.cpp`：

```cpp
std::string storeKey;
if (!isSingleP2POp) {
    // 同一 PG 可能创建多个 communicator，用自增序号区分。
    storeKey = std::to_string(hcclCommCounter_++);
} else {
    // P2P 用设备/对端组合得到的 devicesKey 区分。
    storeKey = devicesKey;
}

if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    // 生成方把 HcclRootInfo 按字节写入 Store。
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(hcclID),
        reinterpret_cast<uint8_t*>(hcclID) + HCCL_ROOT_INFO_BYTES);
    store_->set(storeKey, vec);
} else {
    // 其他 rank 阻塞读取同一个 key，并恢复为本地 HcclRootInfo。
    auto vec = store_->get(storeKey);
    std::memcpy(hcclID, vec.data(), vec.size());
}
```

这里的 Store 只负责 RootInfo 的带外交换：

```text
c10d Store：交换建域凭据 HcclRootInfo
HCCL comm：建域成功后执行真正的 collective/P2P 数据通信
```

### 4.3 最终调用 `HcclCommInitRootInfoConfig`

带注释源码，来自 `ProcessGroupHCCL.cpp`：

```cpp
for (size_t i = 0; i < devices.size(); ++i) {
    // 当前 PG 的大小：默认组是 world_size，子组是 len(ranks)。
    int numRanks = getSize();

    // 当前进程在当前 PG 内的 rank，与 device 下标组合。
    int rank = getRank() * static_cast<int>(devices.size())
             + static_cast<int>(i);

    npuGuard.set_index(devices[i].index());

    switch (commType) {
        case HcclCommType::DEFAULT:
            if (commConfig != nullptr) {
                hcclComms[i] = HCCLComm::create_config(
                    numRanks, rank, hcclID, commConfig);
            } else {
                HcclCommConfig config = createHcclCommConfigWithOptions();
                hcclComms[i] = HCCLComm::create_config(
                    numRanks, rank, hcclID, &config);
            }
            break;

        case HcclCommType::P2P:
            // P2P 是固定两 rank 的临时通信域。
            numRanks = 2;
            rank = p2pRank;
            HcclCommConfig config;
            getHcclCommConfig(&config, true);
            hcclComms[i] = HCCLComm::create_config(
                numRanks, rank, hcclID, &config);
            break;
    }
}
```

最终封装位于 `HCCLUtils.cpp`：

```cpp
std::shared_ptr<HCCLComm> HCCLComm::create_config(
    int numRanks,
    int rank,
    HcclRootInfo& rootInfo,
    HcclCommConfig* config)
{
    auto comm = std::make_shared<HCCLComm>();

    // 最终 HCCL 接口。
    HCCL_CHECK_ERROR(hcclCommInitRootInfoConfig(
        numRanks,
        &rootInfo,
        rank,
        config,
        &(comm->hcclComm_)));

    return comm;
}
```

#### `HcclCommInitRootInfoConfig` 参数最终来源

| 最终形参 | 普通默认组/子组来源 | P2P 特例来源 |
|---|---|---|
| `numRanks` | `getSize()`；默认组来自 `world_size`，子组来自 `len(ranks)` | 固定为 `2` |
| `rootInfo` | 当前组内 rank 0 调用 `HcclGetRootInfo()` 生成，再经当前 PG 的 Store 分发 | `p2pRank==0` 一端生成，再通过 Store 分发 |
| `rank` | `getRank() * devices.size() + i`；`getRank()` 是当前 PG 的组内 rank | `p2pRank`，值为 0 或 1 |
| `config` | 外部 `commConfig`，否则 `createHcclCommConfigWithOptions()` | `getHcclCommConfig(&config, true)`，使用 P2P buffer 配置 |
| 输出 `comm` | `&(comm->hcclComm_)` | 相同，由 HCCL 写入原生 communicator 句柄 |

#### `HcclGetRootInfo` 参数最终来源

```cpp
HcclGetRootInfo(&hcclID)
```

| 参数 | 来源 |
|---|---|
| `&hcclID` | `createHCCLCommOrigin()` 栈上的局部变量 `HcclRootInfo hcclID`；由 HCCL 填充，随后序列化到 c10d Store |

---

## 5. 两条路径共享的 `HcclCommConfig` 来源

如果调用方没有传入专用 `commConfig`，普通 collective 会调用：

```cpp
createHcclCommConfigWithOptions()
```

其来源层次如下：

```text
HcclCommConfigInit(config)              HCCL 默认结构初始化
          │
          ├── HCCL_BUFFSIZE             普通通信 buffer，默认 200
          ├── P2P_HCCL_BUFFSIZE         P2P buffer，当前代码默认 20
          ├── PyTorch deterministic 状态
          ├── ProcessGroup group name   写入 hcclCommName（能力支持时）
          └── options_->hccl_config     用户/上层覆盖项
                  ├── hccl_buffer_size
                  ├── group_name / hcclUdi
                  ├── qos_traffic_class
                  ├── qos_service_level
                  ├── hccl_sdma_qos
                  ├── hccl_op_expansion_mode
                  ├── hccl_world_rank_id
                  ├── hccl_job_id
                  ├── hccl_exec_timeout
                  ├── hccl_algo
                  └── retry 等配置
```

基础初始化位于 `ProcessGroupHCCL.cpp` 的 `getHcclCommConfig()`：

```cpp
void getHcclCommConfig(HcclCommConfig* config, bool isP2P = false)
{
    HcclCommConfigInit(config);

    // 普通 collective 最终读取 HCCL_BUFFSIZE。
    // P2P 最终读取 P2P_HCCL_BUFFSIZE。
    config->hcclBufferSize = !isP2P
        ? OptionsManager::GetHcclBufferSize()
        : OptionsManager::GetP2PBufferSize();

    config->hcclDeterministic = /* CANN 版本和 PyTorch deterministic 状态 */;
}
```

然后 `ProcessGroupHCCL.cpp` 中的 `createHcclCommConfigWithOptions()` 继续把 `options_->hccl_config` 的字段覆盖到结构体。

> 注意：源码注释写 `P2P_HCCL_BUFFSIZE` 默认 `0M`，但当前实际代码默认值是 `20`；本文按执行代码记录为 20，而不是按注释记录为 0。

---

## 6. 默认组与子组的数值例子

假设：

```text
WORLD_SIZE = 8
当前子组 ranks = [1, 3, 6]
当前进程 global rank = 3
一进程一 NPU，即 devices.size() = 1
```

则子组元数据是：

```text
global_ranks_in_group = [1, 3, 6]
getSize()              = 3
getRank()              = 1    // global rank 3 在列表中的下标
HCCL rank              = 1 * 1 + 0 = 1
```

### RankTable 子组接口实参

```text
parentComm   = 默认 8-rank PG 的 communicator
rankNum      = 3
rankIds      = [1, 3, 6]
subCommId    = hash(group_id)
subCommRankId= 1
config       = 当前 PG 的 HcclCommConfig
```

### RootInfo 子组接口实参

```text
numRanks = 3
rootInfo = 子组 group_rank 0（global rank 1）生成并经 Store 分发
rank     = 1
config   = 当前 PG 的 HcclCommConfig
```

两条路径创建的是同一个逻辑成员集合 `[1,3,6]` 的通信域，但建域依据不同：

```text
RankTable：默认全局 comm + [1,3,6] ──派生──► 子 comm
RootInfo ：子组 rank 0 生成 RootInfo ──独立初始化──► 子 comm
```

---

## 7. 日志中如何判断实际走了哪条路径

| 观察到的日志 | 表示的路径 |
|---|---|
| `Create global hccl comm with ranktable success` | 默认 PG 成功走 RankTable |
| `Create sub hccl comm by hcclCreateSubCommConfig success` | 子组/P2P 成功从默认 comm 派生 |
| `The rank_table_file is not available, switch to original interface` | RankTable 文件不可用，即将转 RootInfo |
| `Create global hccl comm with ranktable failed, switch to original interface` | RankTable 默认组创建失败，即将转 RootInfo |
| `Create hccl comm by hcclCommInitRootInfoConfig success` | 最终走 RootInfo 成功 |

只看到 `ProcessGroupHCCL` 对象创建日志不能证明 communicator 已经创建，因为正常 communicator 是懒初始化的。应结合第一次 collective/P2P、上述建域日志和当前加载的 `torch_npu` 动态库版本判断。

---

## 8. 最终接口与参数来源总表

| 路径/场景 | 最终 HCCL 接口 | 决定成员集合的关键参数 | 参数最终来源 |
|---|---|---|---|
| RankTable 默认组 | `HcclCommInitClusterInfoConfig` | `clusterInfo`、`rank` | `RANK_TABLE_FILE` 路径；当前 PG `group_rank` |
| RankTable 子组 | `HcclCreateSubCommConfig` | 父 comm、`rankNum`、`rankIds`、`subCommRankId` | 默认 PG communicator；`len(ranks)`；Python `new_group(ranks)`；当前子组 rank |
| RankTable P2P | `HcclCreateSubCommConfig` | 父 comm、`rankNum=2`、两端 global rank、`p2pRank` | 默认 PG communicator；发送/接收双方 rank 映射 |
| RootInfo 默认组 | `HcclCommInitRootInfoConfig` | `numRanks`、`rootInfo`、`rank` | `WORLD_SIZE`；默认组 rank 0 的 `HcclGetRootInfo`；global rank |
| RootInfo 子组 | `HcclCommInitRootInfoConfig` | `numRanks`、`rootInfo`、`rank` | `len(ranks)`；子组 rank 0 的 `HcclGetRootInfo`；子组内 rank |
| RootInfo P2P | `HcclCommInitRootInfoConfig` | `numRanks=2`、`rootInfo`、`p2pRank` | 两端固定规模；P2P rank 0 生成；本端 0/1 rank |

---

## 9. 源码索引

源码定位统一使用“文件路径 + 稳定符号名”，避免代码增删导致定位漂移。

| 内容 | 文件 | 稳定定位符号 |
|---|---|---|
| Python HCCL backend creator | `torch_npu/__init__.py` | `_new_process_group_hccl_helper()` |
| Python extended backend 参数装配 | `torch_npu/distributed/distributed_c10d.py` | `_patched_new_process_group_helper()` 中 `_DistributedBackendOptions` 装配逻辑 |
| pybind 构造函数绑定 | `torch_npu/csrc/distributed/Init.cpp` | `ProcessGroupHCCL` 的 `py::init` 绑定 |
| `ProcessGroupHCCL` 构造 | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | `ProcessGroupHCCL::ProcessGroupHCCL()` |
| 默认 PG 设置 `global_` | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | 构造函数内 `global_ = this` |
| communicator 缓存查询 | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | `ProcessGroupHCCL::getHCCLComm()` |
| RootInfo 分发 | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | `ProcessGroupHCCL::broadcastMasterID()` |
| RootInfo 路径 | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | `ProcessGroupHCCL::createHCCLCommOrigin()` |
| RankTable 路径 | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | `ProcessGroupHCCL::createHCCLCommEx()` |
| 两条路径总分流 | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | `ProcessGroupHCCL::createHCCLComm()` |
| 配置基础初始化 | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | `getHcclCommConfig()` |
| options 配置覆盖 | `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` | `ProcessGroupHCCL::createHcclCommConfigWithOptions()` |
| RankTable 环境变量读取 | `torch_npu/csrc/core/npu/register/OptionsManager.cpp` | `OptionsManager::GetRankTableFilePath()` |
| buffer 环境变量读取 | `torch_npu/csrc/core/npu/register/OptionsManager.cpp` | `OptionsManager::GetHcclBufferSize()`、`OptionsManager::GetP2PBufferSize()` |
| HCCL C 接口包装 | `torch_npu/csrc/distributed/HCCLUtils.cpp` | `HCCLComm::create_config()`、`createGlobalHcclComm()`、`createSubHcclComm()` |
| HCCL 动态符号检查 | `torch_npu/csrc/distributed/HcclCompile.h` | `hcclCommInitClusterInfoConfigExist()`、`hcclCreateSubCommConfigExist()` 及对应包装函数 |

---

## 10. 结论边界

本文结论来自当前 checkout 的静态源码追踪，已经区分：

- `ProcessGroupHCCL` 对象创建；
- communicator 首次懒创建；
- communicator 缓存复用；
- RankTable 默认组创建；
- RankTable 子通信域派生；
- RootInfo 独立建域；
- P2P 参数覆盖。

本文没有在当前机器上执行 NPU/HCCL 任务，因此“某次实际运行最终选择哪条路径”仍需用当前加载的 wheel/动态库、环境变量和首个建域 PLOG 验证。
