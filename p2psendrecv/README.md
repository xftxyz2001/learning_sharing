# NPU P2P Send/Recv 性能测试

`p2psendrecv.py` 使用 `torch.distributed` 和 HCCL 测试两个 NPU Rank 之间的点对点发送、接收性能，并通过 Ascend PyTorch Profiler 统计带宽和耗时。

## 使用方法

需要安装匹配版本的 PyTorch、`torch_npu` 和 CANN，并准备至少两张可用 NPU。

```bash
python3 p2psendrecv.py
```

默认由 Rank 0 向 Rank 1 传输 `1 GiB` 数据，预热 3 次并采集 5 次。自定义示例：

```bash
python3 p2psendrecv.py --send 0 --recv 1 --size 512M --warm 3 --active 5
```

脚本默认在统计成功后删除 Profiler 原始数据；添加 `--keep-profiler` 可保留采集目录。
