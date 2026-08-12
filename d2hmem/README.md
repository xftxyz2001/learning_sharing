# Swapped Memory Fill 性能测试

`d2hmem.py` 使用 `torch_npu.empty_with_swapped_memory` 创建设备信息为 NPU、实际内存在 Host 侧的特殊 Tensor，采集 `fill_` 操作的耗时和填充吞吐率。

## 使用方法

需要安装匹配版本的 PyTorch、`torch_npu` 和 CANN，并在支持该接口的昇腾环境运行。

```bash
python3 d2hmem.py
```

默认使用 0 卡和 `1 MiB` 数据，预热 3 次并采集 5 次。自定义示例：

```bash
python3 d2hmem.py --device 0 --data-size 128M --warmup-steps 3 --active-steps 5
```

Profiler 数据保存在当前目录生成的 `profiler_output_*` 目录中；脚本会从 `kernel_details.csv` 提取 `aclnnInplaceFillScalar` 的耗时并计算吞吐率。
