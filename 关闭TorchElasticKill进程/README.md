# 关闭 TorchElastic 自动终止进程

该目录提供 PyTorch `2.7.1`～`2.12.0` 的 TorchElastic 补丁，用于阻止 Elastic Agent 退出或收到终止信号时自动关闭其管理的 worker 进程。

## 使用方法

先确认当前 PyTorch 版本：

```bash
pip show torch
```

执行 `package.sh`，为各版本生成压缩包：

```bash
bash package.sh
```

选择与当前 PyTorch 版本一致的压缩包，并解压到 PyTorch 安装目录。例如，使用 `2.10.0`：

```bash
torch_path=$(pip show torch | grep Location | awk '{print $2}')
tar -zxvf 2.10.0.tar.gz -C "$torch_path"
```

## 注意事项

- 补丁会直接覆盖 PyTorch 安装目录中的文件，请确保补丁版本与 PyTorch 版本一致。
- 应用后，TorchElastic 不再自动清理 worker，可能留下残留进程并持续占用设备资源。
- 使用完成后需要手动结束残留进程。
- 如需恢复原始行为，请重新安装对应版本的 PyTorch。
