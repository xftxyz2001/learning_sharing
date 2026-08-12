import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


FLOAT32_ELEMENT_SIZE_BYTES = 4
MICROSECONDS_PER_SECOND = Decimal(1_000_000)
OUTPUT_SEPARATOR = "=" * 64
OUTPUT_SUB_SEPARATOR = "-" * 64


def parse_data_size(value):
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)(?:I?B)?",
        value.strip(),
        re.IGNORECASE,
    )
    if match is None:
        raise argparse.ArgumentTypeError(
            "数据量格式无效，请输入字节数或带 K/M/G/T 后缀的数值，"
            "例如 512M、1G、1.5G"
        )

    try:
        number = Decimal(match.group(1))
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("数据量必须是有效数字") from error

    multipliers = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }
    data_size_bytes = number * multipliers[match.group(2).upper()]
    if data_size_bytes <= 0:
        raise argparse.ArgumentTypeError("数据量必须大于 0")
    if data_size_bytes != data_size_bytes.to_integral_value():
        raise argparse.ArgumentTypeError("换算后的字节数必须是整数")
    if data_size_bytes % FLOAT32_ELEMENT_SIZE_BYTES != 0:
        raise argparse.ArgumentTypeError(
            f"数据量必须是 {FLOAT32_ELEMENT_SIZE_BYTES} 字节的整数倍，"
            "以适配 float32"
        )
    return int(data_size_bytes)


def parse_non_negative_int(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是整数") from error
    if result < 0:
        raise argparse.ArgumentTypeError("不能小于 0")
    return result


def parse_positive_int(value):
    result = parse_non_negative_int(value)
    if result == 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return result


def format_data_size(data_size_bytes):
    value = Decimal(data_size_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    formatted_value = format(value.quantize(Decimal("0.01")), "f")
    formatted_value = formatted_value.rstrip("0").rstrip(".")
    return f"{formatted_value} {unit}"


def format_rate_stats(data_size_bytes, durations_us):
    average_duration = sum(durations_us) / len(durations_us)
    duration_stats = {
        "平均": average_duration,
        "最大": max(durations_us),
        "最小": min(durations_us),
    }
    rates = {
        "平均": Decimal(data_size_bytes) * MICROSECONDS_PER_SECOND / average_duration,
        "最大": Decimal(data_size_bytes)
        * MICROSECONDS_PER_SECOND
        / min(durations_us),
        "最小": Decimal(data_size_bytes)
        * MICROSECONDS_PER_SECOND
        / max(durations_us),
    }

    unit_value = Decimal(1)
    unit = "B/s"
    for candidate in ("KiB/s", "MiB/s", "GiB/s", "TiB/s"):
        if rates["平均"] < unit_value * 1024:
            break
        unit_value *= 1024
        unit = candidate

    return {
        label: {
            "rate": f"{rates[label] / unit_value:.2f} {unit}",
            "duration": f"{duration_stats[label]:.2f}",
        }
        for label in ("平均", "最大", "最小")
    }


def find_trace_file(profiler_path, rank):
    info_files = list(profiler_path.rglob(f"profiler_info_{rank}.json"))
    trace_files = []
    for info_file in info_files:
        trace_file = info_file.parent / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
        if trace_file.is_file():
            trace_files.append(trace_file)

    if not trace_files:
        raise FileNotFoundError(
            f"未找到 Rank {rank} 的 trace_view.json：{profiler_path}"
        )
    if len(trace_files) > 1:
        paths = "\n".join(str(path) for path in trace_files)
        raise RuntimeError(
            f"找到多个 Rank {rank} 的 trace_view.json，无法确定解析目标：\n{paths}"
        )
    return trace_files[0]


def read_notify_wait_durations(trace_file, base_name, active_steps):
    with trace_file.open("r", encoding="utf-8") as file:
        trace_data = json.load(file)

    if isinstance(trace_data, dict):
        trace_events = trace_data.get("traceEvents", [])
    elif isinstance(trace_data, list):
        trace_events = trace_data
    else:
        raise RuntimeError(f"trace_view.json 顶层格式无效：{trace_file}")

    base_timestamps = [
        event.get("ts")
        for event in trace_events
        if event.get("name") == base_name and event.get("ts") is not None
    ]
    if not base_timestamps:
        raise RuntimeError(f"未找到事件 {base_name}：{trace_file}")
    first_base_timestamp = min(base_timestamps)

    target_events = []
    for event in trace_events:
        timestamp = event.get("ts")
        duration = event.get("dur")
        if (
            event.get("name") == "NOTIFY_WAIT"
            and timestamp is not None
            and duration is not None
            and timestamp > first_base_timestamp
        ):
            try:
                parsed_duration = Decimal(str(duration))
            except InvalidOperation as error:
                raise RuntimeError(
                    f"NOTIFY_WAIT 的 dur 不是有效数字：{duration}"
                ) from error
            if parsed_duration <= 0:
                raise RuntimeError(
                    f"NOTIFY_WAIT 的 dur 必须大于 0，实际为：{parsed_duration}"
                )
            target_events.append((timestamp, parsed_duration))

    target_events.sort(key=lambda item: item[0])
    if len(target_events) < active_steps:
        raise RuntimeError(
            f"Rank 对应的 NOTIFY_WAIT 记录不足：需要 {active_steps} 条，"
            f"实际找到 {len(target_events)} 条，文件：{trace_file}"
        )
    return [duration for _, duration in target_events[-active_steps:]]


def build_parser():
    parser = argparse.ArgumentParser(description="测试 NPU P2P Send/Recv 通信性能")
    parser.add_argument("--send", type=parse_non_negative_int, default=0, help="发送方 Rank（默认：0）")
    parser.add_argument("--recv", type=parse_non_negative_int, default=1, help="接收方 Rank（默认：1）")
    parser.add_argument(
        "--size",
        type=parse_data_size,
        default=1024**3,
        metavar="数据量",
        help="通信数据量，支持 K/M/G/T 后缀（默认：1G）",
    )
    parser.add_argument("--warm", type=parse_non_negative_int, default=3, help="预热次数（默认：3）")
    parser.add_argument("--active", type=parse_positive_int, default=5, help="采集次数（默认：5）")
    parser.add_argument(
        "--nproc-per-node",
        type=parse_positive_int,
        default=None,
        help="每节点进程数（默认：本机 NPU 数量）",
    )
    parser.add_argument(
        "--master-addr",
        default=os.getenv("MASTER_ADDR", "localhost"),
        help="主节点地址（默认：MASTER_ADDR 或 localhost）",
    )
    parser.add_argument(
        "--master-port",
        type=parse_positive_int,
        default=int(os.getenv("MASTER_PORT", "20022")),
        help="主节点端口（默认：MASTER_PORT 或 20022）",
    )
    parser.add_argument(
        "--nnodes",
        type=parse_positive_int,
        default=int(os.getenv("NNODES", "1")),
        help="节点数（默认：NNODES 或 1）",
    )
    parser.add_argument(
        "--node-rank",
        type=parse_non_negative_int,
        default=int(os.getenv("NODE_RANK", "0")),
        help="当前节点编号（默认：NODE_RANK 或 0）",
    )
    parser.add_argument(
        "--keep-profiler",
        action="store_true",
        help="保留 Profiler 原始数据（默认：统计成功后删除）",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profiler-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS
    )
    return parser


def print_parameters(args, device_count, nproc_per_node):
    world_size = args.nnodes * nproc_per_node
    print(OUTPUT_SEPARATOR)
    print(f"  发送方：Rank {args.send}")
    print(f"  接收方：Rank {args.recv}")
    print(
        f"  数据大小：{format_data_size(args.size)}"
        f"（float32 元素个数：{args.size // FLOAT32_ELEMENT_SIZE_BYTES:,}）"
    )
    print(f"  预热次数：{args.warm}")
    print(f"  采集次数：{args.active}")
    print(f"  NPU 数量：{device_count}")
    print(f"  进程数量：{world_size}（每节点 {nproc_per_node}）")
    print(OUTPUT_SUB_SEPARATOR, flush=True)


def print_stats(title, stats):
    print(f"  {title}")
    print("  类型 | 带宽 | 耗时（us）")
    print(OUTPUT_SUB_SEPARATOR)
    for label in ("平均", "最大", "最小"):
        print(f"  {label} | {stats[label]['rate']} | {stats[label]['duration']}")


def run_worker(args):
    import torch
    import torch.distributed as dist
    import torch_npu

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://")
    rank = dist.get_rank()
    group = dist.new_group(ranks=[args.send, args.recv], backend="hccl")

    num_elements = args.size // FLOAT32_ELEMENT_SIZE_BYTES
    tensor = None
    if rank == args.send:
        tensor = torch.ones(num_elements, dtype=torch.float32, device=f"npu:{local_rank}")
    elif rank == args.recv:
        tensor = torch.empty(num_elements, dtype=torch.float32, device=f"npu:{local_rank}")

    dist.barrier()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=args.warm,
            active=args.active,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(args.profiler_path)
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=torch_npu.profiler.AiCMetrics.ArithmeticUtilization,
        ),
    ) as prof:
        for _ in range(args.warm + args.active):
            if rank == args.send:
                dist.send(tensor=tensor, dst=args.recv, group=group)
            elif rank == args.recv:
                dist.recv(tensor=tensor, src=args.send, group=group)
            prof.step()

    dist.barrier()
    dist.destroy_process_group()


def run_launcher(args):
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("未安装 torch 或 torch_npu，无法执行 NPU 通信测试") from error

    device_count = torch.npu.device_count()
    if device_count <= 0:
        raise RuntimeError("没有可用的 NPU 设备")

    nproc_per_node = args.nproc_per_node or device_count
    if nproc_per_node > device_count:
        print(
            f"WARN：请求的每节点进程数 {nproc_per_node} 大于 NPU 数量 "
            f"{device_count}，已调整为 {device_count}"
        )
        nproc_per_node = device_count

    world_size = args.nnodes * nproc_per_node
    if args.send >= world_size:
        raise ValueError(f"发送方 Rank {args.send} 不在合法范围 [0, {world_size - 1}]")
    if args.recv >= world_size:
        raise ValueError(f"接收方 Rank {args.recv} 不在合法范围 [0, {world_size - 1}]")
    if args.send == args.recv:
        raise ValueError("发送方和接收方 Rank 不能相同")
    if args.node_rank >= args.nnodes:
        raise ValueError(
            f"节点编号 {args.node_rank} 不在合法范围 [0, {args.nnodes - 1}]"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    profiler_path = Path(__file__).resolve().parent / f"profiler_output_{timestamp}"
    print_parameters(args, device_count, nproc_per_node)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port}",
        f"--nnodes={args.nnodes}",
        f"--node-rank={args.node_rank}",
        f"--nproc-per-node={nproc_per_node}",
        str(Path(__file__).resolve()),
        "--worker",
        f"--send={args.send}",
        f"--recv={args.recv}",
        f"--size={args.size}",
        f"--warm={args.warm}",
        f"--active={args.active}",
        f"--profiler-path={profiler_path}",
    ]
    subprocess.run(command, check=True)

    sender_trace = find_trace_file(profiler_path, args.send)
    receiver_trace = find_trace_file(profiler_path, args.recv)
    sender_durations = read_notify_wait_durations(
        sender_trace, "c10d::send", args.active
    )
    receiver_durations = read_notify_wait_durations(
        receiver_trace, "c10d::recv_", args.active
    )
    sender_stats = format_rate_stats(args.size, sender_durations)
    receiver_stats = format_rate_stats(args.size, receiver_durations)

    print(OUTPUT_SEPARATOR)
    print_stats(f"发送方（Rank {args.send}）", sender_stats)
    print(OUTPUT_SUB_SEPARATOR)
    print_stats(f"接收方（Rank {args.recv}）", receiver_stats)
    if args.keep_profiler:
        print(OUTPUT_SUB_SEPARATOR)
        print(f"  性能数据目录：{profiler_path}")
    print(OUTPUT_SEPARATOR)

    if not args.keep_profiler:
        shutil.rmtree(profiler_path)


def main():
    args = build_parser().parse_args()
    if args.worker:
        if args.profiler_path is None:
            raise ValueError("worker 模式缺少 --profiler-path")
        run_worker(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
