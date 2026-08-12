import argparse
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

import torch
import torch_npu

FLOAT32_ELEMENT_SIZE_BYTES = 4
OUTPUT_SEPARATOR = "=" * 64
OUTPUT_SUB_SEPARATOR = "-" * 64


def parse_data_size(value):
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*([KMG]?)", value.strip(), re.IGNORECASE
    )
    if match is None:
        raise argparse.ArgumentTypeError(
            "数据量格式无效，请输入字节数或带 K/M/G 后缀的数值，例如 128K、1M、1.5G"
        )

    try:
        number = Decimal(match.group(1))
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("数据量必须是有效数字") from error

    multipliers = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    data_size_bytes = number * multipliers[match.group(2).upper()]
    if data_size_bytes <= 0:
        raise argparse.ArgumentTypeError("数据量必须大于 0")
    if data_size_bytes != data_size_bytes.to_integral_value():
        raise argparse.ArgumentTypeError("换算后的字节数必须是整数")
    if data_size_bytes % FLOAT32_ELEMENT_SIZE_BYTES != 0:
        raise argparse.ArgumentTypeError(
            f"数据量必须是 {FLOAT32_ELEMENT_SIZE_BYTES} 字节的整数倍，以适配 float32"
        )
    return int(data_size_bytes)


def parse_device(value):
    try:
        device_id = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("NPU 卡号必须是整数") from error
    if device_id < 0:
        raise argparse.ArgumentTypeError("NPU 卡号不能小于 0")
    return device_id


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
    formatted_value = format(value.quantize(Decimal("0.01")), "f").rstrip("0").rstrip(".")
    return f"{formatted_value} {unit}"


def find_kernel_details(profiler_path):
    kernel_details_files = []
    for info_file in profiler_path.rglob("profiler_info_*.json"):
        kernel_details_file = info_file.parent / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
        if kernel_details_file.is_file():
            kernel_details_files.append(kernel_details_file)

    if not kernel_details_files:
        kernel_details_files = list(profiler_path.rglob("kernel_details.csv"))
    if not kernel_details_files:
        raise FileNotFoundError(f"未找到 kernel_details.csv：{profiler_path}")
    if len(kernel_details_files) > 1:
        paths = "\n".join(str(path) for path in kernel_details_files)
        raise RuntimeError(f"找到多个 kernel_details.csv，无法确定要解析的文件：\n{paths}")
    return kernel_details_files[0]


def read_fill_durations(kernel_details_file, active_steps):
    durations = []
    with kernel_details_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if "Name" not in fieldnames or "Duration(us)" not in fieldnames:
            raise RuntimeError(
                "kernel_details.csv 缺少 Name 或 Duration(us) 列，"
                f"实际列为：{', '.join(fieldnames)}"
            )

        for row in reader:
            if not row["Name"].startswith("aclnnInplaceFillScalar"):
                continue
            try:
                duration = Decimal(row["Duration(us)"].replace(",", ""))
            except InvalidOperation as error:
                raise RuntimeError(
                    f"Duration(us) 不是有效数字：{row['Duration(us)']}"
                ) from error
            if duration <= 0:
                raise RuntimeError(f"Duration(us) 必须大于 0，实际为：{duration}")
            durations.append(duration)

    if len(durations) < active_steps:
        raise RuntimeError(
            "aclnnInplaceFillScalar 记录数量不足："
            f"需要 {active_steps} 条，实际找到 {len(durations)} 条"
        )
    return durations[-active_steps:]


def calculate_throughputs(data_size_bytes, durations_us):
    microseconds_per_second = Decimal(1000000)
    return [
        Decimal(data_size_bytes) * microseconds_per_second / duration
        for duration in durations_us
    ]


def format_throughput_stats(throughputs):
    average = sum(throughputs) / len(throughputs)
    unit_value = Decimal(1)
    unit = "B/s"
    for candidate in ("KiB/s", "MiB/s", "GiB/s", "TiB/s"):
        if average < unit_value * 1024:
            break
        unit_value *= 1024
        unit = candidate

    return {
        "平均": f"{average / unit_value:.2f} {unit}",
        "最大": f"{max(throughputs) / unit_value:.2f} {unit}",
        "最小": f"{min(throughputs) / unit_value:.2f} {unit}",
    }


def format_duration_stats(durations_us):
    average = sum(durations_us) / len(durations_us)
    return {
        "平均": f"{average:.2f}",
        "最大": f"{max(durations_us):.2f}",
        "最小": f"{min(durations_us):.2f}",
    }


parser = argparse.ArgumentParser(description="采集 swapped memory 的 fill_ 性能数据")
parser.add_argument("--device", type=parse_device, default=0, help="NPU 卡号（默认：0）")
parser.add_argument(
    "--data-size",
    type=parse_data_size,
    default=1024**2,
    metavar="数据量",
    help="数据字节数，支持 K/M/G 后缀（默认：1M）",
)
parser.add_argument(
    "--warmup-steps",
    type=parse_non_negative_int,
    default=3,
    metavar="次数",
    help="预热次数（默认：3）",
)
parser.add_argument(
    "--active-steps",
    type=parse_positive_int,
    default=5,
    metavar="次数",
    help="正式采集次数（默认：5）",
)
args = parser.parse_args()

num_elements = args.data_size // FLOAT32_ELEMENT_SIZE_BYTES
device = f"npu:{args.device}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
profiler_path = Path(__file__).resolve().parent / (
    f"profiler_output_device_{args.device}_{args.data_size}_bytes_{timestamp}"
)

print(OUTPUT_SEPARATOR)
print(f"  NPU 设备：{device}")
print(
    f"  数据大小：{format_data_size(args.data_size)}"
    f"（float32 元素个数：{num_elements:,}）"
)
print(f"  预热次数：{args.warmup_steps}")
print(f"  采集次数：{args.active_steps}")
print(OUTPUT_SUB_SEPARATOR)

torch.npu.set_device(device)

# https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/cpp/at_npu-native-empty_with_swapped_memory.md
swapped_tensor = torch_npu.empty_with_swapped_memory(
    [num_elements, 1], dtype=torch.float32, device=device
)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=0,
        warmup=args.warmup_steps,
        active=args.active_steps,
        repeat=1,
        skip_first=0,
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profiler_path)),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        aic_metrics=torch_npu.profiler.AiCMetrics.ArithmeticUtilization,
    ),
) as prof:
    for _ in range(args.warmup_steps + args.active_steps):
        swapped_tensor.fill_(3.14)
        prof.step()

kernel_details_file = find_kernel_details(profiler_path)
durations = read_fill_durations(kernel_details_file, args.active_steps)
throughputs = calculate_throughputs(args.data_size, durations)
throughput_stats = format_throughput_stats(throughputs)
duration_stats = format_duration_stats(durations)
print(OUTPUT_SEPARATOR)
print("  类型 | 填充吞吐率 | 耗时（us）")
print(OUTPUT_SUB_SEPARATOR)
for label in ("平均", "最大", "最小"):
    print(f"  {label} | {throughput_stats[label]} | {duration_stats[label]}")
print(OUTPUT_SUB_SEPARATOR)
print(f"  性能数据目录：{profiler_path}")
print(OUTPUT_SEPARATOR)
