import sys

from typing import List
from dataclasses import dataclass


@dataclass
class Metric:
    batch_size: int
    seq_len: int

    base_time: List[float]
    lora_time: List[float]
    base_mem: List[int]
    lora_mem: List[int]

    gpu_util: List[int]

    forward_time: float
    forward_mem: int

    backward_time: float
    backward_mem: int

    loss_calc_time: float
    loss_calc_mem: int

    optim_time: float
    optim_mem: int

    train_loss: List[float]


def read_all_metric_from_file(file: str):
    with open(file, "r", encoding="utf8") as fp:
        lines = fp.readlines()
    lines = [l.strip() for l in lines]
    lines.append("dummy_end")

    ret_metric: List[Metric] = []
    metric: Metric = None

    is_start: bool = True

    for line in lines:
        if "data size" in line:
            if is_start:
                is_start = False
            else:
                ret_metric.append(metric)
            metric = Metric(0, 0, [], [], [], [], [],
                            0, 0, 0, 0, 0, 0, 0, 0, [])
            arg = line[11:].split()
            metric.batch_size = int(arg[0])
            metric.seq_len = int(arg[1])
        elif "base" in line:
            arg = line[6:].split()
            metric.base_time.append(float(arg[0]))
            metric.base_mem.append(int(arg[1]))
            metric.gpu_util.append(int(arg[2]))
        elif "lora" in line:
            arg = line[6:].split()
            metric.lora_time.append(float(arg[0]))
            metric.lora_mem.append(int(arg[1]))
            metric.gpu_util.append(int(arg[2]))
        elif "forward" in line:
            arg = line[9:].split()
            metric.forward_time = float(arg[0])
            metric.forward_mem = int(arg[1])
            metric.gpu_util.append(int(arg[2]))
        elif "backward" in line:
            arg = line[10:].split()
            metric.backward_time = float(arg[0])
            metric.backward_mem = int(arg[1])
            metric.gpu_util.append(int(arg[2]))
        elif "loss" in line:
            arg = line[6:].split()
            metric.loss_calc_time = float(arg[0])
            metric.loss_calc_mem = int(arg[1])
            metric.gpu_util.append(int(arg[2]))
        elif "train" in line:
            arg = line[7:]
            metric.train_loss.append(float(arg))
        elif "optim" in line:
            arg = line[7:].split()
            metric.optim_time = float(arg[0])
            metric.optim_mem = int(arg[1])
            metric.gpu_util.append(int(arg[2]))
        elif "dummy_end" in line:
            ret_metric.append(metric)

    return ret_metric


def get_total_time_metric(metric: List[Metric]):
    total_time = 0
    for m in metric:
        total_time += m.forward_time + m.backward_time + m.optim_time + m.loss_calc_time
    return total_time


def get_peak_mem_metric(metric: List[Metric]):
    peak_mem = 0
    for m in metric:
        peak_mem = max([peak_mem, m.forward_mem, m.backward_mem,
                       max(m.base_mem, default=0), max(m.lora_mem, default=0), m.loss_calc_mem, m.optim_mem])
    return peak_mem


def get_total_tokens(metric: List[Metric]):
    total_tokens = 0
    for m in metric:
        total_tokens += (m.batch_size * m.seq_len)
    return total_tokens


def get_avg_gpu_metirc(metric: List[Metric]):
    gpu_util: List[int] = []
    for m in metric:
        gpu_util.extend(m.gpu_util)
    return sum(gpu_util) / len(gpu_util)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("No file to analyze")
    else:
        file_name = sys.argv[1]

    metric = read_all_metric_from_file(file_name)
    print(f"total time cost: {get_total_time_metric(metric)}")
    print(f"peak memory usage: {get_peak_mem_metric(metric)}")
    print(f"avg gpu utl: {get_avg_gpu_metirc(metric)}")
