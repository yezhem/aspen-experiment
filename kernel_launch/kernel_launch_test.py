#!/bin/bash
import time
import torch

if __name__ == "__main__":
    w = torch.rand((4096, 8), dtype=torch.float16, device="cuda:0")
    in_datas = []

    for len in range(0, 32):
        in_data = torch.rand(
            (8, 512, 4096), dtype=torch.float16, device="cuda:0")
        in_datas.append(in_data)

    fused_data = torch.stack(in_datas).reshape(32 * 8, 512, 4096)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof0:
        for i in range(0, 100):
            start_time = time.time()
            for in_data in in_datas:
                o = in_data @ w
            end_time = time.time()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof1:
        for i in range(0, 100):
            start_time = time.time()
            o = fused_data @ w
            end_time = time.time()

    print(prof0.key_averages().table())
    print(prof1.key_averages().table())
