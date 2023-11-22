import sys
import argparse
import numpy
import random

from dataclasses import dataclass
from typing import List, Tuple
from copy import deepcopy


parser = argparse.ArgumentParser(description='Dispatcher simu')
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--total_task', type=int, default=10)
parser.add_argument('--running_task', type=int, default=4)
parser.add_argument('--task_window', type=int, default=8)
parser.add_argument('--sync_running_task', type=int, default=2)

parser.add_argument('--min_loc', type=int, default=100)
parser.add_argument('--max_loc', type=int, default=120)

parser.add_argument('--min_data_size', type=int, default=5000)
parser.add_argument('--max_data_size', type=int, default=10000)

parser.add_argument('--min_task_repeat', type=int, default=1)
parser.add_argument('--max_task_repeat', type=int, default=4)

parser.add_argument('--seq_tp', type=int, default=300)
parser.add_argument('--sync_tp', type=int, default=900)
parser.add_argument('--aspen_tp', type=int, default=1000)

args = parser.parse_args()


@dataclass
class Task:
    tokens_len_list: List[int] = None
    start_time: float = 0.0
    end_time: float = 0.0
    server_time: float = 0.0

    # only for sync model
    sync_time: int = 0
    task_index: int = -1


def set_seed():
    random.seed(args.seed)
    numpy.random.seed(args.seed)


def generate_token_list() -> List[int]:
    loc = random.randint(args.min_loc, args.max_loc)
    data_size = random.randint(args.min_data_size, args.max_data_size)
    data = numpy.random.poisson(loc, data_size).tolist()
    random.shuffle(data)
    return data


def generate_task_list() -> List[Task]:
    ret: List[Task] = []
    r_task = args.total_task
    task_index = 0
    while r_task > 0:
        task_num = random.randint(args.min_task_repeat, args.max_task_repeat)
        if task_num > r_task:
            task_num = r_task
        tokens = generate_token_list()
        for _ in range(0, task_num):
            ret.append(Task(tokens_len_list=tokens.copy(), start_time=0.0,
                       end_time=0.0, server_time=0.0, sync_time=0, task_index=task_index))
            task_index += 1
        r_task -= task_num
    return ret


def find_nearest(value: int, data: List[int]) -> int:
    # return the index
    min = sys.maxsize
    ret = 0
    for i in range(0, len(data)):
        if abs(value - min) < min:
            min = abs(value - min)
            ret = i
    return ret


def simu_seq_dispatcher(tasks: List[Task]):
    now_time = 0
    total_tokens = 0
    for t in tasks:
        now_tokens = sum(t.tokens_len_list)
        total_tokens += now_tokens
        t.server_time = now_tokens / args.seq_tp
        now_time += t.server_time
        t.end_time = now_time
    return now_time


def simu_sync_dispatcher(tasks: List[Task]):

    task_window: List[List[Task]] = []
    for _ in range(0, args.sync_running_task):
        task_window.append([])

    rr = 0
    for t in tasks:
        task_window[rr].append(t)
        rr = (rr + 1) % args.sync_running_task

    dispatcher_time: List[int] = []
    for tasks in task_window:
        tmp_t = 0
        for t in tasks:
            tmp_t += len(t.tokens_len_list)
            dispatcher_time.append(tmp_t)

    def find_task(time: int) -> List[Tuple[Task, bool, int]]:
        ret: List[Tuple[Task, bool, int]] = []
        for task_w in task_window:
            t_t_start = 0
            t_t_end = 0
            for task in task_w:
                t_t_end = t_t_start + len(task.tokens_len_list)
                if t_t_end == time:
                    ret.append(
                        tuple([task, True, sum(task.tokens_len_list[task.sync_time: time-t_t_start])]))
                    task.sync_time = time-t_t_start
                    break
                elif t_t_end > time:
                    ret.append(
                        tuple([task, False, sum(task.tokens_len_list[task.sync_time: time-t_t_start])]))
                    task.sync_time = time-t_t_start
                    break
                t_t_start = t_t_end
        return ret

    now_time = 0
    dispatcher_time = sorted(dispatcher_time)
    for t in dispatcher_time:
        total_task = find_task(t)

        if len(total_task) == args.sync_running_task:
            tp = args.sync_tp
        else:
            tp = args.seq_tp

        total_token = 0
        for r_task in total_task:
            _, _, sum_tokens = r_task
            total_token += sum_tokens
        server_time = total_token / tp
        now_time += server_time

        for r_task in total_task:
            task, is_end, _ = r_task
            if is_end:
                task.end_time = now_time
            task.server_time += server_time

    return now_time


def simu_aspen_optim(tasks: List[Task]):
    now_time = 0

    del_tasks = deepcopy(tasks)

    while len(del_tasks) > 0:
        choice_index = random.randrange(len(del_tasks))
        task_choice = del_tasks.pop(choice_index)
        value_index = random.randrange(len(task_choice.tokens_len_list))
        value = task_choice.tokens_len_list.pop(value_index)

        # pad, token_index, task_index
        find_window: List[Tuple[int, int, int]] = []
        for task_index in range(0, len(del_tasks)):
            token_index = find_nearest(
                value, del_tasks[task_index].tokens_len_list)
            diff = abs(
                value - del_tasks[task_index].tokens_len_list[token_index])
            find_window.append(
                tuple([diff, token_index, task_index]))

        find_window = sorted(find_window, key=lambda x: x[0])[
            :args.running_task]

        align_len = value
        pop_list = []
        for find_value in find_window:
            token_index = find_value[1]
            task_index = find_value[2]
            align_len = max(del_tasks[task_index].tokens_len_list.pop(
                token_index), align_len)
            if len(del_tasks[task_index].tokens_len_list) == 0:
                pop_list.append(task_index)

        server_time = (align_len * args.running_task) / args.aspen_tp
        for find_value in find_window:
            tasks[del_tasks[find_value[2]].task_index].server_time += server_time
        now_time += server_time

        pop_list = sorted(pop_list, reverse=True)
        for i in pop_list:
            tasks[del_tasks[i].task_index].end_time = now_time
            del_tasks.pop(i)

        # remain train data
        if len(task_choice.tokens_len_list) > 0:
            del_tasks.append(task_choice)
        else:
            tasks[task_choice.task_index].end_time = now_time

    return now_time


if __name__ == "__main__":
    set_seed()
    tasks: List[Task] = generate_task_list()

    seq_task = deepcopy(tasks)
    sync_task = deepcopy(tasks)
    aspen_task = deepcopy(tasks)

    print(f"baseline seq : {simu_seq_dispatcher(seq_task)}")
    print(f"baseline sync: {simu_sync_dispatcher(sync_task)}")
    print(f"aspen optim  : {simu_aspen_optim(aspen_task)}")
