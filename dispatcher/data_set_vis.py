import matplotlib.pyplot as plt
import argparse
import numpy
import random

from typing import List


parser = argparse.ArgumentParser(description='SIMU dataset')

parser.add_argument('--min_lam', type=int, default=100)
parser.add_argument('--max_lam', type=int, default=120)

parser.add_argument('--min_data_size', type=int, default=50000)
parser.add_argument('--max_data_size', type=int, default=100000)

args = parser.parse_args()


def generate_token_list() -> List[int]:
    lam = random.randint(args.min_lam, args.max_lam)
    data_size = random.randint(args.min_data_size, args.max_data_size)
    data = numpy.random.poisson(lam, data_size).tolist()
    random.shuffle(data)
    return data


if __name__ == "__main__":
    data = generate_token_list()
    total = sum(data)

    count = {}
    for k in data:
        if k not in count:
            count[k] = 0
        count[k] += 1

    fig, axs = plt.subplots()

    axs.bar(count.keys(),
            count.values(), color="#ff0000")

    fig.savefig("data_set_vis.png")
