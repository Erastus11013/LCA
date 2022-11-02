from collections import defaultdict

import tqdm
import matplotlib.pyplot as plt
from time import monotonic_ns
import numpy as np
from random import randint

from rmq import (RMQCartesianTreeLCA, RMQFischerHeun, RMQPrecomputed,
                 RMQSegmentTree, RMQSparseTable, RMQSqrtDecomposition)
from statistics import mean

rmq_constructors = (
    RMQPrecomputed,
    RMQSegmentTree,
    RMQSparseTable,
    RMQSqrtDecomposition,
    RMQFischerHeun,
    RMQCartesianTreeLCA,
)


def benchmark_query_times():
    num_elems_interval = (10, 300, 1)

    times = defaultdict(list)

    for num_elems in tqdm.trange(*num_elems_interval):
        num_iters = 10
        constructor_2_query_times = defaultdict(list)
        test = np.random.randint(0, 100000, num_elems)
        rmq_structures = {rmq_constructor: rmq_constructor(test) for rmq_constructor in rmq_constructors}

        for _ in range(num_iters):
            start = randint(0, num_elems - 1)
            stop = randint(start, num_elems - 1)

            for rmq_constructor, rmq in rmq_structures.items():
                start_time = monotonic_ns()
                _ = rmq[start: stop]
                constructor_2_query_times[rmq_constructor].append(monotonic_ns() - start_time)

        for rmq_constructor, times in constructor_2_query_times.items():
            times[rmq_constructor].append(mean(times))

    x_range = np.arange(*num_elems_interval)
    for rmq_constructor, times in times.items():
        plt.plot(x_range, times, label=f'{rmq_constructor.__name__}')
    plt.xlabel('# elements')
    plt.ylabel('# average query times in ns')
    plt.legend()
    plt.show()


def benchmark_construction_times():
    num_elems_interval = (10, 300, 1)

    times = defaultdict(list)

    for num_elems in tqdm.trange(*num_elems_interval):
        num_iters = 10
        constructor_2_construction_times = defaultdict(list)
        test = np.random.randint(0, 100000, num_elems)

        for rmq_constructor in rmq_constructors:
            for _ in range(num_iters):
                start_time = monotonic_ns()
                _ = rmq_constructor(test)
                constructor_2_construction_times[rmq_constructor].append(monotonic_ns() - start_time)

        for rmq_constructor, times in constructor_2_construction_times.items():
            times[rmq_constructor].append(mean(times))

    x_range = np.arange(*num_elems_interval)
    for rmq_constructor, times in times.items():
        plt.plot(x_range, times, label=f'{rmq_constructor.__name__}')
    plt.xlabel('# elements')
    plt.ylabel('# average construction time in ns')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    benchmark_construction_times()











