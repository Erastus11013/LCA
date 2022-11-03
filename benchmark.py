from collections import defaultdict
from random import randint
from statistics import mean
from time import monotonic_ns

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from rmq import (RMQCartesianTreeLCA, RMQFischerHeun, RMQPrecomputed,
                 RMQSegmentTree, RMQSparseTable, RMQSqrtDecomposition)

rmq_constructors = (
    RMQPrecomputed,
    RMQSegmentTree,
    RMQSparseTable,
    RMQSqrtDecomposition,
    RMQFischerHeun,
    RMQCartesianTreeLCA,
)

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})


def benchmark_query_times():
    num_elems_interval = (10, 500, 1)

    alg_2_times = defaultdict(list)

    for num_elems in tqdm.trange(*num_elems_interval):
        num_iters = 5
        constructor_2_query_times = defaultdict(list)
        test = np.random.randint(0, 100000, num_elems)
        rmq_structures = {
            rmq_constructor: rmq_constructor(test)
            for rmq_constructor in rmq_constructors
        }

        for _ in range(num_iters):
            start = randint(0, num_elems - 1)
            stop = randint(start, num_elems - 1)

            for rmq_constructor, rmq in rmq_structures.items():
                start_time = monotonic_ns()
                _ = rmq[start:stop]
                constructor_2_query_times[rmq_constructor].append(
                    monotonic_ns() - start_time
                )

        for rmq_constructor, times in constructor_2_query_times.items():
            alg_2_times[rmq_constructor].append(mean(times))

    x_range = np.arange(*num_elems_interval)
    for rmq_constructor, times in alg_2_times.items():
        plt.plot(x_range, times, label=f"{rmq_constructor.__name__}")
    plt.xlabel(r"no. elements")
    plt.ylabel(r"average query time (ns)")
    plt.legend()
    plt.show()


def benchmark_construction_times():
    num_elems_interval = (10, 500, 1)

    alg_2_times = defaultdict(list)

    for num_elems in tqdm.trange(*num_elems_interval):
        num_iters = 5
        constructor_2_construction_times = defaultdict(list)
        test = np.random.randint(0, 100000, num_elems)

        for rmq_constructor in rmq_constructors:
            for _ in range(num_iters):
                start_time = monotonic_ns()
                _ = rmq_constructor(test)
                constructor_2_construction_times[rmq_constructor].append(
                    monotonic_ns() - start_time
                )

        for rmq_constructor, times in constructor_2_construction_times.items():
            alg_2_times[rmq_constructor].append(mean(times))

    x_range = np.arange(*num_elems_interval)
    for rmq_constructor, times in alg_2_times.items():
        plt.plot(x_range, times, label=f"{rmq_constructor.__name__}")
    plt.xlabel(r"no. elements")
    plt.ylabel(r"average construction time (ns)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    benchmark_query_times()
