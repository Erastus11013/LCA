from datetime import datetime
from random import randint
from sys import maxsize

import numpy as np
import pytest

from rmq import (RMQCartesianTreeLCA, RMQFischerHeun, RMQPrecomputed,
                 RMQSegmentTree, RMQSparseTable, RMQSqrtDecomposition)


@pytest.fixture
def param_from_fixture():
    num_elems = 100
    low = -67890
    high = 300000
    num_iters = 100
    return num_elems, low, high, num_iters


def test_equality(param_from_fixture):
    max_num_elems, low, high, num_iters = param_from_fixture
    rmq_constructors = (
        RMQPrecomputed,
        RMQSegmentTree,
        RMQSparseTable,
        RMQSqrtDecomposition,
        RMQFischerHeun,
        RMQCartesianTreeLCA,
    )
    # for num_elems in range(10, max_num_elems):
    num_elems = max_num_elems
    for _ in range(num_iters):
        start = randint(0, num_elems - 1)
        stop = randint(start, num_elems - 1)

        test = np.random.randint(low, high, num_elems + 1)
        expected_min = test[start : stop + 1].min(initial=maxsize)
        print()
        for rmq_constructor in rmq_constructors:
            start_time = datetime.now()
            argmin = rmq_constructor(test)[start:stop]
            duration = (datetime.now() - start_time).total_seconds()
            assert (
                test[argmin] == expected_min
            ), f"{rmq_constructor.__name__} failed: expected {expected_min} got {test[argmin]} for rmq{(start, stop)} {test}"
            print(f"{rmq_constructor.__name__:<30} {duration:10.7f} seconds")
