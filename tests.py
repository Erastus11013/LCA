from random import randint
from restricted_rmq import RestrictedRMQ
from rmq import RMQ
from datetime import datetime


# tests
def test_naive_rmq_construct_method(test, i, j):
    """test the naive method of constructing a range minimum query class
        @param: test: the array to be used in the construction of the range minimum query
        @param: i: the start of the interval
        @param: j: the end of the interval
        @return: a tuple of (argmin, min)"""
    r = RMQ(test)
    r.construct_naive_rmq()
    x = r[i: j]
    return x, r.array[x]


def test_sparse_table_algorithm_method(test, i, j):
    """test the sparse table algorithm for constructing a range minimum query class"""
    r = RMQ(test)
    r.construct_sparse_table()
    x = r[i: j]
    return x, r.array[x]


def test_segment_tree_construction_method(test, i, j):
    """test the segment tree construction of a range minimum query"""
    r = RMQ(test)
    r.construct_rmq_segment_tree()
    x = r[i: j]
    return x, r.array[x]


def test_restricted_rmq_construction_method(test, i, j):
    r = RestrictedRMQ(test)
    r.construct_restricted_rmq()
    r.norm = r._normalize_blocks()
    x = r.rmq(i, j)
    return x


def tests(size, fl, cap, n):
    """Testing
        @param: size: size of the random array to be constructed
        @param: fl: the minimum value of randint
        @param cap: the maximum value of randint
        @param: n: the number of times to run the tests"""
    for i in range(n):
        i = randint(0, size - 1)
        j = randint(i, size - 1)

        test = [randint(fl, cap) for _ in range(size)]

        t1 = datetime.now()
        arg_min1, minimum1 = test_naive_rmq_construct_method(test, i, j)
        print("min using naive rmq: ", minimum1, "ran in time", (datetime.now() - t1).total_seconds(), "seconds...")

        t2 = datetime.now()
        arg_min2, minimum2 = test_sparse_table_algorithm_method(test, i, j)
        print("min using sparse rmq: ", minimum2, "ran in time", (datetime.now() - t2).total_seconds(), "seconds...")
        t3 = datetime.now()

        arg_min3, minimum3 = test_segment_tree_construction_method(test, i, j)
        print("min using seg tree: ", minimum3, "ran in time", (datetime.now() - t3).total_seconds(), "seconds...")

        t4 = datetime.now()
        mininum4 = test_restricted_rmq_construction_method(test, i, j)
        print("min using restricted rmq: ", mininum4, "ran in time", (datetime.now() - t4).total_seconds(), "seconds")
        mininum5 = min(test[i:j + 1])

        assert minimum1 == mininum4 == minimum2 == minimum3 == mininum5, print((i, j), test)


if __name__ == '__main__':
    tests(9, -67890, 300000, 50)
