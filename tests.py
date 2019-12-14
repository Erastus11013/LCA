from random import randint
from restricted_rmq import RestrictedRMQ
from rmq import RMQ
from datetime import datetime


# tests
def test_naive_rmq_construct_method(test, i, j):
    """test the naive method of constructing a range minimum query class"""
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
    """test the restricted range minimum query class"""
    r = RestrictedRMQ(test)
    r.construct_restricted_rmq()
    x = r.rmq(i, j)
    return x


def tests(size, fl, cap, n):
    """Main testing function
    Uses random intervals and a random array for testing
    Args:
        size: size of the random array to be constructed
        fl: the minimum value of randint
        cap: the maximum value of randint
        n: the number of times to run the tests

    Returns:
        None

    Raises:
        AssertionError: if the minimum values returned by the five methods, (including inbuilt min) are not equal
    """

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
    tests(1000, -67890, 300000, 1)
