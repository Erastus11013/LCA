import numpy as np
from random import randint
from math import inf, log2
from datetime import datetime


class RMQ:
    """Range Minimum Query class
    Has three methods to construct it
    construct seg_tree is so far the fastest
    @param: a: an array of integers, floats, or complex numbers"""

    def __init__(self, a):
        self.array = a
        self.M = []
        self.algo_used = 't'

    def construct_naive_rmq(self):
        """Trivial algorithm for RMQ
        For every pair of indices (i, j) store the value of RMQ(i, j) in a table M[0, N-1][0, N-1].
        Using an easy dynamic programming approach we can reduce the complexity to <O(N^2), O(1)>
        Uses O(N^2) space"""

        n = len(self.array)
        self.algo_used = 'n'
        self.M = np.full((n, n), inf, dtype='int32', order='F')
        for i in range(n):   # O(n)
            self.M[i][i] = i

        for i in range(n):
            for j in range(i + 1, n):
                if self.array[self.M[i][j - 1]] < self.array[j]:
                    self.M[i][j] = self.M[i][j-1]
                else:
                    self.M[i][j] = j

    def construct_sparse_table(self):
        """preprocess RMQ for sub arrays of length 2k using dynamic programming.
        We will keep an array M[0, N-1][0, logN]
        where M[i][j] is the index of the minimum value in the sub array starting at i having length 2^j.
        So, the overall complexity of the algorithm is <O(N logN), O(1)>
        Uses O(N logN) space"""

        self.algo_used = 'st'
        n = len(self.array)
        m = int(log2(n))
        self.M = np.full((n, m + 1), inf, dtype='int32', order='F')
        for i in range(n):  # intervals of length 1
            self.M[i][0] = i
        for j in range(1, m + 1):  # log(n)
            if (1 << j) > n:  # 1 << j == 2^j
                break
            for i in range(n):
                if (i + (1 << j) - 1) >= n:
                    break  # i + 2^j - 1
                else:
                    if self.array[self.M[i][j - 1]] < self.array[self.M[i + (1 << (j - 1))][j - 1]]:
                        self.M[i][j] = self.M[i][j - 1]
                    else:
                        self.M[i][j] = self.M[i + (1 << (j - 1))][j - 1]

    def construct_rmq_segment_tree(self):
        """A segment tree or segtree is a basically a binary tree used for storing the intervals or segments.
        Each node in the segment tree represents an interval.
        Consider an array A of size N and a corresponding segtree T:
        The root of T will represent the whole array A[0:N-1].
        Each leaf in the segtree T will represent a single element A[i] such that 0 <= i < N.
        The internal nodes in the segtree tree T represent union of elementary intervals A[i:j] where 0 <= i < j < N.
        The root of the segtree will represent the whole array A[0:N-1].
        Then we will break the interval or segment into half and the two children of the root will represent the
        A[0:(N-1) / 2] and A[(N-1) / 2 + 1:(N-1)].
        So in each step we will divide the interval into half and the two children will represent the two halves.
        So the height of the segment tree will be log2N.
        There are N leaves representing the N elements of the array.
        The number of internal nodes is N-1. So total number of nodes are
        O(N) space: Array of size 2N."""

        self.algo_used = 't'
        n = len(self.array)
        max_size = n << 1 + 1
        self.M = np.array([0] * max_size)
        self._initialize(0, 0, n-1)

    def construct_rmq_sqrt(self):
        """An interesting idea is to split the vector in sqrt(N) pieces.
        We will keep in a vector M[0, sqrt(N)-1] the position for the minimum value for each section.
        M can be easily preprocessed in O(N)"""

    def _query_sparse_table(self, low, high):
        """In this operation we can query on an interval or segment and
         return the answer to the problem on that particular interval."""
        length = (high - low) + 1
        k = int(log2(length))
        if self.array[self.M[low][k]] <= self.array[self.M[low + length - (1 << k)][k]]:
            return self.M[low][k]
        else:
            return self.M[high - (1 << k) + 1][k]

    def _initialize(self, current, low, high):
        """Helper method to construct the segment tree"""
        if low == high:  # we are at a leaf
            self.M[current] = low
        else:
            mid = (low + high) >> 1

            left = current * 2 + 1
            right = current * 2 + 2
            self._initialize(left, low, mid)
            self._initialize(right, mid + 1, high)
            if self.array[self.M[left]] <= self.array[self.M[right]]:
                self.M[current] = self.M[left]
            else:
                self.M[current] = self.M[right]

    def _query_segment_tree(self, curr, low, high, i, j):
        """To query on a given range, we need to check 3 conditions:
            range represented by a node is completely inside the given range
            range represented by a node is completely outside the given range
            range represented by a node is partially inside and partially outside the given range"""

        # range represented by a node is completely outside the given range
        if i > high or j < low:
            return -1

        # if the current interval is included in the query interval return M[curr]
        if i <= low and high <= j:
            return self.M[curr]

        #  compute arg_min in the left and right interval
        mid = (low + high) >> 1
        p1 = self._query_segment_tree(2 * curr + 1, low, mid, i, j)
        p2 = self._query_segment_tree(2 * curr + 2, mid + 1, high, i, j)

        # find and return arg_min(self.A[i: j])// return the
        if p2 == -1:
            return p1
        if p1 == -1:
            return p2
        if self.array[p1] <= self.array[p2]:
            return p1
        else:
            return p2

    def rmq(self, i, j):
        """Return the argmin in the range [i:j] of the array"""
        if self.algo_used == 'n':
            return self.M[i][j]
        elif self.algo_used == 'st':  # sparse table algorithm
            return self._query_sparse_table(i, j)
        if self.algo_used == 't':
            return self._query_segment_tree(0, 0, len(self.array) - 1, i, j)

    @property
    def algorithm_used(self):
        if self.algo_used == 't':
            return 'Segment tree'
        elif self.algo_used == 'st':
            return 'Sparse table algorithm'
        elif self.algo_used == 'n':
            return 'Naive dynamic programming algorithm'

    def __getitem__(self, item):
        """Assumes item is a slice object
        returns the argmin of the array in range[start: stop]
        Note that stop is included in the interval
        """
        n = len(self.array)
        stop = item.stop
        if stop is None or item.stop > n - 1:
            stop = n - 1
        else:
            if stop < 0:
                stop += n
        assert isinstance(item, slice), print(item)
        start = 0 if item.start is None else item.start

        if start > stop:
            raise IndexError("make sure start <= stop")
        return self.rmq(start, stop)

    def __repr__(self):
        """Simple repr of the array in the RMQ class"""
        if len(self.array) > 10:
            return 'RMQ [' + '  '.join(map(str, self.array[: 10])) + ' ... ' + str(self.array[-2]) + ']'
        else:
            return 'RMQ [' + '  '.join(map(str, self.array)) + ']'


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
    print(r.algorithm_used)
    x = r[i: j]
    return x, r.array[x]


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
        arg_min, minimum1 = test_naive_rmq_construct_method(test, i, j)
        print("min using naive rmq: ", minimum1, "ran in time", (datetime.now() - t1).total_seconds(), "seconds...")

        t2 = datetime.now()
        arg_min, minimum2 = test_sparse_table_algorithm_method(test, i, j)
        print("min using sparse rmq: ", minimum2, "ran in time", (datetime.now() - t2).total_seconds(), "seconds...")
        t3 = datetime.now()

        arg_min, minimum3 = test_segment_tree_construction_method(test, i, j)
        print("min using seg tree: ", minimum3, "ran in time", (datetime.now() - t3).total_seconds(), "seconds...")
        mininum4 = min(test[i:j + 1])
        assert minimum1 == mininum4 == minimum2 == minimum3


if __name__ == '__main__':
    tests(1000, 13, 200, 3)
