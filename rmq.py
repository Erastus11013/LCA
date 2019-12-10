import numpy as np
from random import randint
from math import inf
from math import log2


class RMQ:
    def __init__(self, a):
        self.array = a
        self.M = []
        self.algo_used = 'st'

    def construct_naive_rmq(self):
        """Trivial algorithm for RMQ
        For every pair of indices (i, j) store the value of RMQ(i, j) in a table M[0, N-1][0, N-1].
        Using an easy dynamic programming approach we can reduce the complexity to <O(N2), O(1)> """
        n = len(self.array)
        self.M = np.full((n, n), inf, dtype='int32')
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
        So, the overall complexity of the algorithm is <O(N logN), O(1)>"""
        n = len(self.array)
        m = int(log2(n))
        self.M = np.full((n, m + 1), inf, dtype='int32')
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
        n = len(self.array)
        self._initialize(1, 0, n)

    def construct_rmq_sqrt(self):
        """An interesting idea is to split the vector in sqrt(N) pieces.
        We will keep in a vector M[0, sqrt(N)-1] the position for the minimum value for each section.
        M can be easily preprocessed in O(N):"""

    def _query_sparse_table(self, low, high):
        length = (high - low) + 1
        k = int(log2(length))
        if self.array[self.M[low][k]] <= self.array[self.M[low + length - (1 << k)][k]]:
            return self.M[low][k]
        else:
            return self.M[high - (1 << k) + 1][k]

    def _initialize(self, current, low, high):
        if low == high:
            self.M[current] = low
        else:
            mid = low + high // 2
            left = Node(current.key * 2)
            right = Node(current.key * 2 + 1)
            self._initialize(left, low, mid)
            self._initialize(right, mid + 1, high)

            if self.array[self.M[2 * current.key]] <= self.array[self.M[2 * current.key + 1]]:
                self.M[current.key] = self.M[current.key * 2]
            else:
                self.M[current.key] = self.M[2 * current.key + 1]

    def _query_segment_tree(self, node, low, high, i, j):

        # if the current interval doesn’t intersect  the query interval
        if i > high or j < low:
            return False

        # if the current interval is included in the query interval return M[node]
        if low >= i and high <= j:
            return self.M[node]

        #  compute argmin in the left and right interval
        p1 = self._query_segment_tree(2 * node, low, (low + high) / 2, i, j)
        p2 = self._query_segment_tree(2 * node + 1, (low + high) / 2 + 1, high, i, j)

        # find and return argmin(self.A[i: j])// return the
        if not p1:
            return p2
        elif not p2:
            return p1
        if self.array[p1] <= self.array[p2]:
            return p1
        else:
            return p2


    def rmq(self, i, j):
        if self.algo_used == 'a':
            return self.M[i][j]
        elif self.algo_used == 'st':  # sparse table algorithm
            return self._query_sparse_table(i, j)
        if self.algo == 't':
            return self._query_segment_tree(1, 0, len(self.array)- 1, i, j)

    def __getitem__(self, item):
        # item is a slice object
        assert isinstance(item, slice), print(item)
        start = 0 if item.start is None else item.start
        stop = len(self.array) - 1 if item.stop is None else item.stop
        return self.rmq(start, stop)

    def __repr__(self):
        if len(self.array) > 10:
            return 'RMQ [' + '  '.join(map(str, self.array[: 10])) + ' ... ' + str(self.array[-1]) + ']'
        else:
            return 'RMQ [' + '  '.join(map(str, self.array)) + ']'


def test_rmq(test, i, j):
    r = RMQ(test)
    r.construct_rmq_segment_tree()
    x = r[i: j]
    return x, r.array[x]


def tests(size, fl, cap, n):
    # test = [7, 5, 0, 30, 25, 7, 9, 18, 13, 3, 20, 2, 12, 4, 13, 13]
    for i in range(n):
        i = randint(0, size - 1)
        j = randint(i, size - 1)
        test = [randint(fl, cap) for _ in range(size)]
        arg_min, minimum = test_rmq(test, i, j)
        assert minimum == min(test[i:j + 1])


if __name__ == '__main__':
    tests(90, 0, 30, 100)
