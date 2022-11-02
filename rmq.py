from abc import ABC, abstractmethod
from math import ceil
from sys import maxsize
from typing import Generic, Iterable, Optional

import numpy as np

from common import T
from treap import build_cartesian_tree, get_lca


class RMQ(Generic[T], ABC):
    def __init__(self, a: Iterable[T]):
        self.array = np.array(a)

    @abstractmethod
    def _query(self, start: int, stop: int) -> int:
        pass

    def __getitem__(self, item: slice | tuple) -> T:
        n = len(self.array)
        if isinstance(item, slice):
            stop = item.stop
            start = 0 if item.start is None else item.start
        else:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError()
            start, stop = item
        if start > stop:
            raise IndexError("make sure start <= stop")
        if stop is None or item.stop > n - 1:
            stop = n - 1
        elif stop < 0:
            stop += n
        return self._query(start, stop)

    def __repr__(self):
        return f"RMQ{repr(self.array)}"


class RMQSparseTable(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.lookup_table = self._construct()

    def _construct(self) -> np.ndarray:
        length = len(self.array)
        log2_length = length.bit_length() - 1
        lookup_table = np.full(
            (length, log2_length + 1), np.iinfo(np.int64).max, dtype=np.int64
        )
        lookup_table[:, 0] = np.arange(0, length)

        # break each index into powers of 2
        for pow2 in range(1, log2_length + 1):  # log(length)
            if (1 << pow2) > length:
                break
            for index in range(length):
                if (index + (1 << pow2) - 1) >= length:
                    break  # index + 2^j - 1
                else:
                    x, y = (
                        lookup_table[index, pow2 - 1],
                        lookup_table[index + (1 << (pow2 - 1)), pow2 - 1],
                    )
                    if self.array[x] < self.array[y]:
                        lookup_table[index, pow2] = x
                    else:
                        lookup_table[index, pow2] = y
        return lookup_table

    def _query(self, start: int, stop: int) -> T:
        """In this operation we can query on an interval or segment and
        return the answer to the problem on that particular interval."""

        length = (stop - start) + 1
        log2_length = length.bit_length() - 1
        if (
            self.array[self.lookup_table[start][log2_length]]
            <= self.array[
                self.lookup_table[start + length - (1 << log2_length)][log2_length]
            ]
        ):
            return self.lookup_table[start][log2_length]
        else:
            return self.lookup_table[stop - (1 << log2_length) + 1][log2_length]


class RMQPrecomputed(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.lookup_table = self._construct()

    def _construct(self) -> np.ndarray:
        """Trivial algorithm for RMQ
        For every pair of indices (i, j) store the value of RMQ(i, j) in a table M[0, N-1][0, N-1].
        Using an easy dynamic programming approach we can reduce the complexity to <O(N^2), O(1)>
        Uses O(N^2) space"""

        len_array = len(self.array)
        lookup_table = np.full(
            (len_array, len_array), np.iinfo(np.int64).max, dtype=np.int64
        )

        np.fill_diagonal(lookup_table, np.arange(0, len_array))

        for i in range(len_array):
            for j in range(i + 1, len_array):
                if self.array[lookup_table[i][j - 1]] < self.array[j]:
                    lookup_table[i][j] = lookup_table[i][j - 1]
                else:
                    lookup_table[i][j] = j
        return lookup_table

    def _query(self, start, stop):
        return self.lookup_table[start][stop]


class RMQSegmentTree(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.lookup_table = self._construct()

    def _construct(self) -> np.ndarray:
        """A segment tree or seg-tree is a basically a binary tree used for storing the intervals or segments.
        Each node in the segment tree represents an interval.
        Consider an array A of size N and a corresponding seg-tree T:
        The root of T will represent the whole array A[0:N-1].
        Each leaf in the seg-tree T will represent a single element A[i] such that 0 <= i < N.
        The internal nodes in the seg-tree tree T represent union of elementary intervals A[i:j] where 0 <= i < j < N.
        The root of the seg-tree will represent the whole array A[0:N-1].
        Then we will break the interval or segment into half and the two children of the root will represent the
        A[0:(N-1) / 2] and A[(N-1) / 2 + 1:(N-1)].
        So in each step we will divide the interval into half and the two children will represent the two halves.
        So the height of the segment tree will be log2N.
        There are N leaves representing the N elements of the array.
        The number of internal nodes is N-1. So total number of nodes are
        O(N) space: Array of size 2N."""

        n = len(self.array)
        return self.__construct_segment_tree(
            np.zeros(n << 1 + 1, dtype=np.int64), 0, 0, n - 1
        )

    def __construct_segment_tree(
        self, lookup_table: np.ndarray, current: int, start: int, stop: int
    ) -> np.ndarray:
        """Helper method to construct the segment tree"""

        if start == stop:  # we are at a leaf
            lookup_table[current] = start
        else:
            mid = (start + stop) >> 1

            left = (current << 1) + 1
            right = left + 1
            self.__construct_segment_tree(lookup_table, left, start, mid)
            self.__construct_segment_tree(lookup_table, right, mid + 1, stop)

            if self.array[lookup_table[left]] <= self.array[lookup_table[right]]:
                lookup_table[current] = lookup_table[left]
            else:
                lookup_table[current] = lookup_table[right]
        return lookup_table

    def _query_impl(
        self,
        curr: int,
        left_bound: int,
        right_bound: int,
        query_start: int,
        query_stop: int,
    ) -> Optional[T]:
        """To query on a given range, we need to check 3 conditions:
        range represented by a node is completely  inside  the given query
        range represented by a node is completely  outside the given query
        range represented by a node is partially   inside and partially outside the given query"""

        # range represented by a node is completely outside the given range
        if query_start > right_bound or query_stop < left_bound:
            return None

        # if the current interval is included in the query interval return M[curr]
        if query_start <= left_bound and right_bound <= query_stop:
            return self.lookup_table[curr]

        #  compute arg_min in the left and right interval
        mid = (left_bound + right_bound) >> 1
        p1 = self._query_impl(2 * curr + 1, left_bound, mid, query_start, query_stop)
        p2 = self._query_impl(
            2 * curr + 2, mid + 1, right_bound, query_start, query_stop
        )

        # find and return arg_min(self.A[i: j])// return the
        if p2 is None:
            return p1
        if p1 is None:
            return p2
        if self.array[p1] <= self.array[p2]:
            return p1
        else:
            return p2

    def _query(self, start: int, stop: int) -> int:
        return self._query_impl(0, 0, len(self.array) - 1, start, stop)


class RMQSqrtDecomposition(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.n_blocks = int(np.ceil(np.sqrt(len(self.array))))
        self.extended_array, self.lookup = self._construct()

    def _construct(self):
        extended_array = np.append(
            self.array, [0] * (self.n_blocks * self.n_blocks - len(self.array))
        )
        lookup = np.empty(self.n_blocks, dtype=np.int64)
        for block_index in range(0, self.n_blocks):
            offset = block_index * self.n_blocks
            lookup[block_index] = offset + np.argmin(
                extended_array[offset : (offset + self.n_blocks)]
            )
        return extended_array, lookup

    def _query(self, start: int, stop: int) -> int:
        start_block_index, start_block_offset = divmod(start, self.n_blocks)
        end_block_index, end_block_offset = divmod(stop, self.n_blocks)

        arg_min = start
        if start_block_index == end_block_index:
            for index in range(start, stop + 1):
                if self.array[index] < self.array[arg_min]:
                    arg_min = index
        else:
            for index in self.lookup[start_block_index + 1 : end_block_index]:
                if self.extended_array[index] < self.extended_array[arg_min]:
                    arg_min = index
            for block_offset in range(start_block_offset, self.n_blocks):
                if (
                    self.extended_array[
                        start_block_index * self.n_blocks + block_offset
                    ]
                    < self.extended_array[arg_min]
                ):
                    arg_min = start_block_index * self.n_blocks + block_offset
            for block_offset in range(end_block_offset + 1):
                if (
                    self.extended_array[end_block_index * self.n_blocks + block_offset]
                    < self.extended_array[arg_min]
                ):
                    arg_min = end_block_index * self.n_blocks + block_offset
        return arg_min


class RMQFischerHeun(RMQ):
    """
    https://link.springer.com/content/pdf/10.1007%2F11780441_5.pdf
    """

    def __init__(self, a):
        super().__init__(a)
        self.block_type_2_lookup_table: dict[
            int, RMQPrecomputed
        ] = {}  # sqrt(n) * 2n/lg n  #  o(n)
        self.block_size = (len(self.array).bit_length() - 1) >> 1
        self.block_count = ceil(len(self.array) / self.block_size)
        assert self.block_size * self.block_count >= len(self.array)
        (
            self.summary_rmq,
            self.block_argmin,
            self.block_types,
        ) = self._construct()  # 2n/ lg n * log ( 2n/ lg n )

    @staticmethod
    def compute_block_type(block) -> int:
        result = ""
        stack: list[int] = []
        for elem in block:
            while stack and stack[-1] > elem:
                stack.pop()
                result += "0"
            stack.append(elem)
            result += "1"
        return int(result[::-1], 2)

    def _compute_summary_rmq(
        self, block_types: np.ndarray
    ) -> tuple[RMQSparseTable, np.ndarray]:
        block_min = np.zeros(self.block_count, dtype=np.int32)
        block_argmin = np.zeros(self.block_count, dtype=np.int32)

        for block_index in range(self.block_count):
            in_block_rmq = self.block_type_2_lookup_table[block_types[block_index]]
            in_block_argmin = in_block_rmq[0 : self.block_size - 1]
            argmin_abs = block_index * self.block_size + in_block_argmin

            block_min[block_index] = self.array[argmin_abs]
            block_argmin[block_index] = argmin_abs

        return RMQSparseTable(block_min), block_argmin

    def _construct(
        self,
    ) -> tuple[RMQSparseTable, np.ndarray, np.ndarray]:
        block_types = np.zeros(self.block_count, dtype=np.int32)

        extended_array = np.append(
            self.array,
            [maxsize] * (self.block_count * self.block_size - len(self.array)),
        )
        for block_index in range(self.block_count):
            block_start = block_index * self.block_size
            block_end = block_start + self.block_size

            block_type = self.compute_block_type(extended_array[block_start:block_end])
            block_types[block_index] = block_type

            if block_type not in self.block_type_2_lookup_table:
                self.block_type_2_lookup_table[block_type] = RMQPrecomputed(
                    extended_array[block_start:block_end]
                )

        return self._compute_summary_rmq(block_types) + (block_types,)

    def _get_lookup_table_for_block_at(self, block_index) -> RMQPrecomputed:
        return self.block_type_2_lookup_table[self.block_types[block_index]]

    def _query(self, start: int, stop: int) -> int:
        if start > stop:
            start, stop = stop, start

        start_block_index, start_block_offset = divmod(start, self.block_size)
        end_block_index, end_block_offset = divmod(stop, self.block_size)

        if start_block_index == end_block_index:
            lookup_table = self._get_lookup_table_for_block_at(start_block_index)
            return (start_block_index * self.block_size) + lookup_table[
                start_block_offset:end_block_offset
            ]
        else:
            lookup_table = self._get_lookup_table_for_block_at(start_block_index)
            start_block_argmin = (
                start_block_index * self.block_size
                + lookup_table[start_block_offset : self.block_size - 1]
            )

            lookup_table = self._get_lookup_table_for_block_at(end_block_index)
            end_block_argmin = (
                end_block_index * self.block_size + lookup_table[0:end_block_offset]
            )

            if end_block_index - start_block_index > 1:
                summary_argmin = self.block_argmin[
                    self.summary_rmq[
                        int(start_block_index + 1) : int(end_block_index - 1)
                    ]
                ]
                argmin = (
                    summary_argmin
                    if self.array[summary_argmin] < self.array[start_block_argmin]
                    else start_block_argmin
                )
            else:
                argmin = start_block_argmin

            return (
                end_block_argmin
                if self.array[end_block_argmin] < self.array[argmin]
                else argmin
            )


class RMQCartesianTreeLCA(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.root = build_cartesian_tree(a)
        # assert list(a) == [node.value for node in self.root.inorder()]

    def _query(self, start: int, stop: int) -> T:
        return get_lca(self.root, self.array[start], self.array[stop]).value_index
