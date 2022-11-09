from abc import ABC, abstractmethod
from math import ceil
from sys import maxsize
from typing import Generic, Iterable, Optional

import numpy as np

from common import T
from treap import build_cartesian_tree, get_lca


class RMQ(Generic[T], ABC):
    def __init__(self, a: Iterable[T]):
        self.elements = np.array(a)

    @abstractmethod
    def _query(self, start: int, stop: int) -> int:
        """
        Private method to find the range min query of a certain range of the underlying array

        Parameters
        ----------
        start : int
                The inclusive start of the range to be queried
        stop  : int
                The inclusive end of the range to be queried


        Returns
        -------
        int
            The RMQ_(`start`, `stop`)

        Notes
        _____
        This method is only meant to be called by internal methods because it assumes neither
        `start` nor `stop` go out of bounds
        The user should find the RMQ using __getitem__

        """
        pass

    def __getitem__(self, item: slice | tuple) -> T:
        n = len(self.elements)
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

    def _arg_min(self, index_a: int, index_b: int, elements=None) -> int:
        """
        Helper method to return the argmin of two indices to elements

        Args:
            index_a: int
                    a within bounds index to `elements`
            index_b: int
                   a within bounds index to `elements`
            elements:
                    the elements to use to compute the argmin
                    if elements is None, then we default `self.elements`

        Returns:
            int
                The argmin of the two indices

        """
        if elements is None:
            elements = self.elements
        if elements[index_a] < elements[index_b]:
            return index_a
        return index_b


class RMQSparseTable(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.lookup_table = self._construct()

    def _construct(self) -> np.ndarray:
        length = len(self.elements)
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
                    lookup_table[index, pow2] = self._arg_min(
                        lookup_table[index, pow2 - 1],
                        lookup_table[index + (1 << (pow2 - 1)), pow2 - 1],
                    )
        return lookup_table

    def _query(self, start: int, stop: int) -> T:
        """
        Args:
            start: Inclusive start of the query range
            stop: Inclusive end of the query range

        Returns:

        """

        length = (stop - start) + 1
        log2_length = length.bit_length() - 1
        if (
            self.elements[self.lookup_table[start][log2_length]]
            <= self.elements[
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
        """Pre-computation of all the results of algorithm for RMQ"""

        len_array = len(self.elements)
        lookup_table = np.full(
            (len_array, len_array), np.iinfo(np.int64).max, dtype=np.int64
        )

        np.fill_diagonal(lookup_table, np.arange(0, len_array))

        for i in range(len_array):
            for j in range(i + 1, len_array):
                lookup_table[i][j] = self._arg_min(lookup_table[i][j - 1], j)
        return lookup_table

    def _query(self, start, stop):
        return self.lookup_table[start][stop]


class RMQSegmentTree(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.tree = self._construct()

    def _construct(self) -> np.ndarray:
        """
        Wrapper method which calls `self._construct_impl`

        Returns: an array representation of the segment tree

        """
        n = len(self.elements)
        tree = np.zeros(n << 1 + 1, dtype=np.int64)
        self._construct_impl(tree=tree, i=0, start=0, stop=n - 1)
        return tree

    def _construct_impl(
        self, *, tree: np.ndarray, i: int, start: int, stop: int
    ) -> None:
        """
        Recursively construct a segment tree for `self.elements`

        Parameters
        ----------
        tree        :np.ndarray
                     A table mapping nodes to elements
        i           :int
                     The index of current node in `tree`
        start       :int
                     The start of the range of elements we are considering
        stop        :int
                     The end of the range of elements we are considering

        Notes
        -----
            A segment tree is basically a binary tree used for storing the intervals or segments.
            Each node in the segment tree represents an interval.

            Consider an array A of size N and a corresponding segment tree T:
            The root of T will represent the whole array A[0:N-1].

            Each leaf in T will represent a single element A[i] such that 0 <= i < N.

            The internal nodes in the segment tree T represent union of
            elementary intervals A[i:j] where 0 <= i < j < N.

            Then we will break the interval or segment into half and the two children of the root will represent the
            A[0:(N-1) / 2] and A[(N-1) / 2 + 1:(N-1)].

            So in each step we will divide the interval into half and the two children will represent the two halves.
            So the height of the segment tree will be log_2(N).

            There are N leaves representing the N elements of the array.

            The number of internal nodes is N-1. So total number of nodes are O(N) space: Array of size 2N.

            If the current node is at index i, the left child of i will be at (2*i) + 1
                                               the right child of i will be at (2*i) + 2

        """

        if start == stop:  # we are at a leaf node
            tree[i] = start
        else:
            mid = (start + stop) >> 1
            self._construct_impl(tree=tree, i=2 * i + 1, start=start, stop=mid)
            self._construct_impl(tree=tree, i=2 * i + 2, start=mid + 1, stop=stop)
            tree[i] = self._arg_min(tree[2 * i + 1], tree[2 * i + 2])

    def _query_impl(
        self,
        i: int,
        bound_start: int,
        bound_stop: int,
        query_start: int,
        query_stop: int,
    ) -> Optional[int]:
        """
        Implementation of a query on a segment tree

        Parameters
        ---------
        i: int
            The index of the current node we are look at in ``self.tree``, i.e ``self.tree[i]``
        bound_start:
            Inclusive index of the start of the range represented by ``self.tree[i]``
        bound_stop:
            Inclusive index of the end of the range represented by ``self.tree[i]``
        query_start:
            Inclusive index of the start of the range we are querying
        query_stop:
            Inclusive index of the end of the range we are querying

        Returns
        -------
        None
            range represented by node `self.tree[i]` is completely outside the given query range
        int
            the argmin in the range [query_start: query_end]

        Notes
        -----
        To query on a given range, we need to check 3 conditions:
            1) range represented by node `self.tree[i]` is completely inside the given query
            2) range represented by node `self.tree[i]` is completely outside the given query
            3) range represented by node `self.tree[i]` is partially inside and partially outside the given query
        """

        # range represented by node `self.tree[i]` is completely outside the given query
        if query_start > bound_stop or query_stop < bound_start:
            return None

        # range represented by node `self.tree[i]` is completely outside the given query
        if query_start <= bound_start and bound_stop <= query_stop:
            return self.tree[i]

        # range represented by node `self.tree[i]` is partially inside and partially outside the given query
        mid = (bound_start + bound_stop) >> 1
        argmin_left = self._query_impl(
            2 * i + 1, bound_start, mid, query_start, query_stop
        )
        argmin_right = self._query_impl(
            2 * i + 2, mid + 1, bound_stop, query_start, query_stop
        )

        if argmin_right is None:
            return argmin_left
        if argmin_left is None:
            return argmin_right
        return self._arg_min(argmin_left, argmin_right)

    def _query(self, start: int, stop: int) -> int:
        return self._query_impl(0, 0, len(self.elements) - 1, start, stop)


class RMQSqrtDecomposition(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.sqrt = int(np.ceil(np.sqrt(len(self.elements))))
        self.extended_array, self.lookup = self._construct()

    def _construct(self):
        extended_array = np.append(
            self.elements, [0] * (self.sqrt * self.sqrt - len(self.elements))
        )
        lookup = np.empty(self.sqrt, dtype=np.int64)
        for block_index in range(0, self.sqrt):
            offset = block_index * self.sqrt
            lookup[block_index] = offset + np.argmin(
                extended_array[offset : (offset + self.sqrt)]
            )
        return extended_array, lookup

    def _query(self, start: int, stop: int) -> int:
        start_block_index, start_block_offset = divmod(start, self.sqrt)
        end_block_index, end_block_offset = divmod(stop, self.sqrt)

        arg_min = start
        if start_block_index == end_block_index:
            for index in range(start, stop + 1):
                if self.elements[index] < self.elements[arg_min]:
                    arg_min = index
        else:
            for index in self.lookup[start_block_index + 1 : end_block_index]:
                if self.extended_array[index] < self.extended_array[arg_min]:
                    arg_min = index
            for block_offset in range(start_block_offset, self.sqrt):
                if (
                    self.extended_array[start_block_index * self.sqrt + block_offset]
                    < self.extended_array[arg_min]
                ):
                    arg_min = start_block_index * self.sqrt + block_offset
            for block_offset in range(end_block_offset + 1):
                if (
                    self.extended_array[end_block_index * self.sqrt + block_offset]
                    < self.extended_array[arg_min]
                ):
                    arg_min = end_block_index * self.sqrt + block_offset
        return arg_min


class RMQFischerHeun(RMQ):
    """

    Notes
    -----
        ``block_type_2_lookup_table`` maps each block type to a lookup table.

        We have 4^(b) block-level RMQ structures, where the space and time complexity of ach
        is O(b^2)

        We set b = (1/4) log_2 n

        Time complexity for building ``block_type_2_lookup_table`` is the sum of:
            * O(n), for computing block types
            * O(4^((1/4) log_2 n) * ((1/4) log_2 n)) * ((1/4) log_2 n)) = O(sqrt(n) * log_2(n)^2) = O(n)
            = O(n)

        Time complexity for building ``summary_rmq``
            * O(n/b log (n/b)) = O((n/log_2(n)) * log_2((n/log_2(n))) = O(n)
        https://link.springer.com/content/pdf/10.1007%2F11780441_5.pdf
    """

    def __init__(self, a):
        super().__init__(a)
        self.block_type_2_lookup_table: dict[int, RMQPrecomputed] = {}
        self.block_size = (len(self.elements).bit_length()) >> 2
        self.block_count = ceil(len(self.elements) / self.block_size)
        assert self.block_size * self.block_count >= len(self.elements)
        (
            self.summary_rmq,
            self.block_argmin,
            self.block_types,
        ) = self._construct()  # 2n/ lg n * log ( 2n/ lg n )

    @staticmethod
    def _compute_block_type(block) -> int:
        result = 0
        stack: list[int] = []
        for elem in block:
            while stack and stack[-1] > elem:
                stack.pop()
                result = result << 1
            stack.append(elem)
            result = (result << 1) | 1
        while stack:
            stack.pop()
            result = result << 1
        return result

    def _compute_summary_rmq(
        self, block_types: np.ndarray
    ) -> tuple[RMQSparseTable, np.ndarray]:
        block_min = np.zeros(self.block_count, dtype=np.int32)
        block_argmin = np.zeros(self.block_count, dtype=np.int32)

        for block_index in range(self.block_count):
            in_block_rmq = self.block_type_2_lookup_table[block_types[block_index]]
            in_block_argmin = in_block_rmq[0 : self.block_size - 1]
            argmin = block_index * self.block_size + in_block_argmin

            block_min[block_index] = self.elements[argmin]
            block_argmin[block_index] = argmin

        return RMQSparseTable(block_min), block_argmin

    def _construct(
        self,
    ) -> tuple[RMQSparseTable, np.ndarray, np.ndarray]:
        block_types = np.zeros(self.block_count, dtype=np.int32)

        extended_array = np.append(
            self.elements,
            [maxsize] * (self.block_count * self.block_size - len(self.elements)),
        )
        for block_index in range(self.block_count):
            block_start = block_index * self.block_size
            block_end = block_start + self.block_size

            block_type = self._compute_block_type(extended_array[block_start:block_end])
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
                    if self.elements[summary_argmin] < self.elements[start_block_argmin]
                    else start_block_argmin
                )
            else:
                argmin = start_block_argmin

            return (
                end_block_argmin
                if self.elements[end_block_argmin] < self.elements[argmin]
                else argmin
            )


class RMQCartesianTreeLCA(RMQ):
    def __init__(self, a: Iterable[T]):
        super().__init__(a)
        self.root = build_cartesian_tree(a)
        # assert list(a) == [node.value for node in self.root.inorder()]

    def _query(self, start: int, stop: int) -> T:
        return get_lca(self.root, self.elements[start], self.elements[stop]).value_index
