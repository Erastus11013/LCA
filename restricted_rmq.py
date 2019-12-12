from rmq import RMQ
from treap import CartesianTreeRMQ
import numpy as np
from math import log2, ceil, inf
from sys import maxsize
from copy import deepcopy


class RestrictedRMQ(RMQ):
    INFINITY = maxsize

    def __init__(self, a):
        super().__init__(a)               # ------------- #
        self.E = None                     # 2n - 1
        self.depths = None                # 2n - 1
        self.R = None                     # n
        self.A = None                     # n
        self.B = None                     # n
        self.T = None                     # n
        self.lookup = None                # sqrt(n) * 2n/lg n  #  o(n)
        self.block_count = 0              # 1
        self.block_size = 0               # 1
        self.max_lookup = 0               # 1
        self.lca = True                   # 1
        self.sparse = None                # 2n/ lg n * log ( 2n/ lg n )
        self._build_cartesian_tree()      # 3n ... but is deleted

    def _build_cartesian_tree(self):
        c = CartesianTreeRMQ(self.array)
        c.build_cartesian_tree()  # step 1
        self.E, self.depths, self.R = c.euler_tour()  # step 2

    @staticmethod
    def get_block_type(block, blocksize) -> int:
        """To determine the type of block which x has, then you need to consider elements
        in the range[0 ... n] if using 1-based indexing. For example if the blocksize is 4, then block must include the
        last element of the previous block, for the first block, send in an array
        """
        bit = 0
        k = blocksize - 1
        for i in range(1, blocksize + 1):
            if block[i] > block[i - 1]:
                bit = bit | (1 << k)  # set bit at position k to 1
            k -= 1
        return bit

    @staticmethod
    def get_first_block_type(block, blocksize) -> int:
        w = 0
        k = blocksize - 2
        for i in range(1, blocksize):
            if block[i] > block[i - 1]:
                w = w | (1 << k)
            k -= 1
        return w

    @staticmethod
    def get_pos(i, j, blocksize):
        return sum(range(blocksize - i - 1), blocksize) + j - (blocksize - i)

    def _normalize_blocks(self):
        L = list(deepcopy(self.depths))
        n = len(self.depths)
        for i in range(2, n):
            L[i] = self.depths[i] - self.depths[i - 1]
            if L[i] < 0:
                L[i] = 0
        return L

    def rmq_compressed_lookup(self, blocksize, max_lookup, shift=None, arr=None, lookup=None):
        """"""
        if lookup is None:
            lookup = np.zeros(max_lookup, dtype='int32')
        if shift is None:
            shift = 0
        if arr is None:
            arr = self.array

        for i in range(blocksize):  # O(n)
            k = shift + self.get_pos(i, i, blocksize)
            lookup[k] = i

        for i in range(blocksize):
            for j in range(i + 1, blocksize):
                k = shift + self.get_pos(i, j, blocksize)
                if arr[lookup[k - 1]] < arr[j]:
                    lookup[k] = lookup[k - 1]
                else:
                    lookup[k] = j

    def populate_sparse_table(self, b_count, b_size, lookup, max_lookup, T, save=True):
        m = int(log2(b_count))

        A = np.zeros(b_count, dtype='int32')
        B = np.zeros(b_count, dtype='int32')

        sparse = np.full((b_count, m + 1), inf, dtype='int32', order='F')

        for i in range(0, b_count):
            # computing the min in each block
            pos = T[i]  # the lookup table for i starts at pos
            min_ind = pos + self.get_pos(0, b_size - 1, b_size)

            index = lookup[min_ind]  # when i = 0, j = b_size - 1; [0][b_size - 1] =
            # 0 * b_comb + b_size - 1
            index = i * b_size + index  # absolute index in the depths array = i * b_size + index

            A[i] = self.E[index]
            B[i] = index

            # in the end, construct the sparse table
        self.construct_sparse_table(A, sparse)
        if save:
            self.sparse = sparse
            self.A = A
            self.B = B

    def build_restricted_rmq(self, save=True):
        """"""
        n = len(self.depths)

        b_size = ceil(log2(n)) // 2
        b_count = ceil(n / b_size)

        max_lookup = (b_size * (b_size + 1)) >> 1  # summation formula for block size of 2, possible blocks [0, 1, 2, 3]

        size_lt = max_lookup * (1 << b_size)  # size of lookup table

        lookup = np.full(size_lt, -1, dtype='int32')  # the lookup table [0, 1, 2, 3 ... b_count]

        T = np.zeros(b_count, dtype='int32')

        # initialization
        b_type = self.get_first_block_type(self.depths[: b_size], b_size)  # get b_type first element
        self.rmq_compressed_lookup(b_size, max_lookup, 0, self.depths, lookup)
        T[0] = b_type * max_lookup

        for i in range(1, b_count):
            b_start = i * b_size
            b_end = min(b_start + b_size - 1, n - 1)
            if b_end - b_start < b_size - 1:  # last block is shorter
                new_block = np.append(self.depths[b_start - 1: b_end + 1],
                                      [self.INFINITY] * (b_end - b_start + 1))
                b_type = self.get_block_type(new_block, b_size)
            else:
                b_type = self.get_block_type(self.depths[b_start - 1: b_end + 1], b_size)

            pos = b_type * max_lookup  # position of b_type in the lookup array
            if lookup[pos] == -1:
                self.rmq_compressed_lookup(b_size, max_lookup, pos, self.depths[b_start: b_end + 1], lookup)
                T[i] = pos
            else:
                T[i] = pos

        self.populate_sparse_table(b_count, b_size, lookup, max_lookup, T, save)
        if save:
            self.block_size = b_size
            self.block_count = b_count
            self.max_lookup = max_lookup
            self.T = T
            self.lookup = lookup

    def _query_restricted_rmq(self, a, b, lca=True):
        """ The overall strategy here is straight from the paper.  We look up
            the blocks that contain u and v (taking care to consider v being
            the inclusive endpoint even though the API understands it to be
            exclusive), then use a sparse_rmq search over the super array among
            blocks strictly between u's and v's blocks, and do naive_rmq
            searches in the representative structures for the blocks with u's
            and v's blocks' shapes.
            Most of what's below is offset math, and
            isn't all that interesting."""

        i, j = self.R[a], self.R[b]

        x = i // self.block_size  # block containing i
        y = j // self.block_size   # block containing y
        u = i % self.block_size  # starting position of i in block x
        v = (j % self.block_size)  # end position of j in block y

        if x == y:   # if they are in the same block.. use lookup

            ipos = self.T[x] + self.get_pos(u, v, self.block_size)
            in_block_min = x * self.block_size + self.lookup[ipos]
            shift = self.block_size * x
            actual_pos = shift + in_block_min
            if lca:
                return self.E[actual_pos]
            else:
                return actual_pos

        else:
            # first block minimum calculation
            fpos = self.T[x] + self.get_pos(u, self.block_size - 1, self.block_size)
            fmin = x * self.block_size + self.lookup[fpos]

            # last block min

            lpos = self.T[y] + self.get_pos(0, v, self.block_size)
            lmin = y * self.block_size + self.lookup[lpos]

            bmin = fmin if self.E[fmin] < self.E[lmin] else lmin
            # super array minimum
            if y - x > 1:  # if there are blocks between x, y
                superpos = self.B[self._query_sparse_table(x + 1, y - 1, self.A, self.sparse)]
                opos = bmin if self.E[bmin] < self.E[superpos] else superpos

            else:
                opos = bmin

            if lca:
                return self.E[opos]
            else:
                return opos

    def rmq(self, a, b):
        return self._query_restricted_rmq(a, b)


if __name__ == '__main__':
    test = [8, 7, 2, 8, 6, 9, 4, 5]
    rq = RestrictedRMQ(test)
    rq.build_restricted_rmq()
    rq.norm = rq._normalize_blocks()
    x = rq.rmq(1, 5)
    print(x)
