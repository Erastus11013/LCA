from rmq import RMQ
from treap import CartesianTreeRMQ
import numpy as np
from math import log2, ceil, inf
from copy import deepcopy
import bitty


class RestrictedRMQ(RMQ):

    def __init__(self, a):
        super().__init__(a)
        self.E = None
        self.depths = None
        self.R = None
        self.block_count = 0
        self.block_size = 0
        self.block_comb = 0
        self.lca = True
        self.sparse = None
        self._build_cartesian_tree()

    def _build_cartesian_tree(self):
        c = CartesianTreeRMQ(self.array)
        c.build_cartesian_tree()  # step 1
        self.E, self.depths, self.R = c.euler_tour()  # step 2

    def _normalize_blocks(self):
        L = deepcopy(self.depths)
        n = len(self.depths)
        for i in range(2, n):
            self.depths[i] = L[i] - L[i - 1]
            if self.depths[i] < 0:
                self.depths[i] = 0

    def _build_look_up_tables(self):
        n = len(self.depths)
        b_size = ceil(log2(n)) // 2
        b_count = ceil(n / b_size)
        b_comb = int((b_size * (b_size + 1)) / 2)  # summation formula
        block_types = range(2**b_size)  # e.g for block size of 2, possible blocks [0, 1, 2, 3]

        self.M = np.zeros(int(b_count * b_comb), dtype='int32')
        for i in range(b_count):
            start = i * b_comb
            row = 0

            # computing the minimum in each block.
            # we
            for j in range(b_size):  # j is the column number
                minimum = self.depths[i * b_size]
                for k in range(j, b_size):  # k the row number
                    if self.depths[(i * b_size) + k - 1] == minimum:
                        if minimum == 1:
                            self.M[start + row] = j
                            row += 1
                        else:
                            self.M[start + row] = k
                            row += 1
                    elif self.depths[(i * b_size) + k - 1] < minimum:
                        self.M[start + row] = k
                        row += 1
                    else:
                        self.M[start + row] = j
                        row += 1
        self.block_count = b_count
        self.block_size = b_size
        self.block_comb = b_comb
        print("size of self.M", len(self.M))

    def _compute_min_out_of_block(self, b_count, b_size, b_comb):
        """assumes that lookup_tables have already been created"""

        m = int(log2(self.block_count))
        self.A = np.zeros(self.block_count, dtype='int32')
        self.B = np.zeros(self.block_count, dtype='int32')
        sparse = np.full((self.block_count, m + 1), inf, dtype='int32', order='F')

        for i in range(0, self.block_count):
            # Get (0, b_size) in each block, that will stored at this location
            idx = self.M[self.block_size - 1 + i * self.block_comb]  # in indexing, x[i, j] = i * len(one row) + j
            idx = (i * self.block_size) + idx

            self.A[i] = self.E[idx - 1]
            self.B[i] = idx

        self.construct_sparse_table(array=self.A, sparse=sparse)
        self.sparse = sparse

    def build_restricted_rmq(self):
        self._normalize_blocks()
        self._build_look_up_tables()
        self._compute_min_out_of_block(self.block_count, self.block_size, self.block_comb)

    def _query_restricted_rmq(self, a, b):
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

        start_outer = i // self.block_size
        end_outer = j // self.block_size
        in_start = i % self.block_size
        in_end = (j % self.block_size)
        in_start_index = -1
        in_end_index = -1

        if start_outer == end_outer:  # we are in the same block
            left = self.block_size - in_start
            in_start_index = start_outer * self.block_comb + (self.block_comb - ((left * (left + 1)) // 2)) + in_end

        else:
            # Border case i.e. i is multiple of self.block_count e.g. 0,5,10,15
            if in_start:  # in_start != 0
                left = ((in_start + 1) * (2 * self.block_size - in_start)) // 2
                in_start_index = (start_outer * self.block_comb) + left - 1
                start_outer += 1

            # Border case i.e. j is 4,9,14,19 boundary of blocksize
            if in_end != self.block_size - 1:
                in_end_index = ((end_outer * self.block_comb) - 1) + in_end + 1  # +1/-1 because we are counting from 0
                end_outer -= 1

        first_block_min = self.E[self.M[in_start_index] + ((i // self.block_size) * self.block_size)]\
            if (in_start_index != -1) else -1
        last_block_min = self.E[self.M[in_end_index] + ((j // self.block_size) * self.block_size)]\
            if (in_end_index != -1) else -1
        outer_block_min = self._query_sparse_table(start_outer, end_outer, self.A, self.sparse)\
            if (end_outer >= start_outer) else -1
        if self.lca:
            minimum = inf if first_block_min == -1 else first_block_min
            if last_block_min != -1 and last_block_min <= minimum:
                minimum = last_block_min
            if outer_block_min != -1 and outer_block_min <= minimum:
                minimum = outer_block_min
            return minimum
        else:
            minimum = inf if first_block_min == -1 else self.array[first_block_min]
            minimum_index = first_block_min
            if last_block_min != -1 and self.array[last_block_min] <= minimum:
                minimum = self.array[last_block_min]
                minimum_index = last_block_min
            if outer_block_min != -1 and self.array[outer_block_min] <= minimum:
                minimum = self.array[outer_block_min]
                minimum_index = outer_block_min
            return minimum_index

    def rmq(self, a, b):
        return self._query_restricted_rmq(a, b)


if __name__ == '__main__':
    test = [9, 1, 7, 5, 8, 12, 10, 20, 15, 18, 5, 12, 14, 98, 75, 19]
    rq = RestrictedRMQ(test)
    rq.build_restricted_rmq()
    print(rq.rmq(0, 14))
