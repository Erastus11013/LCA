from rmq import RMQ
from treap import CartesianTreeRMQ
import numpy as np
from math import log2, ceil, inf
from sys import maxsize


# author: Erastus Murungi
# email: erastusmurungi@gmail.com


class RestrictedRMQ(RMQ):
    """
    Subclasses RMQ
    The Range-Minimum-Query-Problem is to preprocess an array such that the position of the minimum element
    between two specified indices can be obtained efficiently.
    It consumes 9n + O(√n log^2 n) space
    It has O(n) pre processing time and O(1) time queries
    Attributes:
        E: stores elements of array after an euler tour of the Cartesian Tree of array
        D: an array of the depths of nodes in the cartesian tree of array
        R: the representative array: R[i] stores the position of the first occurrence of array[i] in the euler tour
            of the cartesian tree of array
        A: A[i] stores the minimum element in the ith block of the depths array
        B: B[i] stores the position of A[i] in the E and depths arrays
        lookup: a lookup table for each of the normalized blocks of depth array
        blocksize: the size of a block of the the depths array.
            The size of one block is usually 1/2 lg n
        blockcount: the number of blocks the depths array can be divided into.
            This number is equal to 2n/block_size
    A better RMQ structure can be build using the Fischer-Heun algorithm. The link for the paper is here:
        https://link.springer.com/content/pdf/10.1007%2F11780441_5.pdf

    """
    INFINITY = maxsize

    def __init__(self, a):
        """Initializes the class using an array"""
        super().__init__(a)
        self.E = None  # 2n - 1
        self.D = None  # 2n - 1
        self.R = None  # n
        self.A = None  # n
        self.B = None  # n
        self.T = None  # n
        self.lookup = None  # sqrt(n) * 2n/lg n  #  o(n)
        self.blocksize = 0  # 1
        self.blockcount = 0  # 1
        self.ltinterval = 0  # 1
        self.lca = True  # 1
        self.sparse = None  # 2n/ lg n * log ( 2n/ lg n )
        self._build_cartesian_tree()  # 3n ... but is deleted

    def _build_cartesian_tree(self):
        """builds a cartesian tree of the input array
        performs an Euler Tour and stores the arrays A, depths and R in self"""

        c = CartesianTreeRMQ(self.array)
        c.build_cartesian_tree()  # step 1
        self.E, self.D, self.R = c.euler_tour()  # step 2

    @staticmethod
    def get_block_type(block, blocksize) -> int:
        """ Determines the block type of a block
        If blocksize is too small, i.e 1: there is no need for saving blocks in a lookup table
        Calculates the type of normalized block an array would have
        uses the bitvector w to encode a pattern to save space
        Args:
            block: an array of numbers, as long as they differ evenly by +- c
            blocksize: the size of a an array
        Returns:
            an integer w
        """

        if blocksize == 1:
            return 0

        # assumes that blocksize is at least 2
        w = 0
        k = blocksize - 1
        if block[0] > block[1]:
            w = w | (1 << k)
        else:
            w = w | (1 << (k - 1))

        k -= 2  # skip two positions
        for i in range(2, blocksize):
            if block[i] > block[i - 1]:
                w = w | (1 << k)  # set bit at position k to 1
            k -= 1
        return w

    @staticmethod
    def get_pos(i, j, blocksize):
        """ returns the position i, j in the compressed lookup array
        the col = # arithmetic summation from [blocksize .. blocksize - i + 1]
        # col = int(((blocksize << 1) - i + 1) * i) >> 1
        # row = j - i
        """

        if i == 0:
            return j
        return (int(((blocksize << 1) - i + 1) * i) >> 1) + (j - i)

    def rmq_compressed_lookup(self, blocksize, ltinterval, shift=None, arr=None, lookup=None):
        """ fills the lookup table starting from position shift
        Uses an array of size √n * 2^b_size
        Args:
            blocksize: size of a block
            ltinterval: size of a lookup section for one block
            shift: the starting position of block i's lookup table:
                It can be calculated as shift = b_type * lookup_interval.
                Every b_type, i.e [00, 01, 10, 11] occupies a specific index in the lookup_table
            arr: the array whose rmq structure is being built
            lookup: the array containing the lookup sections of each block
        Returns:
            None
        Raises:
            None
        """

        if lookup is None:
            lookup = np.zeros(ltinterval, dtype='int32')
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

    def populate_sparse_table(self, blockcount, blocksize, lookup, T, save=True):
        """
        Populates a sparse table, with information from the lookup table table
        The for loop finds the minimum value of each block and saves the value in an array A
        Inside the for loop, the array B is also populated. B[i] stores the index of A[i] in self.E
        Args:
            blocksize:     The size of block of the underlying array. It is equal to 1/2 lg n
            blockcount:    The number of blocks the underlying has been divided into:
                        It is usually equal to n / b_size
            T:          An array of integers where T[i] to the points the start index of
                        block i's mini-lookup table in the main lookup table
            lookup:     An array of Integers holding answers to queries of the form argmin[i, j] in a certain block
            save:       If True, the functions add the attributes A, B, sparse to the self
        Returns:
               None
        Raises:
               None
        """

        m = int(log2(blockcount))

        A = np.zeros(blockcount, dtype='int32')
        B = np.zeros(blockcount, dtype='int32')

        sparse = np.full((blockcount, m + 1), inf, dtype='int32')

        for i in range(0, blockcount):
            # computing the min in each block
            minpos = T[i] + self.get_pos(0, blocksize - 1, blocksize)

            index = lookup[minpos]  # when i = 0, j = b_size - 1; [0][b_size - 1] =
            # 0 * b_comb + b_size - 1
            index = i * blocksize + index  # absolute index in the depths array = i * b_size + index

            A[i] = self.D[index]  # changed from E to depths
            B[i] = index

            # in the end, construct the sparse table
        self.construct_sparse_table(A, sparse)
        if save:
            self.sparse = sparse
            self.A = A
            self.B = B

    def construct_restricted_rmq(self, save=True):
        """Constructs the range minimum query for an array with the +- 1 property

        Variables:
            n: the length of the array D, i.e the array storing the depth of nodes
            blocksize: the size of a block, i.e 1/2 lg n
            blockcount: the number of blocks of size blocksize. i.e 2n/ lg n
            ltinterval: the size of a lookup section for one block in the lookup array
            sizelt: the size of the whole lookup array
            lookup: the lookup table, usually denoted as P in papers
            T: an array denoting the starting position of block i's lookup section
            bstart: starting index of block i in self.D
            bend: end position of block i in self.D

         Args:
            save: if true, save (blocksize, blockcount, lookup, T) in self
            else return the tuple

        Returns:
            None or a tuple of (blocksize, blockcount, lookup, T)
        """

        n = len(self.D)
        blocksize = ceil(log2(n)) // 2
        blockcount = ceil(n / blocksize)

        # arithmetic summation for block size of 2, possible blocks [0, 1, 2, 3]
        ltinterval = (blocksize * (blocksize + 1)) >> 1
        size_lt = ltinterval * (1 << blocksize)  # size of lookup table

        lookup = np.full(size_lt, -1, dtype='int32')  # the lookup table [0, 1, 2, 3 ... blockcount]
        T = np.zeros(blockcount, dtype='int32')

        for i in range(0, blockcount):
            bstart = i * blocksize
            bend = min(bstart + blocksize - 1, n - 1)
            if bend - bstart < blocksize - 1:  # last block is shorter
                block = np.append(self.D[bstart: bend + 1],
                                  [self.INFINITY] * (bend - bstart + 1))
                btype = self.get_block_type(block, blocksize)
            else:
                btype = self.get_block_type(self.D[bstart: bend + 1], blocksize)
            tpos = btype * ltinterval  # position of btype in the lookup array
            if lookup[tpos] == -1:
                self.rmq_compressed_lookup(blocksize, ltinterval, tpos, self.D[bstart: bend + 1], lookup)
                T[i] = tpos
            else:
                T[i] = tpos

        self.populate_sparse_table(blockcount, blocksize, lookup, T, save)
        if save:
            self.blocksize = blocksize
            self.blockcount = blockcount
            self.ltinterval = ltinterval
            self.T = T
            self.lookup = lookup
        else:
            return blocksize, blocksize, lookup, T

    def _query_restricted_rmq(self, a, b, lca=True):
        """ The overall strategy here is straight from the paper.  We look up
            the blocks that contain i and j, find their mins, and compare that min with
            of [block i .. block j] using the sparse table

            Most of what's below is offset math, and isn't all that interesting.

        Args:
            a: the start position in the original array
            b: the end position in the original array
            lca: tells whether to return argmin or min in the array E

        Returns:
            if lca == True return min in the array E, which is equal to the min in the original array
                    otherwise, return argmin in the array E, not argmin in the original array
        """

        i, j = self.R[a], self.R[b]
        if i > j:
            i, j = j, i

        x = i // self.blocksize  # block containing i
        y = j // self.blocksize  # block containing y
        u = i % self.blocksize  # starting position of i in block x
        v = j % self.blocksize  # end position of j in block y

        if x == y:  # if i, j are in the same block only use lookup

            ipos = self.T[x] + self.get_pos(u, v, self.blocksize)
            imin = x * self.blocksize + self.lookup[ipos]

            if lca:
                return self.E[imin]
            else:
                return imin
        else:
            # first block min calculation
            fpos = self.T[x] + self.get_pos(u, self.blocksize - 1, self.blocksize)
            fmin = x * self.blocksize + self.lookup[fpos]  # argmin in first block

            # last block min calculation
            lpos = self.T[y] + self.get_pos(0, v, self.blocksize)
            lmin = y * self.blocksize + self.lookup[lpos]   # argmin in last block

            bmin = fmin if self.D[fmin] < self.D[lmin] else lmin

            # super array minimum
            if y - x > 1:  # if there are blocks between x, y
                superpos = self.B[self._query_sparse_table(x + 1, y - 1, self.A, self.sparse)]
                opos = bmin if self.D[bmin] < self.D[superpos] else superpos  # overall argmin
            else:
                opos = bmin

            if lca:
                return self.E[opos]
            else:
                return opos

    def __repr__(self):
        return self.__class__.__qualname__ + '\n(array: ' + str(self.array) + '\n' \
               + 'D:' + str(self.D) + ')'

    def rmq(self, a, b):
        """Return the minimum value or argmin"""
        return self._query_restricted_rmq(a, b)


if __name__ == '__main__':
    test = [17, 33, 1, 35, 78, 6, 30, 22, 13, 0]
    x = min(test[1:7])
    r = RestrictedRMQ(test)
    r.construct_restricted_rmq()
    y = r.rmq(1, 6)
    print(x == y, (x, y))
