import numpy as np
from rmq import RMQ


class RMQCatalan(RMQ):
    def __init__(self, a):
        super(RMQCatalan, self).__init__(a)
        self.C = self.compute_ballot_numbers(len(a) + 1)
    
    @staticmethod
    def get_block_type(block, s):
        """Args:
            block: an array of length s
            s: the length of block

            Returns:
                The block type of s"""

        rp = np.full(s + 1, 0, dtype='int64')
        rp[0] = -RMQ.INFINITY  # rp.push(-∞)  # correct
        q = s  # size of the array
        b = 0
        for i in range(1, s + 1):
            while rp[q + i - s - 1] > block[i - 1]:
                b = b + C[q][s - i]
                q = q - 1
            rp[q + i - s] = block[i - 1]
        return int(b)


    def get_block_type(self, block, s):
        """Args:
            block: an array of length s
            s: the length of block

            Returns:
            an integer representing the block type of s"""

        rp = np.full(s + 1, 0, dtype='int64')
        rp[0] = self.INFINITY  # rp.push(-∞)  # correct
        q = s  # size of the array
        b = 0
        for i in range(1, s + 1):
            while rp[q + i - s - 1] > block[i - 1]:
                b = b + self.C[q][s - i]
                q = q - 1
            rp[q + i - s] = block[i - 1]
        return int(b)
