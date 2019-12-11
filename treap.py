import numpy as np
from pprint import pprint


class CartesianTree:
    def __init__(self, a):
        self.array = np.array(a)
        self.root = self.right = self.left = self.parent = None

    def build_cartesian_tree(self):
        """I use arrays for left, right, parent because array indexing is faster than attribute access"""
        if len(self.array) < 1:
            raise ValueError("cannot build cartesian tree from empty array")
        n = len(self.array)
        self.right = np.zeros(n, dtype='int32')
        self.left = np.zeros(n, dtype='int32')
        self.parent = np.zeros(n, dtype='int32')

        self.root = 0
        self.left[0] = -1
        self.right[0] = -1
        self.parent[0] = -1

        for i in range(1, n):
            last = i - 1
            self.right[i] = -1

            while self.array[last] > self.array[i] and last != self.root:
                last = self.parent[last]

            if self.array[last] > self.array[i]:
                self.parent[self.root] = i
                self.left[i] = self.root
                self.root = i

            elif self.right[last] == -1:
                self.right[last] = i
                self.parent[i] = last
                self.left[i] = -1
            else:
                self.parent[self.right[last]] = i
                self.left[i] = self.right[last]
                self.right[last] = i
                self.parent[i] = last

    def in_order(self, start=None):
        """
        In order traversal of a tree can be described in three simple steps:
            1. traverse the left subtree
            2. visit root
            3. traverse the right subtree.

        Args:
            start: the current node.

        Returns:
            an array of the nodes.
        """

        def helper(current, array):
            if self.left[current] == -1 and self.right[current] == -1:
                array.append(self.array[current])
            else:
                if self.left[current] != -1:
                    helper(self.left[current], array)
                array.append(self.array[current])
                if self.right[current] != -1:
                    helper(self.right[current], array)

        if start is None:
            start = self.root
        args = []
        helper(start, args)
        return np.array(args)

    def post_order(self, start=None):

        # noinspection DuplicatedCode
        def helper(current, array):
            if self.left[current] == -1 and self.right[current] == -1:
                array.append(self.array[current])
            else:
                if self.left[current] != -1:
                    helper(self.left[current], array)
                if self.right[current] != -1:
                    helper(self.right[current], array)
                array.append(self.array[current])

        if start is None:
            start = self.root
        args = []
        helper(start, args)
        return np.array(args)

    def pre_order(self, start=None):

        # noinspection DuplicatedCode
        def helper(current, array):
            if self.left[current] == -1 and self.right[current] == -1:
                array.append(self.array[current])
            else:
                array.append(self.array[current])
                if self.left[current] != -1:
                    helper(self.left[current], array)
                if self.right[current] != -1:
                    helper(self.right[current], array)

        if start is None:
            start = self.root
        args = []
        helper(start, args)
        return np.array(args)

    def euler_tour(self, source=None):
        """Works for DAGS"""
        args = []

        def euler_visit(current, order):
            order.append(self.array[current])
            if self.left[current] == -1 and self.right[current] == -1:
                return
            else:
                if self.left[current] != -1:
                    euler_visit(self.left[current], order)
                    order.append(self.array[current])
                if self.right[current] != -1:
                    euler_visit(self.right[current], order)
                    order.append(self.array[current])

        if source is None:
            source = self.root
        euler_visit(source, args)
        print('->'.join(map(str, args)))
        return np.array(args)


class CartesianTreeRMQ(CartesianTree):
    def __init__(self, a):
        super().__init__(a)

    def euler_tour(self, source=None):
        n = len(self.array)
        E = []
        D = []
        R = np.full(n, -1, dtype='int32')
        if self.root is None:
            raise ValueError("Build cartesian tree first")

        def euler_visit(current, labels, depths, rep, curr_depth, index):
            depths.append(curr_depth)
            labels.append(self.array[current])
            if rep[current] == -1:
                rep[current] = index
            if not (self.left[current] == -1 and self.right[current] == -1):
                if self.left[current] != -1:
                    euler_visit(self.left[current], labels, depths, rep, curr_depth + 1, index + 1)
                    labels.append(self.array[current])
                    depths.append(curr_depth)
                    index += 1
                if self.right[current] != -1:
                    euler_visit(self.right[current], labels, depths, rep, curr_depth + 1, index + 1)
                    labels.append(self.array[current])
                    depths.append(curr_depth)
                    index += 1

        if source is None:
            source = self.root
        euler_visit(source, E, D, R, 0, 0)
        # print('->'.join(map(str, E)))
        return np.array(E), np.array(D), np.array(R)


if __name__ == '__main__':
    from datetime import datetime
    from random import randint

    test = [9, 3, 7, 1, 8, 12, 10, 20, 15, 18, 5]
    # test = [randint(0, 500) for _ in range(3000)]
    tic = datetime.now()
    c = CartesianTree(test)
    c.build_cartesian_tree()
    toc = datetime.now()
    print("cartesian tree built in", (toc - tic).total_seconds(), 'seconds')

    cr = CartesianTreeRMQ(test)
    cr.build_cartesian_tree()
    e, d, r = cr.euler_tour()
    pprint(cr.array)
    pprint(e)
    pprint(d)
    pprint(r)

    # assert (len(c.euler_tour()) == len(c.array) * 2 - 1)
