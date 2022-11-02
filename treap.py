from dataclasses import dataclass
from typing import Generic, Optional

import numpy as np

from common import T


@dataclass(slots=True, order=True)
class CartesianTreeNode(Generic[T]):
    value: T
    left: Optional["CartesianTreeNode[T]"] = None
    right: Optional["CartesianTreeNode[T]"] = None
    parent: Optional["CartesianTreeNode[T]"] = None


class CartesianTree:
    def __init__(self, a):
        self.array = np.array(a)
        self.root = self.__construct_cartesian_tree(self.array)

    @staticmethod
    def __construct_cartesian_tree(array):
        if array.size == 0:
            raise ValueError("cannot build cartesian tree from empty array")

        root = CartesianTreeNode(array[0])
        last = root

        for node in map(CartesianTreeNode, array[1:]):
            while last > node and last is not root:
                last = last.parent

            if last > node:
                root.parent = node
                node.left = root
                root = node
            elif last.right is None:
                last.right = node
                node.parent = last
                node.left = None
            else:
                last.right.parent = node
                node.left = last.right
                last.right = node
                node.parent = last

            last = node

        return root


class CartesianTreeRMQ(CartesianTree):
    def __init__(self, a):
        super().__init__(a)

    def euler_tour(self):
        # labels of the visited nodes in an array
        labels = []

        # depths of the visited nodes in the euler tour
        depths = []

        # position of the first occurrence of self.array[i] in the euler tour
        representatives = []

        def euler_visit(node: Optional[CartesianTreeNode], depth):
            if node:
                depths.append(depth)
                labels.append(node.value)
                representatives.append(len(labels) - 1)

                euler_visit(node.left, depth + 1)
                labels.append(node.value)
                depths.append(depth)
                euler_visit(node.right, depth + 1)
                labels.append(node.value)
                depths.append(depth)

        euler_visit(self.root, 0)
        # print('->'.join(map(str, E)))
        return np.array(labels), np.array(depths), np.array(representatives)
