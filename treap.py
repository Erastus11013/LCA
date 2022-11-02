from dataclasses import dataclass
from typing import Generic, Iterable, Optional

import numpy as np

from common import T


@dataclass(slots=True, order=True)
class CartesianTreeNode(Generic[T]):
    value: T
    left: Optional["CartesianTreeNode[T]"] = None
    right: Optional["CartesianTreeNode[T]"] = None
    parent: Optional["CartesianTreeNode[T]"] = None


def build_cartesian_tree(array: Iterable[T]) -> CartesianTreeNode:
    array = np.array(array)
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


@dataclass(slots=True, frozen=True)
class EulerTour(Generic[T]):
    labels: list[T]  # labels of the visited nodes in an array
    depths: list[int]  # depths of the visited nodes in the euler tour
    reps: dict[T, int]  # position of the first occurrence of array[i] in the euler tour


def gen_euler_tour(root: Optional[CartesianTreeNode]) -> EulerTour:
    labels, depths, representatives = [], [], {}

    def euler_visit(node: Optional[CartesianTreeNode], depth: int) -> None:
        if node:
            depths.append(depth)
            labels.append(node.value)
            representatives[node.value] = len(labels) - 1

            euler_visit(node.left, depth + 1)
            labels.append(node.value)
            depths.append(depth)
            euler_visit(node.right, depth + 1)
            labels.append(node.value)
            depths.append(depth)

    euler_visit(root, 0)
    return EulerTour(labels, depths, representatives)
