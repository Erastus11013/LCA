import random
from collections import deque
from dataclasses import dataclass
from math import inf
from typing import Generic, Iterable, Optional

import numpy as np

from shared import T


@dataclass(slots=True, order=True)
class CartesianTreeNode(Generic[T]):
    value: T
    value_index: int
    left: Optional["CartesianTreeNode[T]"] = None
    right: Optional["CartesianTreeNode[T]"] = None
    parent: Optional["CartesianTreeNode[T]"] = None

    def inorder(self):
        stack = []
        current = self
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                current = stack.pop()
                yield current
                current = current.right


def build_cartesian_tree(array: Iterable[T]) -> CartesianTreeNode:
    array = np.array(array)
    if array.size == 0:
        raise ValueError("cannot build cartesian tree from empty array")

    root = CartesianTreeNode(array[0], 0)
    last = root

    for value_index, value in enumerate(array[1:], start=1):
        node = CartesianTreeNode(value, value_index)
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


@dataclass(slots=True, order=True)
class BinaryTreeNode(Generic[T]):
    value: T
    left: Optional["BinaryTreeNode[T]"] = None
    right: Optional["BinaryTreeNode[T]"] = None


def encode_succinct(
    node: Optional[BinaryTreeNode], encoding: deque[int], values: deque[T]
) -> None:
    if node is None:
        encoding.append(0)
    else:
        encoding.append(1)
        values.append(node.value)
        encode_succinct(node.left, encoding, values)
        encode_succinct(node.right, encoding, values)


def decode_succinct(encoding: deque[T], values: deque[T]) -> Optional[BinaryTreeNode]:
    if encoding and encoding.popleft() == 1:
        return BinaryTreeNode(
            values.popleft(),
            decode_succinct(encoding, values),
            decode_succinct(encoding, values),
        )
    else:
        return None


def gen_pseudo_random_binary_tree(
    n_nodes, r: tuple[float, float]
) -> Optional[BinaryTreeNode]:
    if n_nodes == 0:
        return None
    else:
        split = random.randint(1, n_nodes)
        low, high = r
        value = (low + high) / 2
        return BinaryTreeNode(
            value,
            gen_pseudo_random_binary_tree(split - 1, (low, value)),
            gen_pseudo_random_binary_tree(n_nodes - split, (value, high)),
        )


def validate_bst(root: Optional[BinaryTreeNode]) -> None:
    def _validate_bst(node: Optional[BinaryTreeNode], low: float, high: float) -> None:
        if node is not None:
            if not (low < node.value < high):
                raise ValueError()
            _validate_bst(node.left, low, node.value)
            _validate_bst(node.right, node.value, high)

    _validate_bst(root, -inf, inf)


def get_lca_impl(
    root: Optional[BinaryTreeNode | CartesianTreeNode],
    lca: Optional[BinaryTreeNode | CartesianTreeNode],
    x: T,
    y: T,
) -> tuple[bool, BinaryTreeNode | CartesianTreeNode]:
    # base case 1: return false if the tree is empty
    if root is None:
        return False, lca

    # base case 2: return true if either `x` or `y` is found
    # with lca set to the current node
    if root.value == x or root.value == y:
        return True, root

    # recursively check if `x` or `y` exists in the left subtree
    left, lca = get_lca_impl(root.left, lca, x, y)

    # recursively check if `x` or `y` exists in the right subtree
    right, lca = get_lca_impl(root.right, lca, x, y)

    # if `x` is found in one subtree and `y` is found in the other subtree,
    # update lca to the current node
    if left and right:
        lca = root

    # return true if `x` or `y` is found in either left or right subtree
    return (left or right), lca


def get_lca(
    root: BinaryTreeNode | CartesianTreeNode, x: T, y: T
) -> BinaryTreeNode | CartesianTreeNode:
    return get_lca_impl(root, None, x, y)[1]


if __name__ == "__main__":
    r1 = gen_pseudo_random_binary_tree(100, (0, 100))
    validate_bst(r1)
    print(get_lca(r1, 6.25, 43.5638427734375))
    encoding1 = deque()
    values1 = deque()
    encode_succinct(r1, encoding1, values1)
    print(len(set(values1)), len(values1))
    print(encoding1, values1)
    root1 = decode_succinct(encoding1, values1)
