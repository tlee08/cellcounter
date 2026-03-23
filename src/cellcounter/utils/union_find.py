from collections import defaultdict

import numpy as np


class UnionFind:
    """Union Find.

    Used for matching spatial relationships across
    chunked array foreground bodies.
    """

    parent: dict[int, int]
    rank: dict[int, int]
    sort_idx: np.array | None = None
    sorted_keys: np.array | None = None
    sorted_sizes: np.array | None = None

    def __init__(self) -> None:
        """Union Find."""
        self.parent = {}
        self.rank = {}
        self.sort_idx = None
        self.sorted_keys = None
        self.sorted_sizes = None

    def find(self, x: int) -> int:
        """Find method in union find."""
        while self.parent.get(x, x) != x:
            # Doing double-hop as well
            self.parent[x] = self.parent.get(
                self.parent.get(x, x), self.parent.get(x, x)
            )
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Union method in union find."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        rx, ry = self.rank.get(px, 0), self.rank.get(py, 0)
        if rx < ry:
            px, py = py, px
        self.parent[py] = px
        if rx == ry:
            self.rank[px] = rx + 1

    def build_lookup_table(self, labels: np.ndarray, values: np.ndarray) -> None:
        """Build lookup table.

        We sum the values and store the given sorted labels and summed values.
        """
        # Aggregate by component root
        component_sizes: dict[int, int] = defaultdict(int)
        for label, value in zip(labels, values, strict=True):
            component_sizes[self.find(label)] += value
        component_values = np.array(
            [component_sizes[self.find(lbl)] for lbl in labels], dtype=np.int64
        )
        # Build lookup table
        sort_idx = np.argsort(labels)
        self.sorted_keys = labels[sort_idx]
        self.sorted_sizes = component_values[sort_idx]
