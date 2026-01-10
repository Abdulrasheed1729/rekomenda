from dataclasses import dataclass

import numpy as np

from .coo import COOMatrix


@dataclass
class CSCMatrix:
    indptr: np.ndarray
    user_indices: np.ndarray
    user_ids: np.ndarray
    ratings: np.ndarray
    item_ids: np.ndarray
    user_counts: np.ndarray | None
    item_counts: np.ndarray | None
    num_users: int = 0
    num_items: int = 0

    def __post_init__(self):
        """Ensure num_users and num_items are set"""
        if self.num_users == 0:
            self.num_users = len(self.user_ids)
        if self.num_items == 0:
            self.num_items = len(self.item_ids)

    @classmethod
    def from_coo(cls, coo: COOMatrix):
        num_cols = coo.num_items
        sort_order = np.lexsort((coo.user_indices, coo.item_indices))
        col_indices_sorted = coo.item_indices[sort_order]
        row_indices_sorted = coo.user_indices[sort_order]
        ratings_sorted = coo.ratings[sort_order]

        item_counts = np.bincount(col_indices_sorted, minlength=num_cols)
        indptr = np.zeros(num_cols + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(item_counts)

        return cls(
            indptr,
            row_indices_sorted,
            coo.user_ids,
            ratings_sorted,
            coo.item_ids,
            coo.user_counts,
            item_counts,
            coo.num_users,
            coo.num_items,
        )

    @classmethod
    def from_raw_data(cls, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        coo = COOMatrix.from_raw_data(users, items, ratings)
        return cls.from_coo(coo)

    def get_item_users(self, item_id):
        item_idx = np.where(self.item_ids == item_id)[0]
        if len(item_idx) == 0:
            return np.array([]), np.array([])
        item_idx = item_idx[0]
        start, end = self.indptr[item_idx], self.indptr[item_idx + 1]
        return self.user_ids[self.user_indices[start:end]], self.ratings[start:end]
