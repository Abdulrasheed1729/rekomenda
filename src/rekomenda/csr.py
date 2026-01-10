from dataclasses import dataclass

import numpy as np

from .coo import COOMatrix


@dataclass
class CSRMatrix:
    indptr: np.ndarray
    item_indices: np.ndarray
    item_ids: np.ndarray
    ratings: np.ndarray
    user_ids: np.ndarray
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
        num_rows = coo.num_users
        sort_order = np.lexsort((coo.item_indices, coo.user_indices))
        user_indices_sorted = coo.user_indices[sort_order]
        item_indices_sorted = coo.item_indices[sort_order]
        ratings_sorted = coo.ratings[sort_order]

        # FIX: Use pre-computed counts or recompute
        user_counts = np.bincount(user_indices_sorted, minlength=num_rows)
        indptr = np.zeros(num_rows + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(user_counts)

        return cls(
            indptr,
            item_indices_sorted,
            coo.item_ids,
            ratings_sorted,
            coo.user_ids,
            user_counts,
            coo.item_counts,
            coo.num_users,
            coo.num_items,
        )

    @classmethod
    def from_raw_data(cls, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        coo = COOMatrix.from_raw_data(users, items, ratings)
        return cls.from_coo(coo)

    def get_user_items(self, user_id):
        user_idx = np.where(self.user_ids == user_id)[0]
        if len(user_idx) == 0:
            return np.array([]), np.array([])
        user_idx = user_idx[0]
        start, end = self.indptr[user_idx], self.indptr[user_idx + 1]
        return self.item_ids[self.item_indices[start:end]], self.ratings[start:end]
