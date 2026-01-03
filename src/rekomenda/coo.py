from dataclasses import dataclass

import numpy as np
from numba.experimental import jitclass


@dataclass
@jitclass
class COOMatrix:
    user_indices: np.ndarray  # row indices
    user_ids: np.ndarray
    user_counts: np.ndarray | None

    item_indices: np.ndarray  # column indices
    item_ids: np.ndarray
    item_counts: np.ndarray | None

    ratings: np.ndarray
    num_users: int = 0
    num_items: int = 0

    def get_user_items(self, user_id):
        """Get all items and ratings for a specific user ID"""
        user_idx = np.where(self.user_ids == user_id)[0]
        if len(user_idx) == 0:
            return np.array([]), np.array([])
        user_idx = user_idx[0]
        mask = self.user_indices == user_idx
        return self.item_ids[self.item_indices[mask]], self.ratings[mask]

    @classmethod
    def from_raw_data(cls, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        user_ids, user_indices = np.unique(users, return_inverse=True)
        num_users = len(user_ids)
        user_counts = np.bincount(user_indices, minlength=num_users)

        item_ids, item_indices = np.unique(items, return_inverse=True)
        num_items = len(item_ids)
        item_counts = np.bincount(item_indices, minlength=num_items)

        # sort_idx = np.lexsort((item_indices, user_indices))
        # user_indices = user_indices[sort_idx]
        # item_indices = item_indices[sort_idx]
        # ratings = ratings[sort_idx]

        coo = cls(
            user_indices,
            user_ids,
            user_counts,
            item_indices,
            item_ids,
            item_counts,
            ratings,
            num_users,
            num_items,
        )
        return coo
