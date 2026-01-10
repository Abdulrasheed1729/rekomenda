from dataclasses import dataclass

import numpy as np


@dataclass
class COOMatrix:
    user_indices: np.ndarray
    user_ids: np.ndarray
    user_counts: np.ndarray | None
    item_indices: np.ndarray
    item_ids: np.ndarray
    item_counts: np.ndarray | None
    ratings: np.ndarray
    num_users: int = 0
    num_items: int = 0

    def __post_init__(self):
        """Ensure num_users and num_items are set"""
        if self.num_users == 0:
            self.num_users = len(self.user_ids)
        if self.num_items == 0:
            self.num_items = len(self.item_ids)

    def get_user_items(self, user_id):
        user_idx = np.where(self.user_ids == user_id)[0]
        if len(user_idx) == 0:
            return np.array([]), np.array([])
        user_idx = user_idx[0]
        mask = self.user_indices == user_idx
        return self.item_ids[self.item_indices[mask]], self.ratings[mask]

    @classmethod
    def from_raw_data(cls, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        # FIX: Use consistent variable naming
        user_ids, user_inverse = np.unique(users, return_inverse=True)
        num_users = len(user_ids)
        user_counts = np.bincount(user_inverse, minlength=num_users)

        item_ids, item_inverse = np.unique(items, return_inverse=True)
        num_items = len(item_ids)
        item_counts = np.bincount(item_inverse, minlength=num_items)

        return cls(
            user_inverse,  # FIX: Use inverse indices, not unique values
            user_ids,
            user_counts,
            item_inverse,  # FIX: Use inverse indices, not unique values
            item_ids,
            item_counts,
            ratings,
            num_users,
            num_items,
        )
