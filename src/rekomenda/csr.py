from dataclasses import dataclass

import numpy as np

from .coo import COOMatrix


@dataclass
class CSRMatrix:
    """Compressed Sparse Row matrix for efficient row operations"""

    indptr: np.ndarray  # Row pointers (length num_users + 1)
    item_indices: np.ndarray  # Column indices
    item_ids: np.ndarray  # Original item IDs
    ratings: np.ndarray  # Rating values
    user_ids: np.ndarray  # Original user IDs
    user_counts: np.ndarray | None
    item_counts: np.ndarray | None
    num_users: int = 0
    num_items: int = 0

    @classmethod
    def from_coo(cls, coo: COOMatrix):
        """Convert COO to CSR format"""
        # Step 1: Get dimensions
        num_rows = coo.num_users  # Number of unique users
        # num_entries = len(coo.user_indices)  # Number of ratings

        # Step 2: Sort by row index (user) then column index (item)
        sort_order = np.lexsort((coo.item_indices, coo.user_indices))
        user_indices_sorted = coo.user_indices[sort_order]
        item_indices_sorted = coo.item_indices[sort_order]
        ratings_sorted = coo.ratings[sort_order]

        # Step 3: Count entries per row
        user_counts = np.bincount(user_indices_sorted, minlength=num_rows)

        # Step 4: Build indptr using cumulative sum
        indptr = np.zeros(num_rows + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(user_counts)

        return cls(
            indptr=indptr,
            item_indices=item_indices_sorted,
            item_ids=coo.item_ids,
            ratings=ratings_sorted,
            user_ids=coo.user_ids,
            user_counts=coo.user_counts,
            item_counts=coo.item_counts,
            num_users=coo.num_users,
            num_items=coo.num_items,
        )

    @classmethod
    def from_raw_data(cls, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        """Create CSR directly from raw data"""
        coo = COOMatrix.from_raw_data(users, items, ratings)
        return cls.from_coo(coo)

    def get_user_items(self, user_id):
        """Get all items and ratings for a specific user ID"""
        user_idx = np.where(self.user_ids == user_id)[0]
        if len(user_idx) == 0:
            return np.array([]), np.array([])
        user_idx = user_idx[0]
        start, end = self.indptr[user_idx], self.indptr[user_idx + 1]
        return self.item_ids[self.item_indices[start:end]], self.ratings[start:end]
