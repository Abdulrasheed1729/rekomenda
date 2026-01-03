from dataclasses import dataclass

import numpy as np
from numba.experimental import jitclass

from .coo import COOMatrix


@dataclass
@jitclass
class CSCMatrix:
    """Compressed Sparse Column matrix for efficient column operations"""

    indptr: np.ndarray  # Column pointers (length num_items + 1)
    user_indices: np.ndarray  # Row indices
    user_ids: np.ndarray  # Original user IDs
    ratings: np.ndarray  # Rating values
    item_ids: np.ndarray  # Original item IDs
    user_counts: np.ndarray | None
    item_counts: np.ndarray | None
    num_users: int = 0
    num_items: int = 0

    @classmethod
    def from_coo(cls, coo: COOMatrix):
        """Convert COO to CSC format"""
        # Step 1: Get dimensions
        num_cols = coo.num_items  # Number of unique items
        # num_entries = len(coo.item_indices)  # Number of ratings

        # Step 2: Sort by column index (item) then row index (user)
        sort_order = np.lexsort((coo.user_indices, coo.item_indices))
        col_indices_sorted = coo.item_indices[sort_order]
        row_indices_sorted = coo.user_indices[sort_order]
        ratings_sorted = coo.ratings[sort_order]

        # Step 3: Count entries per column
        item_counts = np.bincount(col_indices_sorted, minlength=num_cols)

        # Step 4: Build indptr using cumulative sum
        indptr = np.zeros(num_cols + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(item_counts)

        return cls(
            indptr=indptr,
            user_indices=row_indices_sorted,
            user_ids=coo.user_ids,
            ratings=ratings_sorted,
            item_ids=coo.item_ids,
            user_counts=coo.user_counts,
            item_counts=coo.item_counts,
            num_users=coo.num_users,
            num_items=coo.num_items,
        )

    @classmethod
    def from_raw_data(cls, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        """Create CSC directly from raw data"""
        coo = COOMatrix.from_raw_data(users, items, ratings)
        return cls.from_coo(coo)

    def get_item_users(self, item_id):
        """Get all users and ratings for a specific item ID"""
        item_idx = np.where(self.item_ids == item_id)[0]
        if len(item_idx) == 0:
            return np.array([]), np.array([])
        item_idx = item_idx[0]
        start, end = self.indptr[item_idx], self.indptr[item_idx + 1]
        return self.user_ids[self.user_indices[start:end]], self.ratings[start:end]
