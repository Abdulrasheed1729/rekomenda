import csv
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union

import numpy as np
from numba import njit, prange

from .coo import COOMatrix

# ==============================================================================
# 1. NUMBA-ACCELERATED UPDATE FUNCTIONS
# ==============================================================================


@njit(parallel=True, fastmath=True)
def update_user_biases_numba(
    indptr, item_indices, ratings, item_bias, global_mean, lambda_reg, user_bias
):
    """Numba-accelerated user bias updates (10-20x faster)"""
    num_users = len(indptr) - 1

    for u in prange(num_users):
        start, end = indptr[u], indptr[u + 1]
        if start == end:
            continue

        residual_sum = 0.0
        n_ratings = end - start

        for idx in range(start, end):
            item_idx = item_indices[idx]
            residual = ratings[idx] - global_mean - item_bias[item_idx]
            residual_sum += residual

        user_bias[u] = residual_sum / (n_ratings + lambda_reg)


@njit(parallel=True, fastmath=True)
def update_item_biases_numba(
    indptr, user_indices, ratings, user_bias, global_mean, lambda_reg, item_bias
):
    """Numba-accelerated item bias updates"""
    num_items = len(indptr) - 1

    for i in prange(num_items):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue

        residual_sum = 0.0
        n_ratings = end - start

        for idx in range(start, end):
            user_idx = user_indices[idx]
            residual = ratings[idx] - global_mean - user_bias[user_idx]
            residual_sum += residual

        item_bias[i] = residual_sum / (n_ratings + lambda_reg)


# ==============================================================================
# 2. PARALLEL PROCESSING FOR LATENT FACTORS
# ==============================================================================


def update_user_factors_parallel(
    csr_matrix,
    item_factors,
    item_bias,
    global_mean,
    lambda_reg,
    n_factors,
    user_bias,
    user_factors,
    n_workers=None,
):
    """
    Parallel processing for user factor updates
    Use with ProcessPoolExecutor for CPU-bound tasks
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    def update_user(u):
        start, end = csr_matrix.indptr[u], csr_matrix.indptr[u + 1]
        if start == end:
            return u, user_bias[u], user_factors[u]

        item_indices = csr_matrix.item_indices[start:end]
        ratings = csr_matrix.ratings[start:end]
        n_ratings = end - start

        Q_u = item_factors[item_indices]
        Q_u_aug = np.column_stack([np.ones(n_ratings), Q_u])
        residuals = ratings - global_mean - item_bias[item_indices]

        A = Q_u_aug.T @ Q_u_aug
        A += lambda_reg * np.eye(n_factors + 1)
        b = Q_u_aug.T @ residuals

        x = np.linalg.solve(A, b)
        return u, x[0], x[1:]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(update_user, range(csr_matrix.num_users)))

    for u, bias, factors in results:
        user_bias[u] = bias
        user_factors[u] = factors


# ==============================================================================
# 3. BATCH PROCESSING FOR PREDICTIONS
# ==============================================================================


def predict_batch(
    user_indices,
    item_indices,
    global_mean,
    user_bias,
    item_bias,
    user_factors,
    item_factors,
    batch_size=100000,
):
    """
    Batch processing for predictions to avoid memory issues
    Useful when predicting on very large test sets
    """
    n_samples = len(user_indices)
    predictions = np.zeros(n_samples, dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_users = user_indices[start:end]
        batch_items = item_indices[start:end]

        # Base predictions
        batch_pred = global_mean + user_bias[batch_users] + item_bias[batch_items]

        # Latent factors
        user_factors_batch = user_factors[batch_users]
        item_factors_batch = item_factors[batch_items]
        batch_pred += np.sum(user_factors_batch * item_factors_batch, axis=1)

        predictions[start:end] = batch_pred

    return predictions


def load_data(
    filepath, split_ratio: Optional[int] = None, csv_delimiter: str = ","
) -> Union[Tuple[COOMatrix, COOMatrix], COOMatrix]:
    users = list()
    movies = list()
    ratings = list()

    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter=csv_delimiter)
        next(reader)
        for row in reader:
            users.append(row[0])
            movies.append(row[1])
            ratings.append(float(row[2]))

        users = np.array(users)
        movies = np.array(movies)
        ratings = np.array(ratings)
        if split_ratio is not None:
            mask = np.random.rand(len(ratings)) > split_ratio

            test_mask = mask
            train_mask = ~mask

            train_coo = COOMatrix.from_raw_data(
                users[train_mask], movies[train_mask], ratings[train_mask]
            )
            test_coo = COOMatrix.from_raw_data(
                users[test_mask], movies[test_mask], ratings[test_mask]
            )

            return train_coo, test_coo

        coo_mat = COOMatrix.from_raw_data(users, movies, ratings)

    return coo_mat
