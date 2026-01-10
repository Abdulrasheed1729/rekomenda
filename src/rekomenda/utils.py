import csv
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union

import numpy as np
from numba import njit, prange

from .coo import COOMatrix
from .csc import CSCMatrix
from .csr import CSRMatrix

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


def load_csv(filepath: str, delimiter: str = ",", skip_header: bool = True):
    """
    Load CSV file as numpy array

    Args:
        filepath: Path to CSV file
        delimiter: Column delimiter
        skip_header: Skip first row

    Returns:
        data: numpy array of data
        header: list of column names (if skip_header=True)
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    header = None
    start_idx = 0

    if skip_header:
        header = lines[0].strip().split(delimiter)
        start_idx = 1

    data = []
    for line in lines[start_idx:]:
        row = line.strip().split(delimiter)
        data.append(row)

    return data, header


def load_csv_chunked(
    filepath: str,
    delimiter: str = ",",
    skip_header: bool = True,
    chunk_size: int = 1000000,
):
    """
    Generator that yields chunks of CSV data
    Memory efficient for large files

    Args:
        filepath: Path to CSV file
        delimiter: Column delimiter
        skip_header: Skip first row
        chunk_size: Number of rows per chunk

    Yields:
        chunk: numpy array of chunk_size rows
    """
    with open(filepath, "r") as f:
        if skip_header:
            header = f.readline().strip().split(delimiter)
        else:
            header = None

        chunk = []
        for line in f:
            row = line.strip().split(delimiter)
            chunk.append(row)

            if len(chunk) >= chunk_size:
                yield np.array(chunk), header
                chunk = []

        # Yield remaining rows
        if chunk:
            yield np.array(chunk), header


def load_movielens_data(
    ratings_path: str,
    movies_path: str,
    test_size: float = 0.2,
    random_seed: int = 42,
    chunk_size: int = 1000000,
):
    """
    Load MovieLens dataset using chunked processing (memory efficient)

    Args:
        ratings_path: Path to ratings.csv (userId, movieId, rating, timestamp)
        movies_path: Path to movies.csv (movieId, title, genres)
        test_size: Fraction of data for testing
        random_seed: Random seed for reproducibility
        chunk_size: Process ratings in chunks of this size

    Returns:
        train_csr: CSR matrix for training
        train_csc: CSC matrix for training
        test_data: (test_users, test_items, test_ratings) as arrays
        item_id_to_name: dict mapping item indices to movie names
        movie_id_to_idx: dict mapping original movie IDs to matrix indices
    """

    print("Loading MovieLens data with chunked processing...")

    # Step 1: Load movies (small file, can load all at once)
    print("Loading movie metadata...")
    movies_data, _ = load_csv(movies_path)
    movie_ids_movies = np.array([int(x[0]) for x in movies_data])
    movie_titles = [x[1] for x in movies_data]

    movie_id_to_title = dict(zip(movie_ids_movies, movie_titles))
    print(f"Loaded {len(movie_id_to_title)} movies")
    del movies_data  # Free memory

    # Step 2: First pass - collect unique users and items, count total ratings
    print("\nFirst pass: Collecting unique users and items...")
    unique_users = set()
    unique_items = set()
    total_ratings = 0

    for chunk, header in load_csv_chunked(ratings_path, chunk_size=chunk_size):
        users_chunk = chunk[:, 0].astype(int)
        items_chunk = chunk[:, 1].astype(int)

        unique_users.update(users_chunk)
        unique_items.update(items_chunk)
        total_ratings += len(chunk)

        if total_ratings % 5000000 == 0:
            print(f"  Processed {total_ratings:,} ratings...")

    print(f"Found {len(unique_users):,} unique users")
    print(f"Found {len(unique_items):,} unique items")
    print(f"Total ratings: {total_ratings:,}")

    # Step 3: Create mappings
    print("\nCreating ID mappings...")
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(unique_users))}
    movie_id_to_idx = {
        movie_id: idx for idx, movie_id in enumerate(sorted(unique_items))
    }
    idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}

    item_id_to_name = {
        idx: movie_id_to_title.get(movie_id, f"Movie {movie_id}")
        for idx, movie_id in idx_to_movie_id.items()
    }

    del unique_users, unique_items  # Free memory

    # Step 4: Second pass - load ratings, map IDs, and split
    print("\nSecond pass: Loading ratings and creating train/test split...")
    np.random.seed(random_seed)

    train_users_list = []
    train_items_list = []
    train_ratings_list = []
    test_users_list = []
    test_items_list = []
    test_ratings_list = []

    processed = 0
    for chunk, header in load_csv_chunked(ratings_path, chunk_size=chunk_size):
        users_chunk = chunk[:, 0].astype(int)
        items_chunk = chunk[:, 1].astype(int)
        ratings_chunk = chunk[:, 2].astype(np.float32)

        # Map to matrix indices
        user_indices = np.array([user_id_to_idx[uid] for uid in users_chunk])
        item_indices = np.array([movie_id_to_idx[mid] for mid in items_chunk])

        # Random train/test split for this chunk
        n_chunk = len(chunk)
        is_test = np.random.random(n_chunk) < test_size

        # Split into train and test
        train_mask = ~is_test
        train_users_list.append(user_indices[train_mask])
        train_items_list.append(item_indices[train_mask])
        train_ratings_list.append(ratings_chunk[train_mask])

        test_users_list.append(user_indices[is_test])
        test_items_list.append(item_indices[is_test])
        test_ratings_list.append(ratings_chunk[is_test])

        processed += len(chunk)
        if processed % 5000000 == 0:
            print(f"  Processed {processed:,} ratings...")

    # Step 5: Concatenate all chunks
    print("\nConcatenating chunks...")
    train_users = np.concatenate(train_users_list)
    train_items = np.concatenate(train_items_list)
    train_ratings = np.concatenate(train_ratings_list)

    test_users = np.concatenate(test_users_list)
    test_items = np.concatenate(test_items_list)
    test_ratings = np.concatenate(test_ratings_list)

    # Free memory
    del train_users_list, train_items_list, train_ratings_list
    del test_users_list, test_items_list, test_ratings_list

    print(
        f"Train: {len(train_ratings):,} ratings ({len(train_ratings) / total_ratings * 100:.1f}%)"
    )
    print(
        f"Test: {len(test_ratings):,} ratings ({len(test_ratings) / total_ratings * 100:.1f}%)"
    )

    # Step 6: Create sparse matrices for training data
    print("\nCreating training sparse matrices...")
    train_csr = CSRMatrix.from_raw_data(train_users, train_items, train_ratings)
    train_csc = CSCMatrix.from_raw_data(train_users, train_items, train_ratings)

    print(f"Matrix: {train_csr.num_users} users × {train_csr.num_items} items")
    print(
        f"Sparsity: {len(train_ratings) / (train_csr.num_users * train_csr.num_items) * 100:.3f}%"
    )

    # Free training arrays (now in sparse matrices)
    del train_users, train_items, train_ratings

    # Keep test data as arrays (small)
    test_data = (test_users, test_items, test_ratings)

    print("\n✓ Data loaded successfully with chunked processing")
    print(
        f"✓ Memory efficient: Processed {total_ratings:,} ratings in chunks of {chunk_size:,}"
    )

    return train_csr, train_csc, test_data, item_id_to_name, movie_id_to_idx


def find_movie_by_title(search_term: str, item_id_to_name: dict, top_n=5):
    """Search for movies by title"""
    matches = []
    search_lower = search_term.lower()

    for item_id, name in item_id_to_name.items():
        if search_lower in name.lower():
            matches.append((item_id, name))

    return matches[:top_n]


def compute_pca(embeddings: np.ndarray, n_components: int = 2):
    """
    Compute PCA using eigendecomposition

    Args:
        embeddings: (n_samples, n_features) matrix
        n_components: Number of principal components

    Returns:
        coords_2d: (n_samples, n_components) projected coordinates
        explained_variance_ratio: Fraction of variance explained by each component
    """
    # Center the data
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean

    # Compute covariance matrix
    cov_matrix = np.cov(centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    top_eigenvectors = eigenvectors[:, :n_components]

    # Project data
    coords = centered @ top_eigenvectors

    # Compute explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[:n_components] / total_variance

    return coords, explained_variance_ratio
