#!/usr/bin/env python3
"""
Quickstart example for Rekomenda

This script reproduces the README quickstart: it attempts to load MovieLens-style
CSV files (ratings, movies). If the CSV files are not provided or not found, the
script falls back to a small synthetic dataset so you can test the training
pipeline without external data.

Usage:
    python rekomenda/examples/quickstart.py --ratings data/ratings.csv --movies data/movies.csv

Or run without arguments to use synthetic data:
    python rekomenda/examples/quickstart.py
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

# Import the library modules. Ensure the package is discoverable via PYTHONPATH
# or installed in editable mode (pip install -e .)
try:
    from rekomenda.als import OptimizedLatentFactorALS
    from rekomenda.csc import CSCMatrix
    from rekomenda.csr import CSRMatrix
    from rekomenda.utils import evaluate_model, load_movielens_data
except Exception as exc:  # pragma: no cover - helpful error for end users
    raise RuntimeError(
        "Failed to import rekomenda package. Make sure you installed the project "
        "in editable mode (pip install -e .) or added the repository root to PYTHONPATH."
    ) from exc


def train_on_movielens(
    ratings_path: str, movies_path: str, args: argparse.Namespace
) -> None:
    """Load MovieLens-format CSVs (chunked) and train/evaluate the model."""
    print("Loading MovieLens data (may take some time for large files)...")
    train_csr, train_csc, test_data, item_id_to_name, movie_id_to_idx = (
        load_movielens_data(
            ratings_path,
            movies_path,
            test_size=args.test_size,
            random_seed=args.seed,
            chunk_size=args.chunk_size,
        )
    )

    test_users, test_items, test_ratings = test_data

    model = OptimizedLatentFactorALS(
        n_factors=args.n_factors,
        lambda_reg=args.lambda_reg,
        use_parallel=args.use_parallel,
        n_workers=args.n_workers,
        dtype=np.float32,
    )

    print("Training model...")
    t0 = time.time()
    model.fit(
        train_csr,
        train_csc,
        n_iterations=args.n_iterations,
        verbose=True,
        compute_metrics_every=args.compute_metrics_every,
    )
    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f} seconds")

    print("Evaluating model on test set...")
    metrics = evaluate_model(model, test_users, test_items, test_ratings)
    print("Evaluation metrics:", metrics)


def train_on_synthetic(args: argparse.Namespace) -> None:
    """Create a small synthetic dataset and train/evaluate the model."""
    print("CSV files not provided or not found — using a synthetic dataset")

    rng = np.random.RandomState(args.seed)
    num_users = args.synthetic_users
    num_items = args.synthetic_items
    n_ratings = args.synthetic_ratings

    # Randomly generate user/item indices and ratings in [1, 5]
    users_all = rng.randint(0, num_users, size=n_ratings)
    items_all = rng.randint(0, num_items, size=n_ratings)
    ratings_all = rng.randint(1, 6, size=n_ratings).astype(np.float32)

    # Random train/test split
    is_test = rng.rand(n_ratings) < args.test_size

    train_users = users_all[~is_test]
    train_items = items_all[~is_test]
    train_ratings = ratings_all[~is_test]

    test_users = users_all[is_test]
    test_items = items_all[is_test]
    test_ratings = ratings_all[is_test]

    # Build sparse matrices for training
    train_csr = CSRMatrix.from_raw_data(train_users, train_items, train_ratings)
    train_csc = CSCMatrix.from_raw_data(train_users, train_items, train_ratings)

    model = OptimizedLatentFactorALS(
        n_factors=args.n_factors,
        lambda_reg=args.lambda_reg,
        use_parallel=args.use_parallel,
        n_workers=args.n_workers,
        dtype=np.float32,
    )

    print("Training model on synthetic data...")
    t0 = time.time()
    model.fit(
        train_csr,
        train_csc,
        n_iterations=args.n_iterations,
        verbose=True,
        compute_metrics_every=args.compute_metrics_every,
    )
    t1 = time.time()
    print(f"Training completed in {t1 - t0:.2f} seconds")

    print("Evaluating model on synthetic test set...")
    metrics = evaluate_model(model, test_users, test_items, test_ratings)
    print("Evaluation metrics:", metrics)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quickstart example for Rekomenda")
    p.add_argument(
        "--ratings",
        type=str,
        default=None,
        help="Path to ratings CSV (userId,movieId,rating,...)",
    )
    p.add_argument(
        "--movies",
        type=str,
        default=None,
        help="Path to movies CSV (movieId,title,genres)",
    )
    p.add_argument("--n-factors", type=int, default=32, help="Number of latent factors")
    p.add_argument(
        "--n-iterations", type=int, default=5, help="Number of ALS iterations"
    )
    p.add_argument(
        "--lambda-reg", type=float, default=0.1, help="Regularization lambda"
    )
    p.add_argument(
        "--use-parallel",
        action="store_true",
        help="Use parallel user updates (threaded).",
    )
    p.add_argument(
        "--n-workers", type=int, default=4, help="Number of parallel workers"
    )
    p.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of ratings for test set"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--chunk-size", type=int, default=1_000_000, help="Chunk size for CSV loader"
    )
    p.add_argument(
        "--compute-metrics-every",
        type=int,
        default=1,
        help="How often to compute metrics during training",
    )

    # Synthetic data options
    p.add_argument(
        "--synthetic-users", type=int, default=100, help="Users in synthetic dataset"
    )
    p.add_argument(
        "--synthetic-items", type=int, default=200, help="Items in synthetic dataset"
    )
    p.add_argument(
        "--synthetic-ratings",
        type=int,
        default=5000,
        help="Ratings in synthetic dataset",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    ratings_path = args.ratings
    movies_path = args.movies

    # Only attempt MovieLens loader if both paths are present and exist
    if (
        ratings_path
        and movies_path
        and os.path.exists(ratings_path)
        and os.path.exists(movies_path)
    ):
        train_on_movielens(ratings_path, movies_path, args)
    else:
        if ratings_path or movies_path:
            print(
                "Warning: ratings/movies path provided but one or both files do not exist. "
                "Falling back to synthetic data."
            )
        train_on_synthetic(args)


if __name__ == "__main__":
    main()
