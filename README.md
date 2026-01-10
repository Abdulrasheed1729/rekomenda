# Rekomenda

A lightweight, readable implementation of Alternating Least Squares (ALS) for collaborative filtering recommender systems, implemented in pure Python with optional performance optimizations (Numba, parallel updates, batch prediction). This repository is intended as a teaching / research codebase and a starting point for experimentation with matrix representations and ALS variants.

Contents
- Models: `BiasOnlyALS`, `LatentFactorALS`, `OptimizedLatentFactorALS` (in `src/rekomenda/als.py`)
- Sparse matrix utilities: `COOMatrix`, `CSRMatrix`, `CSCMatrix` (in `src/rekomenda/*.py`)
- Data and utility helpers: data loaders, batch prediction, parallel/numba helpers (in `src/rekomenda/utils.py`)
- Notebooks and a short report in `notebooks/` and `report/`

Table of contents
- [Why this project](#why-this-project)
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [API reference (high level)](#api-reference-high-level)
- [Performance tips](#performance-tips)
- [Project structure](#project-structure)
- [Notebooks and examples](#notebooks-and-examples)
- [Contributing](#contributing)
- [Author & contact](#author--contact)
- [License](#license)

Why this project
----------------
This codebase aims to be:
- Readable: straightforward implementations of ALS variants for learning and modification.
- Practical: includes utilities to load large datasets (e.g. MovieLens) in a memory efficient way and evaluate models.
- Extensible: an `OptimizedLatentFactorALS` implementation provides hooks for Numba acceleration and parallel factor updates.

Use it to:
- Learn ALS internals (bias updates, alternating updates of user/item latent factors).
- Prototype recommender system ideas on medium- to large-scale datasets.
- Compare simple ALS variants (bias-only vs factor models) and measure RMSE / negative log likelihood.

Features
--------
- `BiasOnlyALS`: simple baseline that fits global mean plus per-user and per-item biases.
- `LatentFactorALS`: classic ALS with latent user/item factors and bias terms.
- `OptimizedLatentFactorALS`:
  - Batch prediction helper to avoid memory pressure.
  - Optional parallel user-factor updates via thread workers.
  - Numba-accelerated bias update functions (in `utils.py`).
- Sparse matrix wrappers:
  - `COOMatrix`, `CSRMatrix`, `CSCMatrix` for efficient traversal during updates.
- Data loaders:
  - Chunked MovieLens loader to build train CSR/CSC matrices without loading the entire ratings file into memory.
- Evaluation helpers:
  - `evaluate_model` to compute RMSE and MAE on a test split.
- Simple PCA helper for embedding visualization.

Installation
------------
Requirements
- Python 3.10 - 3.13 (pyproject specifies `>=3.10,<3.14`)
- Core dependencies (from `pyproject.toml`): `numpy`, `numba`, `pandas`, `matplotlib`, `scipy`, `jupyter`

Quick install (editable mode)
1. Create and activate a Python virtual environment:
   - `python -m venv .venv && source .venv/bin/activate` (Linux/macOS)
   - `python -m venv .venv && .venv\\Scripts\\activate` (Windows)
2. Install the package in editable mode:
   - `pip install -e .`
3. (Optional) Install dev tooling:
   - `pip install ruff`

Note: If you don't want to install the package, you can still run the notebooks and import modules by adding the repository root to your `PYTHONPATH`.

Quick start
-----------
Below is a minimal example demonstrating how to load MovieLens-style data, train an `OptimizedLatentFactorALS` model and evaluate it on a held-out split.

```/dev/null/example.py#L1-40
from rekomenda.utils import load_movielens_data, evaluate_model
from rekomenda.als import OptimizedLatentFactorALS

# Example usage (replace paths with your files)
ratings_csv = "data/ratings.csv"  # expected columns: userId,movieId,rating,...
movies_csv = "data/movies.csv"    # expected columns: movieId,title,genres

# Load data (memory-efficient chunked loader)
train_csr, train_csc, test_data, item_id_to_name, movie_id_to_idx = load_movielens_data(
    ratings_csv, movies_csv, test_size=0.2, random_seed=42, chunk_size=1_000_000
)

# Create and fit model
model = OptimizedLatentFactorALS(n_factors=50, lambda_reg=0.1, use_parallel=True, n_workers=4)
model.fit(train_csr, train_csc, n_iterations=10, verbose=True, compute_metrics_every=1)

# Evaluate
test_users, test_items, test_ratings = test_data
metrics = evaluate_model(model, test_users, test_items, test_ratings)
print(metrics)
```

API reference (high level)
--------------------------
- `rekomenda.coo.COOMatrix`
  - Lightweight COO representation created from raw arrays or from `load_*` functions.
- `rekomenda.csr.CSRMatrix`
  - Row-oriented sparse view (fast iterate over user rows). Useful for user-update passes.
  - Use `CSRMatrix.from_raw_data(users, items, ratings)` or `CSRMatrix.from_coo(coo)`.
- `rekomenda.csc.CSCMatrix`
  - Column-oriented sparse view (fast iterate over item columns).
  - Use `CSCMatrix.from_raw_data(...)` or `CSCMatrix.from_coo(coo)`.
- `rekomenda.als.BiasOnlyALS(lambda_reg=0.1, dtype=np.float32)`
  - Methods:
    - `fit(csr_matrix, csc_matrix, n_iterations=10, verbose=True)`
    - `predict(user_indices, item_indices) -> np.ndarray`
    - `plot_metrics(output_path)` (saves PDF)
- `rekomenda.als.LatentFactorALS(n_factors=10, lambda_reg=0.1, dtype=np.float32)`
  - Classic ALS with explicit alternating closed-form solves for per-user/per-item factors + biases.
  - Methods mirror `BiasOnlyALS`.
- `rekomenda.als.OptimizedLatentFactorALS(n_factors=10, lambda_reg=0.1, n_workers=1, use_parallel=False, dtype=np.float32)`
  - More practical for larger datasets. Supports:
    - `fit(csr_matrix, csc_matrix, n_iterations=10, verbose=True, compute_metrics_every=1)`
    - `predict(user_indices, item_indices, batch_size=100000) -> np.ndarray`
    - `plot_metrics(output_path)`
  - Internally uses `predict_batch` for batched predictions, and `update_user_factors_parallel` (thread-based) when `use_parallel=True`.

Utilities (`src/rekomenda/utils.py`)
- `update_user_biases_numba`, `update_item_biases_numba`: Numba-accelerated loops for bias computation (when compiled).
- `update_user_factors_parallel`: thread-worker based parallelization for user factor updates.
- `predict_batch`: batched prediction helper to avoid memory blowups when scoring millions of pairs.
- `load_movielens_data`: chunked loader that creates `CSRMatrix` and `CSCMatrix` for the training set and returns a small test split; designed for MovieLens-style CSVs.

Performance tips
----------------
- dtype: The code uses `float32` by default in `OptimizedLatentFactorALS` to reduce memory usage. This can speed up training on large datasets and lower memory footprint.
- Numba: The bias updates in `utils.py` are `@njit`-decorated and should be compiled on first call; ensure `numba` is installed and allow compilation time before timing training.
- Parallel updates: Setting `use_parallel=True` and `n_workers` > 1 yields faster user-factor updates on multi-core systems; test and tune `n_workers` for your workload.
- Batch prediction: Set `batch_size` in `predict`/`predict_batch` to an appropriate value for your available RAM.
- Metric frequency: For very large datasets, set `compute_metrics_every` to a larger number (e.g., 5 or 10) to avoid expensive scoring every iteration.

Project structure
-----------------
Top-level:
- `src/rekomenda/als.py` — ALS model implementations (Bias-only, Latent factor, Optimized).
- `src/rekomenda/utils.py` — utilities (loaders, numba helpers, batch predict, evaluation).
- `src/rekomenda/coo.py` — COO matrix wrapper.
- `src/rekomenda/csr.py` — CSR matrix wrapper.
- `src/rekomenda/csc.py` — CSC matrix wrapper.
- `notebooks/` — Jupyter notebooks demonstrating experiments and visualizations.
- `report/` — PDF/LaTeX report and figures produced for a project write-up.
- `pyproject.toml` — packaging and dependencies.

Notebooks and examples
----------------------
There are several notebooks under `notebooks/` demonstrating:
- Practical walkthroughs for training and evaluation.
- Visualizations of embeddings (PCA) and training metrics.
- Experiments comparing bias-only vs latent-factor models.

Open the notebook(s) and run them after installing dependencies and preparing the dataset.

Contributing
------------
This project is intended to be simple to understand and modify. If you'd like to contribute:
- Open issues for bugs or enhancement requests.
- For code changes, fork, create a feature branch and open a pull request.
- Run and add tests when possible. (There are no formal tests bundled in the repo currently; adding unit tests for matrix builders and core update routines would be valuable.)

Notes / Known limitations
-------------------------
- No automated tests are currently included; use the notebooks as informal tests.
- The code aims to be readable; some parts (e.g., production-grade sparse data handling, advanced regularization options) are intentionally minimal.
- The `COOMatrix.from_raw_data` and matrix constructors perform internal `np.unique` and `np.bincount` operations; for extremely large ID spaces you may need to pre-map IDs externally.
- This project is licensed under the MIT License. See the `LICENSE` file in the repository root for the full license text and terms.

Author & contact
----------------
Author: Abdulrasheed Fawole  
Email: fawomath@gmail.com

Acknowledgements
----------------
This repository contains standard ALS algorithms and utilities inspired by common recommender systems literature and teaching materials. It is intended as an educational / experimental codebase.

License
-------
This project is released under the MIT License. See the `LICENSE` file at the repository root for the complete license text and permissions.
