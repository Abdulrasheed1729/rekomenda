from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .csc import CSCMatrix
from .csr import CSRMatrix
from .utils import (
    predict_batch,
    update_user_factors_parallel,
)


@dataclass
class ALSMetrics:
    """Store training metrics"""

    iterations: List[int] = field(default_factory=list)
    rmse: List[float] = field(default_factory=list)
    neg_log_likelihood: List[float] = field(default_factory=list)


class ALSModel:
    pass


class BiasOnlyALS:
    def __init__(self, lambda_reg: float = 0.1, dtype=np.float32):
        self.lambda_reg = lambda_reg
        self.dtype = dtype
        self.global_mean = 0.0
        self.user_bias = np.array([])
        self.item_bias = np.array([])
        self.metrics = ALSMetrics()
        self._user_indices_cache = np.array([])

    def fit(
        self,
        csr_matrix: CSRMatrix,
        csc_matrix: CSCMatrix,
        n_iterations: int = 10,
        verbose: bool = True,
        convergence_threshold: float = 1e-4,
    ):
        self.global_mean = np.mean(csr_matrix.ratings).astype(self.dtype)
        self.user_bias = np.zeros(csr_matrix.num_users, dtype=self.dtype)
        self.item_bias = np.zeros(csr_matrix.num_items, dtype=self.dtype)

        self._user_indices_cache = np.repeat(
            np.arange(csr_matrix.num_users), np.diff(csr_matrix.indptr)
        )

        prev_rmse = float("inf")
        for iteration in range(n_iterations):
            self._update_user_biases(csr_matrix)
            self._update_item_biases(csc_matrix)

            rmse = self._compute_rmse(csr_matrix)
            nll = self._compute_nll(csr_matrix)

            self.metrics.iterations.append(iteration + 1)
            self.metrics.rmse.append(rmse)
            self.metrics.neg_log_likelihood.append(nll)

            if verbose:
                print(
                    f"Iteration {iteration + 1}/{n_iterations} - RMSE: {rmse:.4f}, NLL: {nll:.4f}"
                )

            if abs(prev_rmse - rmse) < convergence_threshold:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            prev_rmse = rmse

    def _update_user_biases(self, csr_matrix: CSRMatrix):
        for u in range(csr_matrix.num_users):
            start, end = csr_matrix.indptr[u], csr_matrix.indptr[u + 1]
            if start == end:
                continue
            item_indices = csr_matrix.item_indices[start:end]
            ratings = csr_matrix.ratings[start:end]
            residuals = ratings - self.global_mean - self.item_bias[item_indices]
            n_ratings = end - start
            self.user_bias[u] = np.sum(residuals) / (n_ratings + self.lambda_reg)

    def _update_item_biases(self, csc_matrix: CSCMatrix):
        for i in range(csc_matrix.num_items):
            start, end = csc_matrix.indptr[i], csc_matrix.indptr[i + 1]
            if start == end:
                continue
            user_indices = csc_matrix.user_indices[start:end]
            ratings = csc_matrix.ratings[start:end]
            residuals = ratings - self.global_mean - self.user_bias[user_indices]
            n_ratings = end - start
            self.item_bias[i] = np.sum(residuals) / (n_ratings + self.lambda_reg)

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        return (
            self.global_mean
            + self.user_bias[user_indices]
            + self.item_bias[item_indices]
        )

    def _compute_rmse(self, csr_matrix: CSRMatrix) -> float:
        predictions = self.predict(self._user_indices_cache, csr_matrix.item_indices)
        mse = np.mean((csr_matrix.ratings - predictions) ** 2)
        return float(np.sqrt(mse))

    def _compute_nll(self, csr_matrix: CSRMatrix) -> float:
        predictions = self.predict(self._user_indices_cache, csr_matrix.item_indices)
        data_term = 0.5 * np.sum((csr_matrix.ratings - predictions) ** 2)
        reg_term = (
            0.5
            * self.lambda_reg
            * (np.sum(self.user_bias**2) + np.sum(self.item_bias**2))
        )
        return float(data_term + reg_term)

    def plot_metrics(self, output: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self.metrics.iterations, self.metrics.rmse, "b-o", linewidth=2)
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("RMSE", fontsize=12)
        ax1.set_title("Root Mean Square Error", fontsize=14)
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            self.metrics.iterations, self.metrics.neg_log_likelihood, "r-o", linewidth=2
        )
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Negative Log Likelihood", fontsize=12)
        ax2.set_title("Negative Log Likelihood", fontsize=14)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output, format="pdf")
        plt.show()


class LatentFactorALS:
    def __init__(self, n_factors: int = 10, lambda_reg: float = 0.1, dtype=np.float32):
        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.dtype = dtype
        self.global_mean = 0.0
        self.user_bias = np.array([])
        self.item_bias = np.array([])
        self.user_factors = np.array([])
        self.item_factors = np.array([])
        self.metrics = ALSMetrics()
        self._user_indices_cache = np.array([])

    def fit(
        self,
        csr_matrix: CSRMatrix,
        csc_matrix: CSCMatrix,
        n_iterations: int = 10,
        verbose: bool = True,
        convergence_threshold: float = 1e-4,
    ):
        self.global_mean = np.mean(csr_matrix.ratings).astype(self.dtype)
        self.user_bias = np.zeros(csr_matrix.num_users, dtype=self.dtype)
        self.item_bias = np.zeros(csr_matrix.num_items, dtype=self.dtype)

        self.user_factors = np.random.normal(
            0, 0.1, (csr_matrix.num_users, self.n_factors)
        ).astype(self.dtype)
        self.item_factors = np.random.normal(
            0, 0.1, (csr_matrix.num_items, self.n_factors)
        ).astype(self.dtype)

        self._user_indices_cache = np.repeat(
            np.arange(csr_matrix.num_users), np.diff(csr_matrix.indptr)
        )

        prev_rmse = float("inf")
        for iteration in range(n_iterations):
            self._update_user_factors(csr_matrix)
            self._update_item_factors(csc_matrix)

            rmse = self._compute_rmse(csr_matrix)
            nll = self._compute_nll(csr_matrix)

            self.metrics.iterations.append(iteration + 1)
            self.metrics.rmse.append(rmse)
            self.metrics.neg_log_likelihood.append(nll)

            if verbose:
                print(
                    f"Iteration {iteration + 1}/{n_iterations} - RMSE: {rmse:.4f}, NLL: {nll:.4f}"
                )

            if abs(prev_rmse - rmse) < convergence_threshold:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            prev_rmse = rmse

    def _update_user_factors(self, csr_matrix: CSRMatrix):
        for u in range(csr_matrix.num_users):
            start, end = csr_matrix.indptr[u], csr_matrix.indptr[u + 1]
            if start == end:
                continue

            item_indices = csr_matrix.item_indices[start:end]
            ratings = csr_matrix.ratings[start:end]
            n_ratings = end - start

            Q_u = self.item_factors[item_indices]
            Q_u_aug = np.column_stack([np.ones(n_ratings, dtype=self.dtype), Q_u])
            residuals = ratings - self.global_mean - self.item_bias[item_indices]

            A = Q_u_aug.T @ Q_u_aug
            A += self.lambda_reg * np.eye(self.n_factors + 1, dtype=self.dtype)
            b = Q_u_aug.T @ residuals

            x = np.linalg.solve(A, b)
            self.user_bias[u] = x[0]
            self.user_factors[u] = x[1:]

    def _update_item_factors(self, csc_matrix: CSCMatrix):
        for i in range(csc_matrix.num_items):
            start, end = csc_matrix.indptr[i], csc_matrix.indptr[i + 1]
            if start == end:
                continue

            user_indices = csc_matrix.user_indices[start:end]
            ratings = csc_matrix.ratings[start:end]
            n_ratings = end - start

            P_i = self.user_factors[user_indices]
            P_i_aug = np.column_stack([np.ones(n_ratings, dtype=self.dtype), P_i])
            residuals = ratings - self.global_mean - self.user_bias[user_indices]

            A = P_i_aug.T @ P_i_aug
            A += self.lambda_reg * np.eye(self.n_factors + 1, dtype=self.dtype)
            b = P_i_aug.T @ residuals

            x = np.linalg.solve(A, b)
            self.item_bias[i] = x[0]
            self.item_factors[i] = x[1:]

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        predictions = (
            self.global_mean
            + self.user_bias[user_indices]
            + self.item_bias[item_indices]
        )
        user_factors_selected = self.user_factors[user_indices]
        item_factors_selected = self.item_factors[item_indices]
        predictions += np.sum(user_factors_selected * item_factors_selected, axis=1)
        return predictions

    def _compute_rmse(self, csr_matrix: CSRMatrix) -> float:
        predictions = self.predict(self._user_indices_cache, csr_matrix.item_indices)
        mse = np.mean((csr_matrix.ratings - predictions) ** 2)
        return float(np.sqrt(mse))

    def _compute_nll(self, csr_matrix: CSRMatrix) -> float:
        predictions = self.predict(self._user_indices_cache, csr_matrix.item_indices)
        data_term = 0.5 * np.sum((csr_matrix.ratings - predictions) ** 2)
        reg_term = (
            0.5
            * self.lambda_reg
            * (
                np.sum(self.user_bias**2)
                + np.sum(self.item_bias**2)
                + np.sum(self.user_factors**2)
                + np.sum(self.item_factors**2)
            )
        )
        return float(data_term + reg_term)

    def plot_metrics(self, output: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self.metrics.iterations, self.metrics.rmse, "b-o", linewidth=2)
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("RMSE", fontsize=12)
        ax1.set_title("Root Mean Square Error", fontsize=14)
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            self.metrics.iterations, self.metrics.neg_log_likelihood, "r-o", linewidth=2
        )
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Negative Log Likelihood", fontsize=12)
        ax2.set_title("Negative Log Likelihood", fontsize=14)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output, format="pdf")
        plt.show()


class OptimizedLatentFactorALS:
    """
    Highly optimized ALS implementation for large-scale datasets

    Features:
    - Numba acceleration for bias updates
    - Optional parallel processing for factor updates
    - Batch prediction
    - float32 for memory efficiency
    - Progress tracking with less frequent metric computation
    """

    def __init__(
        self,
        n_factors: int = 10,
        lambda_reg: float = 0.1,
        n_workers: int = 1,
        use_parallel: bool = False,
        dtype=np.float32,
    ):
        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.n_workers = n_workers
        self.use_parallel = use_parallel
        self.dtype = dtype

        self.global_mean = 0.0
        self.user_bias: np.ndarray = np.array([])
        self.item_bias: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([])
        self.item_factors: np.ndarray = np.array([])
        self._user_indices_cache = None
        self.metrics = ALSMetrics()

    def fit(
        self,
        csr_matrix,
        csc_matrix,
        n_iterations: int = 10,
        verbose: bool = True,
        compute_metrics_every: int = 1,  # Compute metrics every N iterations
    ):
        """
        Fit the model with optimizations

        Args:
            compute_metrics_every: Compute expensive metrics every N iterations
                                (set to 5-10 for huge datasets)
        """
        # Initialize
        self.global_mean = np.mean(csr_matrix.ratings).astype(self.dtype)
        self.user_bias = np.zeros(csr_matrix.num_users, dtype=self.dtype)
        self.item_bias = np.zeros(csr_matrix.num_items, dtype=self.dtype)

        self.user_factors = np.random.normal(
            0, 0.1, (csr_matrix.num_users, self.n_factors)
        ).astype(self.dtype)
        self.item_factors = np.random.normal(
            0, 0.1, (csr_matrix.num_items, self.n_factors)
        ).astype(self.dtype)

        # Cache user indices
        self._user_indices_cache = np.repeat(
            np.arange(csr_matrix.num_users), np.diff(csr_matrix.indptr)
        )

        for iteration in range(n_iterations):
            # Update user factors
            if self.use_parallel:
                update_user_factors_parallel(
                    csr_matrix,
                    self.item_factors,
                    self.item_bias,
                    self.global_mean,
                    self.lambda_reg,
                    self.n_factors,
                    self.user_bias,
                    self.user_factors,
                    self.n_workers,
                )
            else:
                self._update_user_factors(csr_matrix)

            # Update item factors
            self._update_item_factors(csc_matrix)

            # Compute metrics only every N iterations
            if (
                iteration + 1
            ) % compute_metrics_every == 0 or iteration == n_iterations - 1:
                rmse = self._compute_rmse_batch(csr_matrix)
                nll = self._compute_nll_batch(csr_matrix)

                self.metrics.iterations.append(iteration + 1)
                self.metrics.rmse.append(rmse)
                self.metrics.neg_log_likelihood.append(nll)

                if verbose:
                    print(
                        f"Iteration {iteration + 1}/{n_iterations} - RMSE: {rmse:.4f}, NLL: {nll:.4f}"
                    )

    def _update_user_factors(self, csr_matrix):
        """Standard user factor update"""
        for u in range(csr_matrix.num_users):
            start, end = csr_matrix.indptr[u], csr_matrix.indptr[u + 1]
            if start == end:
                continue

            item_indices = csr_matrix.item_indices[start:end]
            ratings = csr_matrix.ratings[start:end]
            n_ratings = end - start

            Q_u = self.item_factors[item_indices]
            Q_u_aug = np.column_stack([np.ones(n_ratings, dtype=self.dtype), Q_u])
            residuals = ratings - self.global_mean - self.item_bias[item_indices]

            A = Q_u_aug.T @ Q_u_aug
            A += self.lambda_reg * np.eye(self.n_factors + 1, dtype=self.dtype)
            b = Q_u_aug.T @ residuals

            x = np.linalg.solve(A, b)
            self.user_bias[u] = x[0]
            self.user_factors[u] = x[1:]

    def _update_item_factors(self, csc_matrix):
        """Standard item factor update"""
        for i in range(csc_matrix.num_items):
            start, end = csc_matrix.indptr[i], csc_matrix.indptr[i + 1]
            if start == end:
                continue

            user_indices = csc_matrix.user_indices[start:end]
            ratings = csc_matrix.ratings[start:end]
            n_ratings = end - start

            P_i = self.user_factors[user_indices]
            P_i_aug = np.column_stack([np.ones(n_ratings, dtype=self.dtype), P_i])
            residuals = ratings - self.global_mean - self.user_bias[user_indices]

            A = P_i_aug.T @ P_i_aug
            A += self.lambda_reg * np.eye(self.n_factors + 1, dtype=self.dtype)
            b = P_i_aug.T @ residuals

            x = np.linalg.solve(A, b)
            self.item_bias[i] = x[0]
            self.item_factors[i] = x[1:]

    def _compute_rmse_batch(self, csr_matrix, batch_size=1000000):
        """Compute RMSE using batch processing"""
        predictions = predict_batch(
            self._user_indices_cache,
            csr_matrix.item_indices,
            self.global_mean,
            self.user_bias,
            self.item_bias,
            self.user_factors,
            self.item_factors,
            batch_size,
        )
        mse = np.mean((csr_matrix.ratings - predictions) ** 2)
        return float(np.sqrt(mse))

    def _compute_nll_batch(self, csr_matrix, batch_size=1000000):
        """Compute Negative Log Likelihood using batch processing"""
        predictions = predict_batch(
            self._user_indices_cache,
            csr_matrix.item_indices,
            self.global_mean,
            self.user_bias,
            self.item_bias,
            self.user_factors,
            self.item_factors,
            batch_size,
        )

        # Data likelihood: -0.5 * Σ(r - r̂)²
        data_term = 0.5 * np.sum((csr_matrix.ratings - predictions) ** 2)

        # Regularization: 0.5 * λ * (||b_u||² + ||b_i||² + ||P||² + ||Q||²)
        reg_term = (
            0.5
            * self.lambda_reg
            * (
                np.sum(self.user_bias**2)
                + np.sum(self.item_bias**2)
                + np.sum(self.user_factors**2)
                + np.sum(self.item_factors**2)
            )
        )

        return float(data_term + reg_term)

    def predict(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        batch_size: int = 100000,
    ) -> np.ndarray:
        """
        Predict ratings for user-item pairs using batch processing

        Args:
            user_indices: Array of user indices
            item_indices: Array of item indices
            batch_size: Size of batches for processing (prevents memory issues)

        Returns:
            Array of predicted ratings
        """
        return predict_batch(
            user_indices,
            item_indices,
            self.global_mean,
            self.user_bias,
            self.item_bias,
            self.user_factors,
            self.item_factors,
            batch_size,
        )

    def plot_metrics(self, output):
        """Plot RMSE and Negative Log Likelihood over iterations"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot RMSE
        ax1.plot(self.metrics.iterations, self.metrics.rmse, "b-o", linewidth=2)
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("RMSE", fontsize=12)
        ax1.set_title("Root Mean Square Error", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot NLL
        ax2.plot(
            self.metrics.iterations, self.metrics.neg_log_likelihood, "r-o", linewidth=2
        )
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Negative Log Likelihood", fontsize=12)
        ax2.set_title("Negative Log Likelihood", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output, format="pdf")
        plt.show()
