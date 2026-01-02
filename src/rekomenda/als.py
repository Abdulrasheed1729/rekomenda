from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .csc import CSCMatrix
from .csr import CSRMatrix


@dataclass
class ALSMetrics:
    """Store training metrics"""

    iterations: List[int] = field(default_factory=list)
    rmse: List[float] = field(default_factory=list)
    neg_log_likelihood: List[float] = field(default_factory=list)


class BiasOnlyALS:
    """
    Bias-only Alternating Least Squares
    Model: r_ui = μ + b_u + b_i
    """

    def __init__(self, lambda_reg: float = 0.1):
        """
        Args:
            lambda_reg: L2 regularization parameter
        """
        self.lambda_reg = lambda_reg
        self.global_mean = 0.0
        self.user_bias = np.array([])
        self.item_bias = np.array([])
        self.metrics = ALSMetrics()

    def fit(
        self,
        csr_matrix: CSRMatrix,
        csc_matrix: CSCMatrix,
        n_iterations: int = 10,
        verbose: bool = True,
    ):
        """
        Fit the bias-only model using ALS

        Args:
            csr_matrix: CSRMatrix for efficient user operations
            csc_matrix: CSCMatrix for efficient item operations
            n_iterations: Number of ALS iterations
            verbose: Print progress
        """
        # Initialize
        self.global_mean = np.mean(csr_matrix.ratings)
        self.user_bias = np.zeros(csr_matrix.num_users)
        self.item_bias = np.zeros(csr_matrix.num_items)

        for iteration in range(n_iterations):
            # Update user biases (fix item biases)
            self._update_user_biases(csr_matrix)

            # Update item biases (fix user biases)
            self._update_item_biases(csc_matrix)

            # Compute metrics
            rmse = self._compute_rmse(csr_matrix)
            nll = self._compute_nll(csr_matrix)

            self.metrics.iterations.append(iteration + 1)
            self.metrics.rmse.append(rmse)
            self.metrics.neg_log_likelihood.append(nll)

            if verbose:
                print(
                    f"Iteration {iteration + 1}/{n_iterations} - RMSE: {rmse:.4f}, NLL: {nll:.4f}"
                )

    def _update_user_biases(self, csr_matrix: CSRMatrix):
        """Update user biases using closed-form solution"""
        for u in range(csr_matrix.num_users):
            start, end = csr_matrix.indptr[u], csr_matrix.indptr[u + 1]
            if start == end:
                continue

            item_indices = csr_matrix.item_indices[start:end]
            ratings = csr_matrix.ratings[start:end]

            # Residuals: r_ui - μ - b_i
            residuals = ratings - self.global_mean - self.item_bias[item_indices]

            # Closed-form solution: b_u = (Σ residuals) / (n_u + λ)
            n_ratings = end - start
            self.user_bias[u] = np.sum(residuals) / (n_ratings + self.lambda_reg)

    def _update_item_biases(self, csc_matrix: CSCMatrix):
        """Update item biases using closed-form solution"""
        for i in range(csc_matrix.num_items):
            start, end = csc_matrix.indptr[i], csc_matrix.indptr[i + 1]
            if start == end:
                continue

            user_indices = csc_matrix.user_indices[start:end]
            ratings = csc_matrix.ratings[start:end]

            # Residuals: r_ui - μ - b_u
            residuals = ratings - self.global_mean - self.user_bias[user_indices]

            # Closed-form solution: b_i = (Σ residuals) / (n_i + λ)
            n_ratings = end - start
            self.item_bias[i] = np.sum(residuals) / (n_ratings + self.lambda_reg)

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """Predict ratings for user-item pairs"""
        return (
            self.global_mean
            + self.user_bias[user_indices]
            + self.item_bias[item_indices]
        )

    def _compute_rmse(self, csr_matrix: CSRMatrix) -> float:
        """Compute Root Mean Square Error"""
        # Reconstruct user indices from indptr
        user_indices = np.repeat(
            np.arange(csr_matrix.num_users), np.diff(csr_matrix.indptr)
        )

        predictions = self.predict(user_indices, csr_matrix.item_indices)
        mse = np.mean((csr_matrix.ratings - predictions) ** 2)
        return np.sqrt(mse)

    def _compute_nll(self, csr_matrix: CSRMatrix) -> float:
        """Compute Negative Log Likelihood (without constant terms)"""
        # Reconstruct user indices from indptr
        user_indices = np.repeat(
            np.arange(csr_matrix.num_users), np.diff(csr_matrix.indptr)
        )

        predictions = self.predict(user_indices, csr_matrix.item_indices)

        # Data likelihood: -0.5 * Σ(r - r̂)²
        data_term = 0.5 * np.sum((csr_matrix.ratings - predictions) ** 2)

        # Regularization: 0.5 * λ * (||b_u||² + ||b_i||²)
        reg_term = (
            0.5
            * self.lambda_reg
            * (np.sum(self.user_bias**2) + np.sum(self.item_bias**2))
        )

        return data_term + reg_term

    def plot_metrics(self):
        """Plot RMSE and Negative Log Likelihood over iterations"""
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
        plt.show()


class LatentFactorALS:
    """
    Latent Factor ALS with biases
    Model: r_ui = μ + b_u + b_i + p_u^T q_i
    """

    def __init__(self, n_factors: int = 10, lambda_reg: float = 0.1):
        """
        Arguments:
            n_factors: Number of latent factors
            lambda_reg: L2 regularization parameter
        """
        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.global_mean = 0.0
        self.user_bias = np.array([])
        self.item_bias = np.array([])
        self.user_factors = np.array([])  # P matrix (num_users x n_factors)
        self.item_factors = np.array([])  # Q matrix (num_items x n_factors)
        self.metrics = ALSMetrics()

    def fit(
        self,
        csr_matrix: CSRMatrix,
        csc_matrix: CSCMatrix,
        n_iterations: int = 10,
        verbose: bool = True,
    ):
        """
        Fit the latent factor model using ALS

        Args:
            csr_matrix: CSRMatrix for efficient user operations
            csc_matrix: CSCMatrix for efficient item operations
            n_iterations: Number of ALS iterations
            verbose: Print progress
        """
        # Initialize
        self.global_mean = np.mean(csr_matrix.ratings)
        self.user_bias = np.zeros(csr_matrix.num_users)
        self.item_bias = np.zeros(csr_matrix.num_items)

        # Initialize latent factors with small random values
        self.user_factors = np.random.normal(
            0, 0.1, (csr_matrix.num_users, self.n_factors)
        )
        self.item_factors = np.random.normal(
            0, 0.1, (csr_matrix.num_items, self.n_factors)
        )

        for iteration in range(n_iterations):
            # Update user factors and biases (fix item factors)
            self._update_user_factors(csr_matrix)

            # Update item factors and biases (fix user factors)
            self._update_item_factors(csc_matrix)

            # Compute metrics
            rmse = self._compute_rmse(csr_matrix)
            nll = self._compute_nll(csr_matrix)

            self.metrics.iterations.append(iteration + 1)
            self.metrics.rmse.append(rmse)
            self.metrics.neg_log_likelihood.append(nll)

            if verbose:
                print(
                    f"Iteration {iteration + 1}/{n_iterations} - RMSE: {rmse:.4f}, NLL: {nll:.4f}"
                )

    def _update_user_factors(self, csr_matrix: CSRMatrix):
        """Update user factors and biases using ridge regression"""
        for u in range(csr_matrix.num_users):
            start, end = csr_matrix.indptr[u], csr_matrix.indptr[u + 1]
            if start == end:
                continue

            item_indices = csr_matrix.item_indices[start:end]
            ratings = csr_matrix.ratings[start:end]
            n_ratings = end - start

            # Get item factors for rated items: Q_u (n_ratings x n_factors)
            Q_u = self.item_factors[item_indices]

            # Augment with bias column: [1, q_i] for each item
            Q_u_aug = np.column_stack([np.ones(n_ratings), Q_u])

            # Residuals: r_ui - μ - b_i
            residuals = ratings - self.global_mean - self.item_bias[item_indices]

            # Solve: (Q_u^T Q_u + λI) x = Q_u^T residuals
            # where x = [b_u, p_u]
            A = Q_u_aug.T @ Q_u_aug
            A += self.lambda_reg * np.eye(self.n_factors + 1)
            b = Q_u_aug.T @ residuals

            x = np.linalg.solve(A, b)
            self.user_bias[u] = x[0]
            self.user_factors[u] = x[1:]

    def _update_item_factors(self, csc_matrix: CSCMatrix):
        """Update item factors and biases using ridge regression"""
        for i in range(csc_matrix.num_items):
            start, end = csc_matrix.indptr[i], csc_matrix.indptr[i + 1]
            if start == end:
                continue

            user_indices = csc_matrix.user_indices[start:end]
            ratings = csc_matrix.ratings[start:end]
            n_ratings = end - start

            # Get user factors for users who rated this item: P_i (n_ratings x n_factors)
            P_i = self.user_factors[user_indices]

            # Augment with bias column: [1, p_u] for each user
            P_i_aug = np.column_stack([np.ones(n_ratings), P_i])

            # Residuals: r_ui - μ - b_u
            residuals = ratings - self.global_mean - self.user_bias[user_indices]

            # Solve: (P_i^T P_i + λI) x = P_i^T residuals
            # where x = [b_i, q_i]
            A = P_i_aug.T @ P_i_aug
            A += self.lambda_reg * np.eye(self.n_factors + 1)
            b = P_i_aug.T @ residuals

            x = np.linalg.solve(A, b)
            self.item_bias[i] = x[0]
            self.item_factors[i] = x[1:]

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """Predict ratings for user-item pairs"""
        # Base prediction: μ + b_u + b_i
        predictions = (
            self.global_mean
            + self.user_bias[user_indices]
            + self.item_bias[item_indices]
        )

        # Add latent factor interaction: p_u^T q_i
        for idx, (u, i) in enumerate(zip(user_indices, item_indices)):
            predictions[idx] += np.dot(self.user_factors[u], self.item_factors[i])

        return predictions

    def _compute_rmse(self, csr_matrix: CSRMatrix) -> float:
        """Compute Root Mean Square Error"""
        # Reconstruct user indices from indptr
        user_indices = np.repeat(
            np.arange(csr_matrix.num_users), np.diff(csr_matrix.indptr)
        )

        predictions = self.predict(user_indices, csr_matrix.item_indices)
        mse = np.mean((csr_matrix.ratings - predictions) ** 2)
        return np.sqrt(mse)

    def _compute_nll(self, csr_matrix: CSRMatrix) -> float:
        """Compute Negative Log Likelihood"""
        # Reconstruct user indices from indptr
        user_indices = np.repeat(
            np.arange(csr_matrix.num_users), np.diff(csr_matrix.indptr)
        )

        predictions = self.predict(user_indices, csr_matrix.item_indices)

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

        return data_term + reg_term

    def plot_metrics(self):
        """Plot RMSE and Negative Log Likelihood over iterations"""
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
        plt.show()
