from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from .csc import CSCMatrix


class ALSRecommendationAnalyzer:
    """Analyze recommendations and identify polarizing movies from trained ALS models"""

    def __init__(self, model, item_id_to_name: Dict = dict()):
        """
        model: Trained LatentFactorALS or BiasOnlyALS model
        item_id_to_name: Dictionary mapping item IDs to names
        """
        self.model = model
        self.item_id_to_name = item_id_to_name or {}

    def create_dummy_user(
        self, liked_items: List[Tuple[int, float]]
    ) -> Tuple[float, np.ndarray]:
        """
        Create a dummy user with specific ratings

        Args:
            liked_items: List of (item_id, rating) tuples

        Returns:
            (user_bias, user_factors) tuple
        """
        item_indices = np.array([item_id for item_id, _ in liked_items])
        ratings = np.array([rating for _, rating in liked_items])
        n_ratings = len(liked_items)

        # Check if model has latent factors
        if hasattr(self.model, "item_factors") and self.model.item_factors.size > 0:
            # Latent factor model
            Q_u = self.model.item_factors[item_indices]
            Q_u_aug = np.column_stack([np.ones(n_ratings), Q_u])
            residuals = (
                ratings - self.model.global_mean - self.model.item_bias[item_indices]
            )

            A = Q_u_aug.T @ Q_u_aug
            A += self.model.lambda_reg * np.eye(self.model.n_factors + 1)
            b = Q_u_aug.T @ residuals

            x = np.linalg.solve(A, b)
            return x[0], x[1:]
        else:
            # Bias-only model
            residuals = (
                ratings - self.model.global_mean - self.model.item_bias[item_indices]
            )
            user_bias = np.sum(residuals) / (n_ratings + self.model.lambda_reg)
            return user_bias, np.array([])

    def get_recommendations(
        self,
        user_bias: float,
        user_factors: np.ndarray,
        exclude_items: List[int] | Set[int] = list(),
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Get top-k recommendations for a user

        Returns:
            List of (item_id, predicted_rating, item_name) tuples
        """
        exclude_items = set(exclude_items or [])

        # Predict ratings for all items
        if hasattr(self.model, "item_factors") and self.model.item_factors.size > 0:
            predictions = (
                self.model.global_mean
                + user_bias
                + self.model.item_bias
                + np.dot(self.model.item_factors, user_factors)
            )
        else:
            predictions = self.model.global_mean + user_bias + self.model.item_bias

        results = []
        for item_id, pred in enumerate(predictions):
            if item_id not in exclude_items:
                item_name = self.item_id_to_name.get(item_id, f"Item {item_id}")
                results.append((item_id, pred, item_name))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def find_similar_items(
        self, item_id: int, top_k: int = 10, use_cosine: bool = True
    ) -> List[Tuple[int, float, str]]:
        """Find items similar to a given item based on latent factors"""
        if not hasattr(self.model, "item_factors") or self.model.item_factors.size == 0:
            raise ValueError(
                "Model must have latent factors for similarity computation"
            )

        reference_factors = self.model.item_factors[item_id]
        similarities = []

        for i, item_factors in enumerate(self.model.item_factors):
            if i == item_id:
                continue

            if use_cosine:
                similarity = 1 - cosine(reference_factors, item_factors)
            else:
                similarity = -np.linalg.norm(reference_factors - item_factors)

            item_name = self.item_id_to_name.get(i, f"Item {i}")
            similarities.append((i, similarity, item_name))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_polarizing_movies(self, top_k: int = 20) -> List[Tuple[int, float, str]]:
        """
        Find polarizing movies by latent factor magnitude

        Returns:
            List of (item_id, factor_norm, item_name) tuples
        """
        if not hasattr(self.model, "item_factors") or self.model.item_factors.size == 0:
            raise ValueError("Model must have latent factors for polarization analysis")

        factor_norms = np.linalg.norm(self.model.item_factors, axis=1)
        results = []

        for item_id, norm in enumerate(factor_norms):
            item_name = self.item_id_to_name.get(item_id, f"Item {item_id}")
            results.append((item_id, norm, item_name))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compute_rating_statistics_fast(
        self, csc_matrix: CSCMatrix, top_k: int = 20
    ) -> pd.DataFrame:
        """Compute rating statistics for all items to identify polarizing movies"""
        stats = []

        for item_id in range(csc_matrix.num_items):
            start, end = csc_matrix.indptr[item_id], csc_matrix.indptr[item_id + 1]

            if start < end:
                item_ratings = csc_matrix.ratings[start:end]

                if (
                    hasattr(self.model, "item_factors")
                    and self.model.item_factors.size > 0
                ):
                    factor_norm = np.linalg.norm(self.model.item_factors[item_id])
                else:
                    factor_norm = abs(self.model.item_bias[item_id])

                stats.append(
                    {
                        "item_id": item_id,
                        "item_name": self.item_id_to_name.get(
                            item_id, f"Item {item_id}"
                        ),
                        "n_ratings": len(item_ratings),
                        "mean_rating": np.mean(item_ratings),
                        "rating_std": np.std(item_ratings),
                        "rating_variance": np.var(item_ratings),
                        "factor_norm": factor_norm,
                        "polarization_score": factor_norm * np.std(item_ratings),
                    }
                )

        df = pd.DataFrame(stats)
        return df.sort_values("polarization_score", ascending=False).head(top_k)

    def visualize_item_factors(
        self, item_ids: List[int], item_names: List[str] = list()
    ):
        """Visualize the latent factors for specific items"""
        if not hasattr(self.model, "item_factors") or self.model.item_factors.size == 0:
            raise ValueError("Model must have latent factors for visualization")

        if item_names is None:
            item_names = [self.item_id_to_name.get(i, f"Item {i}") for i in item_ids]

        factors = self.model.item_factors[item_ids]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(self.model.n_factors)
        width = 0.8 / len(item_ids)

        for i, (item_id, name) in enumerate(zip(item_ids, item_names)):
            offset = (i - len(item_ids) / 2) * width
            ax.bar(x + offset, factors[i], width, label=name, alpha=0.8)

        ax.set_xlabel("Latent Factor", fontsize=12)
        ax.set_ylabel("Factor Value", fontsize=12)
        ax.set_title("Latent Factor Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i}" for i in range(self.model.n_factors)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def test_recommendation_quality(
        self,
        seed_movie_id: int,
        seed_movie_name: str,
        seed_rating: float = 5.0,
        expected_similar_ids: List[int] = list(),
        top_k: int = 10,
    ):
        """Test if recommendations make sense for a user who liked a specific movie"""
        print(f"\n{'=' * 80}")
        print(
            f"RECOMMENDATION TEST: User who gave '{seed_movie_name}' a {seed_rating}/5"
        )
        print(f"{'=' * 80}\n")

        user_bias, user_factors = self.create_dummy_user([(seed_movie_id, seed_rating)])

        recommendations = self.get_recommendations(
            user_bias, user_factors, exclude_items=[seed_movie_id], top_k=top_k
        )

        print(f"Top {top_k} Recommendations:")
        print("-" * 80)
        for rank, (item_id, pred_rating, item_name) in enumerate(recommendations, 1):
            print(f"{rank:2d}. {item_name:50s} (predicted: {pred_rating:.2f})")

        if hasattr(self.model, "item_factors") and self.model.item_factors.size > 0:
            print(f"\n\nMost Similar Items to '{seed_movie_name}':")
            print("-" * 80)
            similar_items = self.find_similar_items(seed_movie_id, top_k=10)
            for rank, (item_id, similarity, item_name) in enumerate(similar_items, 1):
                print(f"{rank:2d}. {item_name:50s} (similarity: {similarity:.4f})")
        else:
            similar_items = []

        if expected_similar_ids:
            recommended_ids = {item_id for item_id, _, _ in recommendations}
            matches = recommended_ids.intersection(set(expected_similar_ids))
            print(
                f"\n\nExpected items found in top {top_k}: {len(matches)}/{len(expected_similar_ids)}"
            )
            if matches:
                print(
                    f"Matched items: {[self.item_id_to_name.get(i, f'Item {i}') for i in matches]}"
                )

        return {
            "recommendations": recommendations,
            "similar_items": similar_items,
            "user_bias": user_bias,
            "user_factors": user_factors,
        }
