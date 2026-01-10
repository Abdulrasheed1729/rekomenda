from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

from .csc import CSCMatrix
from .utils import compute_pca


class ALSRecommendationAnalyzer:
    def __init__(self, model, item_id_to_name: Dict = dict()):
        self.model = model
        self.item_id_to_name = item_id_to_name or {}

    def create_dummy_user(
        self, liked_items: List[Tuple[int, float]]
    ) -> Tuple[float, np.ndarray]:
        item_indices = np.array([item_id for item_id, _ in liked_items])
        ratings = np.array([rating for _, rating in liked_items])
        n_ratings = len(liked_items)

        if hasattr(self.model, "item_factors") and self.model.item_factors.size > 0:
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
        exclude_items = set(exclude_items or [])

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
        if not hasattr(self.model, "item_factors") or self.model.item_factors.size == 0:
            raise ValueError("Model must have latent factors for polarization analysis")

        factor_norms = np.linalg.norm(self.model.item_factors, axis=1)
        results = []

        for item_id, norm in enumerate(factor_norms):
            item_name = self.item_id_to_name.get(item_id, f"Item {item_id}")
            results.append((item_id, norm, item_name))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compute_rating_statistics_fast(self, csc_matrix: CSCMatrix, top_k: int = 20):
        """Compute rating statistics - returns dict instead of DataFrame"""
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
                        "mean_rating": float(np.mean(item_ratings)),
                        "rating_std": float(np.std(item_ratings)),
                        "rating_variance": float(np.var(item_ratings)),
                        "factor_norm": float(factor_norm),
                        "polarization_score": float(factor_norm * np.std(item_ratings)),
                    }
                )

        # Sort by polarization score
        stats.sort(key=lambda x: x["polarization_score"], reverse=True)
        return stats[:top_k]

    def visualize_item_factors(
        self,
        item_ids: List[int],
        item_names: List[str] = list(),
    ):
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

    def plot_embedding_scatter(
        self,
        n_items: int = 0,
        highlight_items: List[int] = list(),
        filter_by_title: str | None = None,
        min_ratings: int = 50,
        csc_matrix: CSCMatrix | None = None,
        figsize: Tuple[int, int] = (14, 10),
        alpha: float = 0.6,
        show_labels: bool = False,
        label_top_n: int = 20,
    ):
        """Plot 2D PCA scatter of item embeddings using custom PCA"""
        if not hasattr(self.model, "item_factors") or self.model.item_factors.size == 0:
            raise ValueError("Model must have latent factors for visualization")

        # Filter items by minimum ratings
        valid_items = np.arange(len(self.model.item_factors))
        if csc_matrix is not None and min_ratings > 0:
            item_rating_counts = np.diff(csc_matrix.indptr)
            valid_items = np.where(item_rating_counts >= min_ratings)[0]
            print(
                f"Filtering to {len(valid_items)} items with >= {min_ratings} ratings"
            )

        # Filter by title substring
        if filter_by_title:
            filtered = []
            for item_id in valid_items:
                name = self.item_id_to_name.get(item_id, "")
                if filter_by_title.lower() in name.lower():
                    filtered.append(item_id)
            valid_items = np.array(filtered)
            print(f"Filtered to {len(valid_items)} items matching '{filter_by_title}'")

        # Sample items if n_items specified
        if n_items is not None and len(valid_items) > n_items:
            norms = np.linalg.norm(self.model.item_factors[valid_items], axis=1)
            top_indices = np.argsort(norms)[-n_items:]
            valid_items = valid_items[top_indices]
            print(f"Sampling {n_items} most polarizing items")

        # Get embeddings
        embeddings = self.model.item_factors[valid_items]

        # Apply custom PCA
        print(f"Reducing {embeddings.shape[1]}D embeddings to 2D using custom PCA...")
        coords_2d, explained_var = compute_pca(embeddings, n_components=2)

        title = f"Item Embeddings (PCA - {explained_var[0]:.1%} + {explained_var[1]:.1%} = {sum(explained_var):.1%} variance)"

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot all points
        ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c="steelblue",
            alpha=alpha,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

        # Highlight specific items
        if highlight_items:
            highlight_mask = np.isin(valid_items, highlight_items)
            if highlight_mask.any():
                ax.scatter(
                    coords_2d[highlight_mask, 0],
                    coords_2d[highlight_mask, 1],
                    c="red",
                    s=200,
                    alpha=0.8,
                    edgecolors="darkred",
                    linewidth=2,
                    marker="*",
                    label="Highlighted",
                    zorder=5,
                )

        # Add labels
        if show_labels:
            for i, item_id in enumerate(valid_items):
                name = self.item_id_to_name.get(item_id, f"Item {item_id}")
                ax.annotate(
                    name,
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    fontsize=6,
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords="offset points",
                )
        else:
            norms = np.linalg.norm(embeddings, axis=1)
            top_indices = np.argsort(norms)[-label_top_n:]

            for idx in top_indices:
                item_id = valid_items[idx]
                name = self.item_id_to_name.get(item_id, f"Item {item_id}")
                ax.annotate(
                    name,
                    (coords_2d[idx, 0], coords_2d[idx, 1]),
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
                    xytext=(5, 5),
                    textcoords="offset points",
                    zorder=4,
                )

        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2)

        if highlight_items:
            ax.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

        return coords_2d, valid_items

    def test_recommendation_quality(
        self,
        seed_movie_id: int,
        seed_movie_name: str,
        seed_rating: float = 5.0,
        expected_similar_ids: List[int] = list(),
        top_k: int = 10,
    ):
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
