"""Drift adapter for embedding model transitions.

Implements Orthogonal Procrustes alignment to adapt query embeddings
from a new embedding model space to an old model space using a learned
rotation matrix.

The adapter trains on paired embeddings (old and new) from the same source
and learns an optimal rotation that minimizes the squared Frobenius norm
of the difference between the rotated new embeddings and old embeddings.
"""

import numpy as np
from typing import Optional


class DriftAdapter:
    """Adapter for aligning embeddings across model versions via Orthogonal Procrustes."""

    def __init__(self, old_dim: int, new_dim: int):
        """
        Initialize drift adapter.

        Args:
            old_dim: Embedding dimension of the old (legacy) model
            new_dim: Embedding dimension of the new model

        Attributes:
            rotation_matrix: Learned orthogonal rotation matrix (d×d, where d=new_dim)
                           Set to None until train() is called.
        """
        self.old_dim = old_dim
        self.new_dim = new_dim
        self.rotation_matrix: Optional[np.ndarray] = None

    def train(
        self, old_embeddings: np.ndarray, new_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Train the adapter using paired embeddings.

        Applies Orthogonal Procrustes alignment:
        1. Center both embedding matrices (subtract mean)
        2. Compute optimal orthogonal rotation matrix via scipy.linalg.orthogonal_procrustes
        3. Store rotation matrix for later use in adapt_query()

        Args:
            old_embeddings: Old model embeddings, shape (n_samples, old_dim)
            new_embeddings: New model embeddings, shape (n_samples, new_dim)
                           Typically new_dim == old_dim for alignment

        Returns:
            rotation_matrix: Orthogonal rotation matrix, shape (new_dim, new_dim)
        """
        from scipy.linalg import orthogonal_procrustes

        # Center embeddings (subtract mean)
        old_centered = old_embeddings - old_embeddings.mean(axis=0)
        new_centered = new_embeddings - new_embeddings.mean(axis=0)

        # Compute optimal orthogonal rotation
        rotation_matrix, _ = orthogonal_procrustes(new_centered, old_centered)

        # Store as float32 for consistency
        self.rotation_matrix = rotation_matrix.astype(np.float32)

        return self.rotation_matrix

    def adapt_query(self, new_query_embedding: np.ndarray) -> np.ndarray:
        """
        Transform a query embedding from new model space to old model space.

        Applies the learned rotation matrix: adapted = new_query @ rotation_matrix.T

        Args:
            new_query_embedding: Query embedding from new model, shape (new_dim,)

        Returns:
            Adapted query embedding in old model space, shape (old_dim,)

        Raises:
            ValueError: If the adapter has not been trained yet
        """
        if self.rotation_matrix is None:
            raise ValueError("Adapter not trained. Call train() before adapt_query().")

        # Transform via matrix transpose (rotation_matrix.T converts new space -> old space)
        adapted = new_query_embedding @ self.rotation_matrix.T
        return adapted.astype(np.float32)

    def save(self, path: str) -> None:
        """
        Save rotation matrix to disk.

        Args:
            path: File path to save .npy file (rotation matrix stored as float32)
        """
        if self.rotation_matrix is None:
            raise ValueError("No rotation matrix to save. Train adapter first.")

        np.save(path, self.rotation_matrix.astype(np.float32))

    @staticmethod
    def load(path: str) -> "DriftAdapter":
        """
        Load a trained adapter from disk.

        Validates that the loaded matrix is square and float32 or float64.

        Args:
            path: File path to .npy file containing rotation matrix

        Returns:
            New DriftAdapter instance with loaded rotation matrix

        Raises:
            ValueError: If matrix is not square or dtype is not float32/float64
        """
        matrix = np.load(path, allow_pickle=False)

        # Validate: must be square
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Rotation matrix must be square. Got shape {matrix.shape}"
            )

        # Validate: dtype must be float32 or float64
        if matrix.dtype not in (np.float32, np.float64):
            raise ValueError(
                f"Rotation matrix dtype must be float32 or float64. Got {matrix.dtype}"
            )

        # Create adapter and set the loaded matrix
        d = matrix.shape[0]
        adapter = DriftAdapter(d, d)
        adapter.rotation_matrix = matrix.astype(np.float32)

        return adapter


if __name__ == "__main__":
    # Test basic functionality
    print("=== Drift Adapter Initialization Test ===")

    adapter = DriftAdapter(768, 768)
    print(f"Adapter initialized: old_dim={adapter.old_dim}, new_dim={adapter.new_dim}")
    print(f"Initial rotation_matrix: {adapter.rotation_matrix}")
    print("\nBasic initialization test passed")
