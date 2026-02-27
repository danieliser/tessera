"""Tests for the drift adapter module."""
import os
import tempfile
import numpy as np
import pytest
from tessera.drift_adapter import DriftAdapter


class TestDriftAdapter:
    def test_train_returns_rotation_matrix(self):
        adapter = DriftAdapter(64, 64)
        old_emb = np.random.randn(50, 64).astype(np.float32)
        new_emb = np.random.randn(50, 64).astype(np.float32)
        matrix = adapter.train(old_emb, new_emb)
        assert matrix.shape == (64, 64)

    def test_adapt_query_transforms_embedding(self):
        adapter = DriftAdapter(64, 64)
        old_emb = np.random.randn(50, 64).astype(np.float32)
        new_emb = np.random.randn(50, 64).astype(np.float32)
        adapter.train(old_emb, new_emb)
        query = np.random.randn(64).astype(np.float32)
        adapted = adapter.adapt_query(query)
        assert adapted.shape == (64,)

    def test_adapt_query_without_training_raises(self):
        adapter = DriftAdapter(64, 64)
        query = np.random.randn(64).astype(np.float32)
        with pytest.raises(ValueError):
            adapter.adapt_query(query)

    def test_save_and_load(self):
        adapter = DriftAdapter(64, 64)
        old_emb = np.random.randn(50, 64).astype(np.float32)
        new_emb = np.random.randn(50, 64).astype(np.float32)
        adapter.train(old_emb, new_emb)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            adapter.save(path)
            loaded = DriftAdapter.load(path)
            assert loaded.rotation_matrix is not None
            assert loaded.rotation_matrix.shape == (64, 64)
            np.testing.assert_array_almost_equal(
                adapter.rotation_matrix, loaded.rotation_matrix, decimal=5
            )
        finally:
            os.unlink(path)

    def test_load_rejects_non_square_matrix(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        np.save(path, np.zeros((10, 20)))
        try:
            with pytest.raises(ValueError, match="square"):
                DriftAdapter.load(path)
        finally:
            os.unlink(path)

    def test_load_rejects_bad_dtype(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        np.save(path, np.zeros((10, 10), dtype=np.int32))
        try:
            with pytest.raises(ValueError, match="float"):
                DriftAdapter.load(path)
        finally:
            os.unlink(path)

    def test_load_uses_allow_pickle_false(self):
        """Security: ensure allow_pickle=False is used."""
        # Create a file that would need pickle
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        np.save(path, np.zeros((10, 10), dtype=np.float32))
        try:
            # This should work (no pickle needed)
            loaded = DriftAdapter.load(path)
            assert loaded.rotation_matrix is not None
        finally:
            os.unlink(path)
