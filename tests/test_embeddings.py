"""Tests for embeddings.py module."""

import pytest
from unittest.mock import Mock, patch
import httpx
from codemem.embeddings import EmbeddingClient, EmbeddingUnavailableError


class TestEmbeddingClientInit:
    """Test EmbeddingClient initialization."""

    def test_client_initialization(self):
        """Test basic client initialization."""
        client = EmbeddingClient(
            endpoint="http://localhost:8000/v1/embeddings",
            model="default"
        )
        assert client.endpoint == "http://localhost:8000/v1/embeddings"
        assert client.model == "default"
        client.close()

    def test_client_custom_timeout(self):
        """Test client with custom timeout."""
        client = EmbeddingClient(
            endpoint="http://localhost:8000/v1/embeddings",
            model="default",
            timeout=60.0
        )
        # Just verify client was created successfully
        assert client.endpoint == "http://localhost:8000/v1/embeddings"
        client.close()

    def test_client_default_endpoint(self):
        """Test client with default endpoint."""
        client = EmbeddingClient()
        assert client.endpoint == "http://localhost:8000/v1/embeddings"
        assert client.model == "default"
        client.close()


class TestEmbeddingClientCache:
    """Test embedding caching."""

    def test_cache_initialization(self):
        """Test that cache is initialized empty."""
        client = EmbeddingClient()
        assert len(client._cache) == 0
        client.close()

    @patch("httpx.Client.post")
    def test_cache_stores_embeddings(self, mock_post):
        """Test that embeddings are cached."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
            ]
        }
        mock_post.return_value = mock_response

        client = EmbeddingClient()
        result = client.embed(["hello"])

        assert len(client._cache) == 1
        assert "hello" in client._cache
        client.close()

    @patch("httpx.Client.post")
    def test_cache_reuses_embeddings(self, mock_post):
        """Test that cached embeddings are reused."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
            ]
        }
        mock_post.return_value = mock_response

        client = EmbeddingClient()

        # First call
        result1 = client.embed(["hello"])
        assert mock_post.call_count == 1

        # Second call with same text
        result2 = client.embed(["hello"])
        # Should still be 1 call (cached)
        assert mock_post.call_count == 1

        # Results should be identical
        assert result1 == result2
        client.close()


class TestEmbeddingSingleText:
    """Test single text embedding."""

    @patch("httpx.Client.post")
    def test_embed_single_text(self, mock_post):
        """Test embed_single convenience method."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
            ]
        }
        mock_post.return_value = mock_response

        client = EmbeddingClient()
        result = client.embed_single("hello")

        assert result == [0.1, 0.2, 0.3]
        client.close()


class TestEmbeddingBatching:
    """Test embedding batching."""

    @patch("httpx.Client.post")
    def test_embed_batch_under_limit(self, mock_post):
        """Test batch under 100 item limit."""
        texts = ["text1", "text2", "text3"]
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
                {"embedding": [0.5, 0.6]},
            ]
        }
        mock_post.return_value = mock_response

        client = EmbeddingClient()
        result = client.embed(texts)

        # Should make 1 API call
        assert mock_post.call_count == 1
        assert len(result) == 3
        client.close()

    @patch("httpx.Client.post")
    def test_embed_batch_over_limit(self, mock_post):
        """Test batch over 100 item limit (should split)."""
        texts = [f"text{i}" for i in range(150)]

        # Mock will be called twice (100 + 50)
        mock_response1 = Mock()
        mock_response1.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]} for _ in range(100)]
        }

        mock_response2 = Mock()
        mock_response2.json.return_value = {
            "data": [{"embedding": [0.3, 0.4]} for _ in range(50)]
        }

        mock_post.side_effect = [mock_response1, mock_response2]

        client = EmbeddingClient()
        result = client.embed(texts)

        # Should make 2 API calls
        assert mock_post.call_count == 2
        assert len(result) == 150
        client.close()


class TestEmbeddingErrorHandling:
    """Test error handling."""

    @patch("httpx.Client.post")
    def test_embed_request_error(self, mock_post):
        """Test handling of request errors."""
        mock_post.side_effect = httpx.RequestError("Connection failed")

        client = EmbeddingClient()

        with pytest.raises(EmbeddingUnavailableError):
            client.embed(["hello"])

        client.close()

    @patch("httpx.Client.post")
    def test_embed_http_error(self, mock_post):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error",
            request=Mock(),
            response=Mock()
        )
        mock_post.return_value = mock_response

        client = EmbeddingClient()

        with pytest.raises(EmbeddingUnavailableError):
            client.embed(["hello"])

        client.close()

    @patch("httpx.Client.post")
    def test_embed_invalid_response_missing_embedding(self, mock_post):
        """Test handling of invalid response format (missing embedding field)."""
        mock_response = Mock()
        # Response has data but missing 'embedding' field
        mock_response.json.return_value = {
            "data": [
                {"content": "hello"}  # Missing 'embedding' field
            ]
        }
        mock_post.return_value = mock_response

        client = EmbeddingClient()

        with pytest.raises(EmbeddingUnavailableError):
            client.embed(["hello"])

        client.close()


class TestEmbeddingContextManager:
    """Test context manager interface."""

    def test_context_manager_enter_exit(self):
        """Test client works as context manager."""
        with EmbeddingClient() as client:
            assert client is not None
            assert client.endpoint == "http://localhost:8000/v1/embeddings"


class TestEmbeddingEmpty:
    """Test handling of empty inputs."""

    def test_embed_empty_list(self):
        """Test embedding empty list."""
        client = EmbeddingClient()
        result = client.embed([])
        assert result == []
        client.close()
