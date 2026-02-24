"""Embedding client for local OpenAI-compatible model endpoint.

Phase 1: Client initialization and caching strategy.
Will implement:
  - OpenAI-compatible API client (httpx)
  - Batch embedding requests
  - Response caching (to avoid redundant embedding calls)
  - Error handling and retry logic for transient failures
  - Support for configurable embedding dimensions
"""

import httpx
from typing import Optional


class EmbeddingUnavailableError(Exception):
    """Raised when the embedding endpoint is unreachable."""
    pass


class EmbeddingClient:
    """Client for local OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1/embeddings",
        model: str = "default",
        timeout: float = 30.0,
    ):
        """
        Initialize embedding client.

        Args:
            endpoint: URL of OpenAI-compatible embedding endpoint
            model: Model identifier
            timeout: HTTP request timeout in seconds
        """
        self.endpoint = endpoint
        self.model = model
        self._client = httpx.Client(timeout=timeout)
        self._cache_max = 10000
        self._cache = {}  # Bounded text -> embedding cache (LRU eviction)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts via batch API.

        POST to endpoint with:
        {"input": texts, "model": self.model}

        Args:
            texts: List of texts to embed

        Returns:
            list of embedding vectors (list of floats)

        Raises:
            EmbeddingUnavailableError: if endpoint is down or unreachable
        """
        if not texts:
            return []

        # Split into batches of max 100
        batch_size = 100
        all_results = []

        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch = texts[batch_start:batch_end]

            # Separate cached and uncached texts
            batch_results = []
            texts_to_embed = []
            uncached_indices = []  # indices into batch_results needing API fill

            for i, text in enumerate(batch):
                if text in self._cache:
                    batch_results.append(self._cache[text])
                else:
                    uncached_indices.append(len(batch_results))
                    batch_results.append(None)  # Placeholder
                    texts_to_embed.append(text)

            # Request embeddings for uncached texts
            if texts_to_embed:
                try:
                    response = self._client.post(
                        self.endpoint,
                        json={
                            "input": texts_to_embed,
                            "model": self.model,
                        }
                    )
                    response.raise_for_status()
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    raise EmbeddingUnavailableError(
                        f"Failed to reach embedding endpoint {self.endpoint}: {e}"
                    ) from e

                try:
                    data = response.json()
                    embeddings_data = data.get("data", [])

                    if len(embeddings_data) != len(texts_to_embed):
                        raise EmbeddingUnavailableError(
                            f"Embedding count mismatch: sent {len(texts_to_embed)}, got {len(embeddings_data)}"
                        )

                    # Cache and fill in results by position (not text key)
                    for i, item in enumerate(embeddings_data):
                        embedding = item.get("embedding")
                        if embedding is None:
                            raise KeyError("embedding field missing")

                        text = texts_to_embed[i]
                        self._cache[text] = embedding
                        # Evict oldest entries if cache exceeds max size
                        if len(self._cache) > self._cache_max:
                            # Remove first (oldest) entry
                            oldest_key = next(iter(self._cache))
                            del self._cache[oldest_key]
                        batch_results[uncached_indices[i]] = embedding
                except (KeyError, ValueError, IndexError) as e:
                    raise EmbeddingUnavailableError(
                        f"Invalid response format from embedding endpoint: {e}"
                    ) from e

            all_results.extend(batch_results)

        return all_results

    def embed_single(self, text: str) -> list[float]:
        """
        Convenience: embed a single text.

        Args:
            text: Text to embed

        Returns:
            embedding vector as list of floats
        """
        return self.embed([text])[0]

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Test client initialization
    print("=== Embedding Client Initialization Test ===")

    client = EmbeddingClient(
        endpoint="http://localhost:8000/v1/embeddings",
        model="default"
    )

    print(f"Client initialized successfully")
    print(f"  Endpoint: {client.endpoint}")
    print(f"  Model: {client.model}")
    print(f"  Cache size: {len(client._cache)}")

    client.close()
    print("Client closed successfully")

    # Note: Not calling real endpoint (it's likely not running)
    print("\nNote: Real endpoint test skipped (endpoint likely not running)")
