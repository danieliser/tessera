"""Embedding providers: HTTP (OpenAI-compatible) and local (fastembed).

Provides duck-typed embedding clients and a cross-encoder reranker.
Use create_embedding_client() and create_reranker() factories for provider selection.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("tessera.embeddings")


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

            for _i, text in enumerate(batch):
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

    def embed_query(self, text: str) -> list[float]:
        """Embed text as a search query with retrieval prefix.

        Uses query prefix for instruct-embedding models that distinguish
        between query and document embeddings.
        """
        prompt = f"Represent this search query for retrieval: {text}"
        return self.embed_single(prompt)

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class FastembedClient:
    """Local embedding client using fastembed (ONNX Runtime, no PyTorch)."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info("Loading fastembed model: %s (first run downloads automatically)", self.model_name)
        self._model = TextEmbedding(model_name=self.model_name)
        self._cache: dict[str, list[float]] = {}
        self._cache_max = 10000

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local fastembed model."""
        if not texts:
            return []

        # Separate cached and uncached
        results: list[list[float] | None] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for text in texts:
            if text in self._cache:
                results.append(self._cache[text])
            else:
                uncached_indices.append(len(results))
                results.append(None)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = list(self._model.embed(uncached_texts))
            for i, emb in enumerate(embeddings):
                vec = emb.tolist()
                text = uncached_texts[i]
                self._cache[text] = vec
                if len(self._cache) > self._cache_max:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                results[uncached_indices[i]] = vec

        return results  # type: ignore[return-value]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed text as a search query with retrieval prefix."""
        prompt = f"Represent this search query for retrieval: {text}"
        return self.embed_single(prompt)

    def close(self):
        """No-op (fastembed has no connection to close)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class HTTPReranker:
    """Reranker client for Cohere-compatible /v1/rerank endpoint."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8800/v1/rerank",
        model: str = "jina-reranker",
        timeout: float = 30.0,
    ):
        self.endpoint = endpoint
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Rerank documents via Cohere-compatible API.

        Returns list of (original_index, score) sorted by descending score.
        """
        if not documents:
            return []

        try:
            response = self._client.post(
                self.endpoint,
                json={
                    "query": query,
                    "documents": documents,
                    "top_n": top_k,
                    "model": self.model,
                },
            )
            response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning("Reranker endpoint unavailable: %s", e)
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]

        data = response.json()
        results = data.get("results", [])
        return [(r["index"], float(r["relevance_score"])) for r in results]

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FastembedReranker:
    """Cross-encoder reranker using fastembed's TextCrossEncoder."""

    DEFAULT_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

    def __init__(self, model_name: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info("Loading reranker model: %s", self.model_name)
        self._model = TextCrossEncoder(model_name=self.model_name)

    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Returns list of (original_index, score) sorted by descending score.
        """
        if not documents:
            return []
        scores = list(self._model.rerank(query, documents))
        indexed = [(i, float(s)) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_embedding_client(
    provider: str = "auto",
    embedding_endpoint: str | None = None,
    embedding_model: str | None = None,
) -> EmbeddingClient | FastembedClient | None:
    """Create an embedding client based on provider selection.

    Provider logic:
        - "http": Use HTTP endpoint (returns None if no endpoint)
        - "fastembed": Use local fastembed (raises ImportError if not installed)
        - "auto": HTTP endpoint if provided, else fastembed if installed, else None
    """
    if provider == "http":
        if not embedding_endpoint:
            return None
        return EmbeddingClient(endpoint=embedding_endpoint, model=embedding_model or "default")

    if provider == "fastembed":
        try:
            return FastembedClient(model_name=embedding_model)
        except ImportError:
            raise ImportError(
                "fastembed not installed. Install with: pip install tessera-idx[embed]"
            ) from None

    # auto: try HTTP first (explicit config wins), then fastembed, then None
    if embedding_endpoint:
        logger.info("Using HTTP embedding endpoint: %s", embedding_endpoint)
        return EmbeddingClient(endpoint=embedding_endpoint, model=embedding_model or "default")

    try:
        client = FastembedClient(model_name=embedding_model)
        logger.info("Using local fastembed embeddings (%s)", client.model_name)
        return client
    except ImportError:
        logger.info("No embeddings available (install tessera-idx[embed] for local embeddings)")
        return None


def create_reranker(
    provider: str = "auto",
    reranking_model: str | None = None,
    reranking_endpoint: str | None = None,
    enabled: bool = True,
) -> HTTPReranker | FastembedReranker | None:
    """Create a reranker based on provider selection.

    Provider logic:
        - "http": Use HTTP endpoint (returns None if no endpoint)
        - "fastembed": Use local fastembed (returns None if not installed)
        - "auto": HTTP endpoint if provided, else fastembed if installed, else None
    """
    if not enabled:
        return None

    if provider == "http":
        if not reranking_endpoint:
            return None
        logger.info("Using HTTP reranker endpoint: %s", reranking_endpoint)
        return HTTPReranker(endpoint=reranking_endpoint, model=reranking_model or "jina-reranker")

    if provider == "fastembed":
        try:
            reranker = FastembedReranker(model_name=reranking_model)
            logger.info("Reranker loaded (%s)", reranker.model_name)
            return reranker
        except ImportError:
            logger.debug("fastembed not installed, reranking disabled")
            return None

    # auto: HTTP endpoint if provided, else fastembed, else None
    if reranking_endpoint:
        logger.info("Using HTTP reranker endpoint: %s", reranking_endpoint)
        return HTTPReranker(endpoint=reranking_endpoint, model=reranking_model or "jina-reranker")

    try:
        reranker = FastembedReranker(model_name=reranking_model)
        logger.info("Reranker loaded (%s)", reranker.model_name)
        return reranker
    except ImportError:
        logger.debug("fastembed not installed, reranking disabled")
        return None
    except Exception as e:
        logger.warning("Failed to load reranker: %s", e)
        return None
