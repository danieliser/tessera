"""CLI entry point for CodeMem.

Supports:
  - indexing: python -m tessera index <path> --embedding-endpoint <url>
  - serving: python -m tessera serve --project <path>
"""

import argparse
import asyncio
import logging
import sys
import time

from .server import run_server


def _run_index(args) -> int:
    """Run the indexing pipeline."""
    from .embeddings import create_embedding_client
    from .indexer import IndexerPipeline

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_path = args.path

    embedding_client = create_embedding_client(
        provider=args.embedding_provider,
        embedding_endpoint=args.embedding_endpoint,
        embedding_model=args.embedding_model,
    )
    if embedding_client:
        print(f"Embeddings: {type(embedding_client).__name__}")
    else:
        print("No embeddings — indexing without semantic vectors.")

    pipeline = IndexerPipeline(
        project_path=project_path,
        embedding_client=embedding_client,
    )
    pipeline.register()

    start = time.perf_counter()

    if args.incremental:
        print(f"Incremental index: {project_path}")
        stats = pipeline.index_changed()
    else:
        print(f"Full index: {project_path}")
        stats = pipeline.index_project()

    elapsed = time.perf_counter() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Files: {stats.files_processed} indexed, {stats.files_skipped} skipped, {stats.files_failed} failed")
    print(f"  Symbols: {stats.symbols_extracted}")
    print(f"  Chunks: {stats.chunks_created} ({stats.chunks_embedded} embedded)")

    return 0 if stats.files_failed == 0 else 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tessera",
        description="CodeMem: Hierarchical codebase indexing and memory system"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a project")
    index_parser.add_argument("path", help="Path to project to index")
    index_parser.add_argument(
        "--embedding-provider",
        choices=["auto", "http", "fastembed"],
        default="auto",
        help="Embedding provider: auto (fastembed if installed, else HTTP), http, or fastembed"
    )
    index_parser.add_argument(
        "--embedding-endpoint",
        default=None,
        help="OpenAI-compatible embedding endpoint URL (used when provider is http or auto)"
    )
    index_parser.add_argument(
        "--embedding-model",
        default=None,
        help="Model identifier (default varies by provider)"
    )
    index_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only re-index files changed since last indexed commit"
    )
    index_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the MCP server")
    serve_parser.add_argument(
        "--project",
        required=False,
        default=None,
        help="Path to project directory (locks server to this project; omit for multi-project mode)"
    )
    serve_parser.add_argument(
        "--global-db",
        help="Path to global.db (optional)"
    )
    serve_parser.add_argument(
        "--embedding-provider",
        choices=["auto", "http", "fastembed"],
        default="auto",
        help="Embedding provider: auto (fastembed if installed, else HTTP), http, or fastembed"
    )
    serve_parser.add_argument(
        "--embedding-endpoint",
        default=None,
        help="OpenAI-compatible embedding endpoint URL (e.g. http://localhost:8800/v1/embeddings)"
    )
    serve_parser.add_argument(
        "--embedding-model",
        default=None,
        help="Model identifier (default varies by provider)"
    )
    serve_parser.add_argument(
        "--reranking-model",
        default=None,
        help="Cross-encoder reranking model (default: jinaai/jina-reranker-v2-base-multilingual)"
    )
    serve_parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable cross-encoder reranking"
    )

    args = parser.parse_args()

    if args.command == "index":
        return _run_index(args)
    elif args.command == "serve":
        return asyncio.run(run_server(
            args.project, args.global_db,
            args.embedding_endpoint, args.embedding_model,
            args.embedding_provider, args.reranking_model,
            args.no_reranking,
        ))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
