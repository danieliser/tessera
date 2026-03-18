"""CLI entry point for CodeMem.

Supports:
  - indexing: python -m tessera index <path> --embedding-endpoint <url>
  - serving: python -m tessera serve --project <path>
"""

import argparse
import asyncio
import logging
import os
import sys
import time

from .config import load_config
from .server import run_server


def apply_cpu_priority(nice_value: int | None) -> None:
    """Lower process CPU priority via os.nice().

    Note: os.nice() is additive — it increments the current nice value, not
    sets it absolutely. For CLI usage this is fine (processes start at 0).

    Args:
        nice_value: Nice increment (1-19, clamped). Higher = lower priority.
                    None or 0 means no change.
    """
    if not nice_value:
        return
    nice_value = max(1, min(19, nice_value))
    try:
        actual = os.nice(nice_value)
        logging.getLogger(__name__).info(f"CPU priority: nice {actual}")
    except PermissionError:
        logging.getLogger(__name__).warning(
            f"Could not set nice value {nice_value} (permission denied). "
            "Running at normal priority."
        )
    except OSError as exc:
        logging.getLogger(__name__).warning(f"Could not set nice value: {exc}")


def resolve_nice_value(cli_value: int | None) -> int | None:
    """Resolve nice value from CLI flag → env var → config file → default (10).

    Returns None only if explicitly set to 0 (opt-out).
    """
    # CLI flag takes precedence
    if cli_value is not None:
        return cli_value if cli_value > 0 else None

    # Env var next
    env_nice = os.environ.get("TESSERA_NICE")
    if env_nice is not None:
        try:
            val = int(env_nice)
            return val if val > 0 else None
        except ValueError:
            logging.getLogger(__name__).warning(
                f"Ignoring invalid TESSERA_NICE={env_nice!r} (must be integer 1-19)"
            )

    # Config file next
    config = load_config()
    config_nice = config.get("nice")
    if config_nice is not None:
        return int(config_nice) if int(config_nice) > 0 else None

    # Default: nice 10 for CLI indexing
    return 10


def _run_index(args) -> int:
    """Run the indexing pipeline."""
    from .embeddings import create_embedding_client
    from .indexer import IndexerPipeline

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Apply CPU priority before any heavy work.
    # Precedence: --nice flag → TESSERA_NICE env → config file → default 10
    nice_value = resolve_nice_value(args.nice)
    apply_cpu_priority(nice_value)

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
        stats = pipeline.index_changed_sync()
    else:
        print(f"Full index: {project_path}")
        stats = pipeline.index_project_sync()

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
        "--nice",
        type=int,
        default=None,
        metavar="N",
        help="CPU scheduling priority (1-19, default: 10). "
             "19 = lowest priority. 0 = disable throttling. "
             "Also configurable via TESSERA_NICE env var or ~/.tessera/config.toml"
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
        "--reranking-endpoint",
        default=None,
        help="Cohere-compatible reranking endpoint URL (e.g. http://localhost:8800/v1/rerank)"
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
            args.reranking_endpoint, args.no_reranking,
        ))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
