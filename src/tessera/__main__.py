"""CLI entry point for CodeMem.

Supports:
  - indexing: python -m tessera index <path> --embedding-endpoint <url>
  - serving: python -m tessera serve --project <path>
"""

import argparse
import sys
from .server import run_server
import asyncio


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
        "--embedding-endpoint",
        required=True,
        help="OpenAI-compatible embedding endpoint URL"
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
        "--embedding-endpoint",
        default=None,
        help="OpenAI-compatible embedding endpoint URL (e.g. http://localhost:8800/v1/embeddings)"
    )
    serve_parser.add_argument(
        "--embedding-model",
        default=None,
        help="Model identifier for the embedding endpoint (e.g. nomic-embed)"
    )

    args = parser.parse_args()

    if args.command == "index":
        print(f"Index command stub: {args.path} with endpoint {args.embedding_endpoint}")
        return 0
    elif args.command == "serve":
        return asyncio.run(run_server(
            args.project, args.global_db,
            args.embedding_endpoint, args.embedding_model,
        ))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
