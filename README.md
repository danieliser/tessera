# CodeMem

Hierarchical, scope-gated codebase indexing and persistent memory system for always-on AI agents.

## Overview

CodeMem is an MCP server that provides search, symbol navigation, reference tracing, and impact analysis across multi-language, multi-repo projects.

## Architecture

- **Structural Index**: SQLite (symbols, references, graph edges)
- **Semantic Index**: LanceDB (code chunks with embeddings)
- **Graph Retrieval**: Tree-sitter AST parsing + SQLite adjacency tables
- **Embeddings**: Local OpenAI-compatible model endpoint
- **Interface**: MCP server

## Status

Phase 1: Architecture validation and prototyping.
