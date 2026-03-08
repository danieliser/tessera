# Paper 6: Scope-Gated Search Federation for Multi-Agent Code Intelligence

**Status:** IDEA
**Target venue:** Workshop at ASE 2026 or CCS 2026 (position paper, 4 pages)

## Abstract (Draft)

As AI coding agents gain access to multi-project workspaces through tool-use protocols like MCP (Model Context Protocol), a new security challenge emerges: how to provide broad search capabilities while preventing data leakage across project boundaries. An agent working on Project A should not inadvertently surface proprietary code from Project B, even when both projects share an index for cross-project dependency analysis.

We present a capability-based scope control model for federated code search with three access tiers: project (single codebase), collection (related projects), and global (all indexed content). Scopes are bound to MCP sessions at connection time and enforced at query time — data remains physically partitioned at the project level and is merged only within authorized scope boundaries.

Our design makes two contributions: (1) a formal model for capability-based access control in AI agent tool use, addressing the under-specified security properties of current MCP implementations; and (2) a search-time federation architecture that enables cross-project queries (e.g., finding all consumers of a shared library) without requiring a single unified index, preserving data isolation by default.

## Key Claims

1. **Agent tool access needs capability scoping** — current MCP implementations grant all-or-nothing access, which is insufficient for multi-tenant or multi-project environments.

2. **Search-time federation preserves data isolation** while enabling cross-project analysis — no shared index required.

3. **Three-tier scope model** (project/collection/global) maps naturally to organizational boundaries and agent trust levels.

## Required Literature Review

- [ ] MCP specification and security model
- [ ] Capability-based security (Dennis & Van Horn, 1966; Capsicum)
- [ ] Multi-tenant search architectures
- [ ] Information flow control in software systems
- [ ] AI agent safety and tool use security
- [ ] Federated information retrieval

## Data / Experiments Needed

- Formal specification of the scope model
- Security analysis: demonstrate isolation guarantees
- Performance evaluation: federation overhead vs. unified index
- Case study: multi-project WordPress ecosystem (PM + PM Pro + extensions)

## Risks

- Position papers have limited impact
- MCP is still evolving — specification may change
- May be too early to publish — need more real-world deployment data
- Overlap with emerging MCP security research from Anthropic and others
