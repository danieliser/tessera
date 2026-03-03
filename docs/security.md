# Security & Access Control in Tessera

Tessera is built on a **deny-by-default scope gating model**. Agents only see what they're explicitly granted access to. This is essential for multi-agent workflows where orchestrators delegate to sub-agents without leaking access to sensitive projects.

This guide explains how the access control system works, how to configure it, and practical workflows for securing multi-agent setups.

## Overview

Tessera's security model has three layers:

1. **Session tokens** — Orchestrators create scoped tokens and pass them to sub-agents
2. **Project/collection scope** — Tokens specify which projects an agent can access
3. **File-level filtering** — `.tesseraignore` blocks sensitive files (credentials, keys) from indexing

No layer can be bypassed. An agent without a valid session token gets no access. An agent with a session token can only search projects in their scope. Even if a project is in scope, credentials and secrets are stripped from indexing.

---

## Deny-by-Default Model

### What "Deny-by-Default" Means

Without a session token, an agent gets **no access** to search, navigate, or inspect any project.

**Development mode exception:** When running Tessera locally without authentication configured, tools work without session tokens. This is for local development and testing only. Production always requires valid tokens.

### How It Works

1. **Agent calls a tool** — e.g., `search(query="login handler")`
2. **Tessera resolves the session token** — checked in order:
     1. Explicit `session_id` parameter on the tool call
     2. `TESSERA_SESSION_ID` environment variable (set once at process startup)
     3. Neither → dev mode (no scoping)
3. **If validation fails**, the tool returns an error:
   - Token not found → "Error: Invalid session"
   - Token expired → "Error: Session expired"
   - Insufficient scope → "Error: Insufficient scope. Required: global, have: project"
4. **If validation succeeds**, the tool runs and filters results to the agent's authorized projects

### Development Mode

When no session tokens are provided (`session_id=""`), Tessera enters **development mode**. The `_check_session()` function returns `(None, None)` — no error, but no scope enforcement either.

This is useful for:
- Local development and testing
- Single-developer setups
- Interactive debugging

**Do not rely on development mode in production.** Always provide session tokens when running Tessera as a multi-agent server.

---

## Session Tokens

A session token is a UUID4 string that grants an agent access to specific projects for a limited time.

### Token Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ Orchestrator                                            │
│ call: create_scope_tool(agent_id="task-1", ...)        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ (returns session_id)
┌─────────────────────────────────────────────────────────┐
│ Sessions Table (SQLite)                                 │
│ session_id | agent_id | level | projects | valid_until │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ (agent uses token)
┌─────────────────────────────────────────────────────────┐
│ Sub-Agent (Task Agent)                                  │
│ call: search(..., session_id="...")                     │
│ Results filtered to authorized projects                 │
└─────────────────────────────────────────────────────────┘
```

### Creating a Token

Use the `create_scope_tool` MCP tool. **Only the orchestrator (with global scope) can create tokens.**

```python
result = create_scope_tool(
    agent_id="task-agent-1",
    scope_level="project",
    project_ids=[1, 2],          # Agent can search projects 1 and 2
    ttl_minutes=30               # Token expires in 30 minutes
)
# Returns:
# {
#   "session_id": "550e8400-e29b-41d4-a716-446655440000",
#   "agent_id": "task-agent-1",
#   "scope_level": "project",
#   "ttl_minutes": 30
# }
```

Pass the session_id to the sub-agent via MCP's `initialize` message:

```json
{
  "protocolVersion": "2024-11-05",
  "capabilities": {},
  "clientInfo": {
    "name": "task-agent-1",
    "version": "1.0.0"
  },
  "initializationOptions": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

### Token Expiration

Tokens have a **time-to-live (TTL)** in minutes. Default: 30 minutes.

Once expired, `validate_session()` raises `SessionExpiredError`, and the agent's tools stop working.

```python
# Short-lived token for temporary access (good for one-off tasks)
create_scope_tool(
    agent_id="disposable-task",
    scope_level="project",
    project_ids=[5],
    ttl_minutes=5           # Expires in 5 minutes
)

# Long-lived token for persistent agents (still expires for hygiene)
create_scope_tool(
    agent_id="background-worker",
    scope_level="project",
    project_ids=[1, 2, 3],
    ttl_minutes=480         # Expires in 8 hours
)
```

Automatic cleanup of expired sessions happens periodically. The `cleanup_expired_sessions()` function removes all tokens with `valid_until < now`.

---

## Scope Levels

A token's **scope level** determines what the agent can see.

### Project Scope

Access to specific projects. The agent sees **only** those projects.

```python
# Agent can search projects 1, 2, 3. Cannot see project 4.
create_scope_tool(
    agent_id="task-1",
    scope_level="project",
    project_ids=[1, 2, 3]
)
```

When the agent calls `search()`, results are filtered to those 3 projects. Cross-project searches won't include project 4.

**Deny-by-default:** If the agent requests a project outside their scope, the tool returns no results for that project.

### Collection Scope

Access to all projects in one or more collections.

```python
# Agent can access all projects in collection 1
create_scope_tool(
    agent_id="task-2",
    scope_level="collection",
    collection_ids=[1]
)

# Agent can access all projects in collections 1 AND 2
create_scope_tool(
    agent_id="task-3",
    scope_level="collection",
    collection_ids=[1, 2]
)
```

Internally, Tessera resolves collection IDs to their project memberships. The scope check is identical to project scope, but decoupled from project IDs — making it easier to add/remove projects from a collection without updating agent tokens.

**Use case:** Plugin ecosystem. Collections can represent different plugins, sub-teams, or customer accounts. Token creation doesn't hardcode project IDs.

### Global Scope

Access to **all** projects.

```python
# Admin agent can see everything
create_scope_tool(
    agent_id="orchestrator",
    scope_level="global",
    project_ids=[]      # Ignored; global implies all projects
)
```

**Restrict this carefully.** Only the orchestrator (or trusted admin agents) should have global scope.

### Scope Hierarchy

Scope levels are strictly ordered:

```
project < collection < global
```

- A global-scoped token can create project-scoped tokens (downgrade).
- A project-scoped token **cannot** create any token (insufficient scope to call `create_scope_tool`).

The `_check_session()` function enforces this:

```python
# In _state.py
if _SCOPE_LEVELS.get(scope.level, -1) < _SCOPE_LEVELS.get(required_level, 0):
    return None, f"Error: Insufficient scope. Required: {required_level}, have: {scope.level}"
```

---

## Revoking Access

Revoke all tokens for an agent immediately using `revoke_scope_tool`.

```python
# Orchestrator revokes all sessions for task-agent-1
result = revoke_scope_tool(agent_id="task-agent-1")
# Returns:
# {
#   "agent_id": "task-agent-1",
#   "sessions_revoked": 2    # Had 2 active tokens; both deleted
# }
```

The agent's tools immediately stop working. Any in-flight requests fail with "Error: Invalid session."

**Use cases:**
- Task completed; agent should no longer have access.
- Agent compromised; cut off immediately.
- Switching orchestrators or credentials.

**Note:** Revocation is instant but asynchronous for distributed agents. If an agent has already fetched results but not consumed them, those results are still available locally. Only new tool calls are blocked.

---

## Path Traversal Protection

Tessera blocks path traversal attacks that attempt to escape the project root.

### How It Works

When indexing or reading files, Tessera normalizes and validates paths:

```python
# From auth.py
def normalize_and_validate_path(project_root: str, user_path: str) -> str:
    """
    1. Resolve both paths to absolute
    2. Check that resolved user_path starts with resolved project_root
    3. Raise PathTraversalError if not
    4. Return the validated absolute path
    """
```

### Examples

**Valid paths:**

```python
normalize_and_validate_path('/projects/pm', 'src/Hooks.php')
# → '/projects/pm/src/Hooks.php'

normalize_and_validate_path('/projects/pm', '/projects/pm/src/Hooks.php')
# → '/projects/pm/src/Hooks.php'
```

**Blocked paths:**

```python
normalize_and_validate_path('/projects/pm', '../../etc/passwd')
# → PathTraversalError: Path escapes project root

normalize_and_validate_path('/projects/pm', '/etc/passwd')
# → PathTraversalError: Path escapes project root
```

### Symlink Handling

Symlinks are resolved to their target before validation. This prevents symlink-based escapes:

```python
# /projects/pm/link → /outside (symlink)
normalize_and_validate_path('/projects/pm', 'link/../../../etc/passwd')
# Symlink is resolved; target is outside project root
# → PathTraversalError
```

### Why This Matters

In multi-project setups, Tessera operates on multiple project roots. Path traversal could leak files from one project to another, or access system files outside all projects. Strict validation prevents this.

---

## File-Level Access Control: `.tesseraignore`

Tessera blocks sensitive files from indexing using a **two-tier ignore system**:

1. **Security tier** — Un-negatable patterns for credentials, keys, and secrets
2. **Default tier** — Common build artifacts and caches (can be overridden)

### Security Patterns (Un-Negatable)

These patterns **cannot be overridden**. Attempting to negate them (with `!` in `.tesseraignore`) is logged as a warning and ignored.

```
.env*
*.pem
*.key
*.p12
*.pfx
*credentials*
*secret*
id_rsa
id_ed25519
*.token
service-account.json
```

**Why un-negatable?** Security patterns protect against accidental leaks. Even if a developer adds `!.env` to their `.tesseraignore`, Tessera still blocks it.

**Example:**

```bash
# Project root: /home/alice/webapp

# Files in the project:
# .env                    (blocked by security pattern .env*)
# .env.local              (blocked by security pattern .env*)
# config/secrets.json     (blocked by security pattern *secret*)
# certs/server.pem        (blocked by security pattern *.pem)
# src/main.py             (indexed)
```

### Default Patterns (Negatable)

These patterns are ignored by default but can be overridden in `.tesseraignore`:

```
.git/
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.egg-info/
dist/
build/
node_modules/
vendor/
composer.lock
.next/
.turbo/
.vscode/
.idea/
*.swp
*.swo
.DS_Store
coverage/
.nyc_output/
.tessera/
*.log
.gitignore
```

**Override example:**

```bash
# Project root contains node_modules with important type definitions
# But .tesseraignore excludes node_modules by default

# File: .tesseraignore
!node_modules/@mycompany/shared-types/

# Now node_modules/@mycompany/shared-types/ is indexed
# But node_modules/lodash/, node_modules/react/, etc. are still excluded
```

### Adding Custom Patterns

Use `.tesseraignore` (same syntax as `.gitignore`) to customize:

```bash
# File: /home/alice/webapp/.tesseraignore

# Ignore temporary test files
__test_tmp__/
*.tmp.js

# Ignore generated code (already in git, but expensive to index)
generated/

# Ignore archived data
archives/

# Override defaults: index our vendored dependencies
!vendor/
!vendor/mylib/

# Attempt to negate a security pattern (will be logged as warning, ignored)
# !.env.backup        ← This is ignored; .env* is always excluded
```

### Checking What's Indexed

The `IgnoreFilter` class determines whether each file is indexed:

```python
from tessera.ignore import IgnoreFilter

ignore = IgnoreFilter('/home/alice/webapp')

# Check individual files
ignore.should_ignore('.env')                      # True (security pattern)
ignore.should_ignore('node_modules/lodash/index.js')  # True (default pattern)
ignore.should_ignore('src/main.py')               # False (allowed)
```

---

## Passing Tokens to Sub-Agents

The orchestrator creates a scoped token via `create_scope_tool`, then passes it to sub-agents. How the token reaches the sub-agent depends on your setup.

### `TESSERA_SESSION_ID` Environment Variable

Tessera reads `TESSERA_SESSION_ID` from the environment at startup. When set, every tool call from that process is automatically scoped — agents don't need to know about session tokens at all.

**Precedence:** explicit `session_id` tool parameter > `TESSERA_SESSION_ID` env var > dev mode (no scoping).

### Method 1: CLI Orchestrator (Inline Env Var)

The simplest approach. The orchestrator spawns a sub-agent CLI process with the token as an inline environment variable:

```bash
TESSERA_SESSION_ID=550e8400-e29b-41d4-a716-446655440000 \
  claude "refactor the auth module"
```

The env var propagates to the `claude` process, which propagates it to the Tessera MCP subprocess. Every tool call is scoped automatically.

**Shared scope for all sub-agents:** Export once, every child inherits:

```bash
export TESSERA_SESSION_ID=550e8400-e29b-41d4-a716-446655440000
claude "refactor auth module"
claude "update tests for auth"
claude "review auth changes"
# All three agents share the same scoped access
```

**Per-agent scope:** Inline env var overrides the parent:

```bash
export TESSERA_SESSION_ID=global-orchestrator-token

# This agent gets a different, narrower scope
TESSERA_SESSION_ID=project-scoped-token-for-auth \
  claude "refactor the auth module"

# This agent gets the parent's global scope
claude "check cross-project dependencies"
```

### Method 2: MCP Config with Baked-In Token

The orchestrator writes a temporary `.mcp.json` with the token in the env block, then passes it to the sub-agent via `--mcp-config`:

```python
import json, tempfile

# Orchestrator creates a scoped token
token = create_scope_tool(
    agent_id="auth-task",
    scope_level="project",
    project_ids=[1],
    ttl_minutes=30
)

# Write temp MCP config with token baked in
config = {
    "mcpServers": {
        "tessera": {
            "command": "uv",
            "args": ["--directory", "/path/to/tessera",
                     "run", "python", "-m", "tessera", "serve"],
            "env": {"TESSERA_SESSION_ID": token["session_id"]}
        }
    }
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(config, f)
    config_path = f.name

# Spawn sub-agent with scoped config
subprocess.run(["claude", "--mcp-config", config_path, "refactor auth"])
```

This is ideal when different sub-agents need different MCP server configurations (not just different tokens).

### Method 3: API-Driven (Tool Injection)

For non-CLI orchestrators that build prompts and inject tools programmatically, patch the token directly into tool call parameters at the application layer:

```python
# Orchestrator creates a scoped token
token = create_scope_tool(
    agent_id="task-agent",
    scope_level="project",
    project_ids=[1, 2]
)

# During prompt building / tool injection, the orchestrator
# patches session_id into every Tessera tool definition
for tool in tessera_tools:
    tool["parameters"]["session_id"] = token["session_id"]

# Sub-agent calls search(query="...") — the session_id is
# already baked in. The agent never knows it's scoped.
```

This is the cleanest path for orchestrators that already manage tool definitions (e.g., the persistence layer during prompt construction).

---

## Practical Workflows

### Workflow 1: CLI Orchestrator with Task Agents

**Setup:**
- Orchestrator has global scope
- Task agents get project-scoped tokens via inline env vars

```python
# Orchestrator creates scoped tokens for each task
auth_token = create_scope_tool(
    agent_id="auth-task",
    scope_level="project",
    project_ids=[1],        # Only auth-service
    ttl_minutes=30
)

api_token = create_scope_tool(
    agent_id="api-task",
    scope_level="project",
    project_ids=[2, 3],     # api-gateway + web-ui
    ttl_minutes=30
)
```

```bash
# Spawn sub-agents with scoped access
TESSERA_SESSION_ID=<auth_token> claude "fix the password validation bug"
TESSERA_SESSION_ID=<api_token> claude "add rate limiting to the API"

# Each agent can only search their authorized projects
# auth-task sees auth-service only
# api-task sees api-gateway + web-ui only
```

After task completion:
```python
revoke_scope_tool(agent_id="auth-task")
revoke_scope_tool(agent_id="api-task")
```

### Workflow 2: Collection-Based Customer Isolation

```python
# Group projects by customer
collection_a = create_collection_tool(name="customer-a")
for proj_id in [1, 2, 3, 4, 5]:
    add_to_collection_tool(collection_id=collection_a["id"], project_id=proj_id)

# Create collection-scoped token
token_a = create_scope_tool(
    agent_id="customer-a-agent",
    scope_level="collection",
    collection_ids=[collection_a["id"]],
    ttl_minutes=480
)
```

```bash
# Agent sees all 5 customer-a projects, nothing else
TESSERA_SESSION_ID=<token_a> claude "refactor the database layer"
```

### Workflow 3: Securing Credentials During Indexing

```bash
# Project structure
/projects/myapp/
├── .env                      # Blocked by security pattern .env*
├── .env.production           # Blocked by security pattern .env*
├── config/
│   ├── secrets.json          # Blocked by security pattern *secret*
│   └── app.yaml              # Indexed
├── .ssh/
│   └── deploy_key.pem        # Blocked by security pattern *.pem
├── src/
│   └── main.py               # Indexed
├── node_modules/             # Blocked by default pattern
├── __pycache__/              # Blocked by default pattern
└── .tesseraignore
    # Custom exclusions
    temp_backups/
    *.tmp.sql
    # Security patterns cannot be negated:
    # !.env    ← This line is logged as warning and ignored
```

---

## Session Storage & Lifecycle

### Where Sessions Are Stored

Sessions live in the **global database** (`~/.tessera/global.db` or configured path):

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    level TEXT NOT NULL,                    -- 'project', 'collection', or 'global'
    projects_list TEXT,                     -- JSON array of project IDs
    collections_list TEXT,                  -- JSON array of collection IDs
    capabilities TEXT,                      -- JSON array (reserved for future)
    created_at TEXT NOT NULL,               -- ISO 8601 timestamp
    valid_until TEXT NOT NULL               -- ISO 8601 timestamp
);
```

### Lifecycle Events

1. **Creation** — `create_scope()` inserts row, returns session_id
2. **Validation** — `validate_session()` checks existence and expiration
3. **Use** — Tools filter results using `check_scope()`
4. **Expiration** — `cleanup_expired_sessions()` deletes rows with `valid_until < now`
5. **Revocation** — `revoke_scope()` deletes rows by agent_id

### Session Cleanup

Expired sessions are automatically deleted by the background task (runs periodically) or manually via `cleanup_expired_sessions()`.

```python
# Manual cleanup
count = cleanup_expired_sessions(db_conn)
print(f"Deleted {count} expired sessions")
```

---

## Common Mistakes & Pitfalls

### ❌ Mistake 1: Not Setting the Session Token for Sub-Agents

**Problem:**
```bash
# Orchestrator creates a token but doesn't pass it to the sub-agent
claude "search the codebase for auth bugs"
# In dev mode: works (no scoping)
# In production: agent has no access
```

**Solution:** Use `TESSERA_SESSION_ID` environment variable:
```bash
TESSERA_SESSION_ID=550e8400-e29b-41d4-a716-446655440000 \
  claude "search the codebase for auth bugs"
```

Or bake it into the MCP config's `env` block (see [Passing Tokens to Sub-Agents](#passing-tokens-to-sub-agents)).

### ❌ Mistake 2: Attempting to Negate Security Patterns

**Problem:**
```bash
# File: .tesseraignore
!.env          # Trying to index .env file
!*.pem         # Trying to index PEM certificates
```

**What happens:**
- Tessera logs a warning
- Security patterns are still enforced
- `.env` and `*.pem` remain blocked

**Solution:** Don't try to negate security patterns. Use environment variables or separate config files outside the project.

### ❌ Mistake 3: Hardcoding Project IDs

**Problem:**
```python
# Creating tokens with hardcoded IDs
create_scope_tool(
    agent_id="task-1",
    scope_level="project",
    project_ids=[1, 2, 3]  # ← Brittle if projects are re-registered
)
```

**Solution:** Use collections for groups of projects:
```python
# Create collection once
collection = create_collection_tool(name="customer-a")

# Add projects to collection (can change without updating tokens)
add_to_collection_tool(collection_id=collection["id"], project_id=project_id)

# Create token for collection (future-proof)
create_scope_tool(
    agent_id="task-1",
    scope_level="collection",
    collection_ids=[collection["id"]]
)
```

### ❌ Mistake 4: Not Revoking Expired Tokens

**Problem:**
```python
# Create a 1-hour token for a task
token = create_scope_tool(agent_id="task-1", ttl_minutes=60)

# Task completes in 5 minutes
# But token is still valid for 55 more minutes
# Sub-agent retains access even though the job is done
```

**Solution:** Revoke manually when task completes:
```python
revoke_scope_tool(agent_id="task-1")  # Immediate revocation
```

Or use very short TTLs for one-off tasks:
```python
token = create_scope_tool(agent_id="task-1", ttl_minutes=5)
```

---

## Debugging & Auditing

### Check Audit Logs

Every tool call is logged to Tessera's audit log (both Python logger and GlobalDB):

```python
# From _state.py
_log_audit(
    tool_name="search",
    result_count=42,
    agent_id="task-1",
    scope_level="project",
    ppr_used=False
)
```

Audit logs include:
- Agent ID
- Scope level
- Tool called
- Result count
- Timestamp

**Use case:** Verify which agents accessed which projects, and when.

### Check Active Sessions

Query the sessions table directly:

```python
from tessera.db import GlobalDB

db = GlobalDB('~/.tessera/global.db')
sessions = db.conn.execute(
    "SELECT agent_id, level, projects_list, valid_until FROM sessions"
).fetchall()

for agent_id, level, projects, valid_until in sessions:
    print(f"{agent_id} ({level}): {projects} until {valid_until}")
```

### Test Scope Validation

Use `check_scope()` to verify what an agent can see:

```python
from tessera.auth import validate_session, check_scope

scope = validate_session(db_conn, session_id)

# Check if agent can access project 5
can_access = check_scope(scope, "5")
print(f"Agent can access project 5: {can_access}")
```

---

## Best Practices

1. **Least privilege** — Create the most restrictive token possible for each agent
2. **Short TTLs** — Use 30-60 minutes for task agents; longer for persistent workers
3. **Revoke on completion** — Don't rely on token expiration; revoke immediately
4. **Use collections** — Group related projects; avoid hardcoding IDs
5. **Monitor audit logs** — Log who accessed what, and when
6. **Override `.tesseraignore` carefully** — Don't negate security patterns; add project-specific exclusions only
7. **Test in dev mode first** — Verify scope logic locally before enabling production auth

---

## API Reference

### `create_scope_tool()`

```python
create_scope_tool(
    agent_id: str,                       # Unique agent identifier
    scope_level: str,                    # 'project', 'collection', or 'global'
    project_ids: list[int] = None,       # Project IDs (for project scope)
    collection_ids: list[int] = None,    # Collection IDs (for collection scope)
    ttl_minutes: int = 30,               # Token lifetime in minutes
    session_id: str = ""                 # Orchestrator's session_id (for auth)
) -> str
```

**Returns:** JSON with `session_id`, `agent_id`, `scope_level`, `ttl_minutes`

**Requires:** Global scope (or dev mode)

### `revoke_scope_tool()`

```python
revoke_scope_tool(
    agent_id: str,                       # Agent to revoke
    session_id: str = ""                 # Orchestrator's session_id (for auth)
) -> str
```

**Returns:** JSON with `agent_id`, `sessions_revoked` (count)

**Requires:** Global scope (or dev mode)

### `validate_session()`

```python
from tessera.auth import validate_session

scope = validate_session(db_conn, session_id)
# Returns: ScopeInfo(
#     session_id=...,
#     agent_id=...,
#     level=...,
#     projects=[...],
#     collections=[...],
#     capabilities=[...]
# )
```

**Raises:**
- `SessionNotFoundError` — Session ID not found
- `SessionExpiredError` — Token has expired

### `check_scope()`

```python
from tessera.auth import check_scope

allowed = check_scope(scope, project_id="5")
# Returns: True if project_id is in scope, False otherwise
```

---

## References

- **Auth module:** `src/tessera/auth.py` — Token creation, validation, revocation
- **Ignore filter:** `src/tessera/ignore.py` — Two-tier security patterns
- **Server state:** `src/tessera/server/_state.py` — Session validation at request time
- **Tests:** `tests/unit/test_auth.py`, `tests/unit/test_ignore.py` — Comprehensive examples

