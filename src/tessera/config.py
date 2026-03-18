"""Tessera configuration file support.

Loads settings from ~/.tessera/config.toml with sensible defaults.
Config values are overridden by environment variables and CLI flags.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_config_cache: dict | None = None

CONFIG_DIR = Path.home() / ".tessera"
CONFIG_PATH = CONFIG_DIR / "config.toml"


def load_config() -> dict:
    """Load config from ~/.tessera/config.toml.

    Returns empty dict if file doesn't exist or can't be parsed.
    Caches after first load.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    _config_cache = {}

    if not CONFIG_PATH.exists():
        return _config_cache

    try:
        # Python 3.11+ has tomllib in stdlib
        import tomllib
        with open(CONFIG_PATH, "rb") as f:
            _config_cache = tomllib.load(f)
        logger.debug("Loaded config from %s", CONFIG_PATH)
    except Exception as e:
        logger.warning("Failed to load %s: %s", CONFIG_PATH, e)
        _config_cache = {}

    return _config_cache


def get_nice_value() -> int | None:
    """Resolve nice value from env var → config file.

    Used by MCP server reindex (no CLI flag available).
    Returns None if not configured or explicitly 0.
    """
    # Env var takes precedence
    env_nice = os.environ.get("TESSERA_NICE")
    if env_nice is not None:
        try:
            val = int(env_nice)
            return val if val > 0 else None
        except ValueError:
            pass

    # Config file
    config = load_config()
    config_nice = config.get("nice")
    if config_nice is not None:
        try:
            val = int(config_nice)
            return val if val > 0 else None
        except (ValueError, TypeError):
            pass

    return None
