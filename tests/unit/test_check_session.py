"""Tests for _check_session env var fallback."""

import os
from unittest.mock import patch

from tessera.auth import ScopeInfo
from tessera.server._state import _check_session


def _make_scope(**overrides):
    defaults = dict(
        session_id="test-session",
        agent_id="test-agent",
        level="project",
        projects=["1"],
        collections=[],
        capabilities=["search"],
    )
    defaults.update(overrides)
    return ScopeInfo(**defaults)


class TestCheckSessionEnvVar:
    """Verify _check_session reads TESSERA_SESSION_ID from environment."""

    def test_env_var_used_when_no_argument(self):
        """When session_id not in arguments, falls back to TESSERA_SESSION_ID env var."""
        mock_scope = _make_scope()
        with patch.dict(os.environ, {"TESSERA_SESSION_ID": "env-token-123"}), \
             patch("tessera.server._state._global_db") as mock_gdb, \
             patch("tessera.server._state.validate_session", return_value=mock_scope):
            mock_gdb.__bool__ = lambda self: True
            scope, err = _check_session({})
            assert scope is not None
            assert scope.level == "project"
            assert err is None

    def test_argument_takes_precedence_over_env(self):
        """Explicit session_id argument wins over env var."""
        mock_scope = _make_scope()
        with patch.dict(os.environ, {"TESSERA_SESSION_ID": "env-token"}), \
             patch("tessera.server._state._global_db") as mock_gdb, \
             patch("tessera.server._state.validate_session", return_value=mock_scope) as mock_validate:
            mock_gdb.__bool__ = lambda self: True
            _check_session({"session_id": "arg-token"})
            mock_validate.assert_called_once()
            assert mock_validate.call_args[0][1] == "arg-token"

    def test_dev_mode_when_neither_arg_nor_env(self):
        """No session_id anywhere → dev mode (None, None)."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TESSERA_SESSION_ID", None)
            scope, err = _check_session({})
            assert scope is None
            assert err is None
