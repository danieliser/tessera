"""Two-tier .tesseraignore system with security and user-configurable patterns.

Phase 4: Ignore filtering for incremental indexing.
Implements:
  - Security patterns (un-negatable, always enforced)
  - Default patterns (mergeable with user .tesseraignore)
  - Pathspec-based glob matching for both tiers
  - Warning logged if user attempts to negate security pattern
"""

import logging
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)


class IgnoreFilter:
    """Two-tier ignore pattern matcher: security patterns + user-configurable defaults.

    Security patterns are always enforced and cannot be negated.
    User patterns (from .tesseraignore) are merged with defaults.
    """

    SECURITY_PATTERNS = [
        ".env*",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "*credentials*",
        "*secret*",
        "id_rsa",
        "id_ed25519",
        "*.token",
        "service-account.json",
    ]

    DEFAULT_PATTERNS = [
        ".git/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        ".venv/",
        "venv/",
        ".egg-info/",
        "dist/",
        "build/",
        "node_modules/",
        "npm-debug.log",
        ".npm/",
        "vendor/",
        "composer.lock",
        ".next/",
        "out/",
        ".turbo/",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        ".DS_Store",
        ".tsc/",
        "coverage/",
        ".nyc_output/",
        ".tessera/",
        "*.log",
        ".gitignore",
    ]

    def __init__(self, project_root: str, ignore_file: str = ".tesseraignore"):
        """Initialize the ignore filter for a project.

        Args:
            project_root: Root directory of the project
            ignore_file: Name of the ignore file (default: .tesseraignore)
        """
        self.project_root = Path(project_root)
        self.ignore_file = ignore_file

        # Compile security patterns (always enforced)
        self._security_spec = pathspec.PathSpec.from_lines(
            "gitwildmatch", self.SECURITY_PATTERNS
        )

        # Load and compile user patterns
        self._user_spec = self._load_user_patterns()

    def _load_user_patterns(self) -> pathspec.PathSpec:
        """Load user patterns from .tesseraignore and merge with defaults.

        Security patterns are always enforced and cannot be negated.
        Warning is logged if user attempts to negate a security pattern.

        Returns:
            PathSpec compiled from merged user and default patterns.
        """
        ignore_path = self.project_root / self.ignore_file
        patterns = list(self.DEFAULT_PATTERNS)

        if ignore_path.exists():
            try:
                with open(ignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()

                        # Skip empty lines and comments
                        if not stripped or stripped.startswith("#"):
                            continue

                        # Check if user is trying to negate a security pattern
                        if stripped.startswith("!"):
                            negated_pattern = stripped[1:].strip()
                            # Use glob matching to check if negated pattern matches security patterns
                            if self._security_spec.match_file(negated_pattern):
                                logger.warning(
                                    f"Cannot negate security pattern '{negated_pattern}' "
                                    f"in {self.ignore_file}"
                                )
                                continue

                        patterns.append(stripped)
            except Exception as e:
                logger.warning(f"Failed to read {ignore_file}: {e}")

        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def should_ignore(self, rel_path: str) -> bool:
        """Check if a path should be ignored.

        Security patterns are always checked first and are un-negatable.
        User patterns are checked second.

        Args:
            rel_path: Path relative to project root

        Returns:
            True if the path should be ignored, False otherwise.
        """
        # Security patterns are always enforced
        if self._security_spec.match_file(rel_path):
            return True

        # User patterns (including defaults) are checked second
        return self._user_spec.match_file(rel_path)
