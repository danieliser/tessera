"""Tests for the two-tier .tesseraignore system."""
import os
import tempfile
import pytest
from tessera.ignore import IgnoreFilter


class TestIgnoreFilter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_security_patterns_always_enforced(self):
        f = IgnoreFilter(self.tmpdir)
        assert f.should_ignore(".env") is True
        assert f.should_ignore(".env.local") is True
        assert f.should_ignore("config.pem") is True
        assert f.should_ignore("server.key") is True
        assert f.should_ignore("cert.p12") is True
        assert f.should_ignore("my_credentials.json") is True
        assert f.should_ignore("app_secret_key.txt") is True
        assert f.should_ignore("id_rsa") is True
        assert f.should_ignore("id_ed25519") is True
        assert f.should_ignore("auth.token") is True
        assert f.should_ignore("service-account.json") is True

    def test_default_patterns(self):
        f = IgnoreFilter(self.tmpdir)
        assert f.should_ignore("node_modules/foo.js") is True
        assert f.should_ignore("vendor/lib.php") is True
        assert f.should_ignore("__pycache__/mod.pyc") is True
        assert f.should_ignore(".venv/bin/python") is True
        assert f.should_ignore(".DS_Store") is True

    def test_normal_files_not_ignored(self):
        f = IgnoreFilter(self.tmpdir)
        assert f.should_ignore("src/main.py") is False
        assert f.should_ignore("README.md") is False
        assert f.should_ignore("docs/api.yaml") is False

    def test_user_cannot_negate_security_patterns(self):
        ignore_path = os.path.join(self.tmpdir, ".tesseraignore")
        with open(ignore_path, "w") as fh:
            fh.write("!.env\n!*.pem\n!*credentials*\n")
        f = IgnoreFilter(self.tmpdir)
        assert f.should_ignore(".env") is True
        assert f.should_ignore("test.pem") is True
        assert f.should_ignore("credentials.json") is True

    def test_user_can_negate_defaults(self):
        ignore_path = os.path.join(self.tmpdir, ".tesseraignore")
        with open(ignore_path, "w") as fh:
            fh.write("!node_modules/\n")
        f = IgnoreFilter(self.tmpdir)
        assert f.should_ignore("node_modules/foo.js") is False

    def test_user_can_add_custom_patterns(self):
        ignore_path = os.path.join(self.tmpdir, ".tesseraignore")
        with open(ignore_path, "w") as fh:
            fh.write("custom_dir/\n*.bak\n")
        f = IgnoreFilter(self.tmpdir)
        assert f.should_ignore("custom_dir/file.txt") is True
        assert f.should_ignore("backup.bak") is True

    def test_glob_based_security_negation_check(self):
        """CTO Condition C2: negation uses glob matching, not string equality."""
        ignore_path = os.path.join(self.tmpdir, ".tesseraignore")
        # Try to negate with a variant that wouldn't match string equality
        with open(ignore_path, "w") as fh:
            fh.write("!.env.local\n")  # .env* should still block this
        f = IgnoreFilter(self.tmpdir)
        assert f.should_ignore(".env.local") is True
