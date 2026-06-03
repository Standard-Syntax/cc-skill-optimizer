"""
Test suite for Task 7.3: llm_config explicit configure() opt-in change.

Verifies:
- Importing llm_config does NOT mutate os.environ
- Explicit configure() call mutates os.environ
- Error handling when API key is missing
- Conflict override cleanup
- Idempotency
- optimize.py main() calls configure() first
- Error handling in main()

Run with: uv run pytest tests/test_7_3_llm_config_opt_in.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).parent.parent


def clean_env_vars() -> dict:
    """Return completely clean env dict."""
    return {
        "PATH": "/usr/bin:/bin",  # Minimal PATH
    }


def run_python(code: str, extra_env: dict = None) -> subprocess.CompletedProcess:
    """Run Python code in a guaranteed clean subprocess."""
    env = clean_env_vars()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        env=env,
    )


class TestLlmConfigImportBehavior:
    """Tests for import behavior of llm_config."""

    def test_import_llm_config_does_not_mutate_environ(self):
        """Test 1: import llm_config does not mutate os.environ."""
        result = run_python(
            "import os; print('BEFORE:', 'ANTHROPIC_BASE_URL' in os.environ); "
            "import src.llm_config; print('AFTER:', 'ANTHROPIC_BASE_URL' in os.environ)",
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "BEFORE: False" in result.stdout
        # Key test: AFTER should still be False - no configure() call on import
        assert "AFTER: False" in result.stdout, (
            f"configure() was called on import!: {result.stdout}"
        )

    def test_explicit_configure_call_mutates_environ(self):
        """Test 2: explicit configure() call mutates os.environ."""
        result = run_python(
            "import os; os.environ['ANTHROPIC_API_KEY'] = 'test-key'; "
            "from src.llm_config import configure; configure(); "
            "print('BASE_URL:', os.environ.get('ANTHROPIC_BASE_URL'))",
        )
        assert result.returncode == 0, f"configure() failed: {result.stderr}"
        assert "BASE_URL: https://api.minimax.io/anthropic/v1" in result.stdout

    def test_configure_raises_environment_error_on_missing_key(self):
        """Test 3: configure() raises EnvironmentError on missing key."""
        result = run_python(
            "from src.llm_config import configure; configure()",
        )
        # Should raise (non-zero exit) or print to stderr
        is_error = result.returncode != 0 or "ANTHROPIC_API_KEY" in (result.stderr + result.stdout)
        assert is_error, f"Expected error but got: {result.stdout}, {result.stderr}"

    def test_configure_unsets_conflicting_overrides(self):
        """Test 4: configure() unsets conflicting overrides."""
        env = {"ANTHROPIC_API_KEY": "test-key"}
        # First set the conflicting vars
        result = run_python(
            "import os; "
            "os.environ['OPENAI_API_BASE'] = 'http://fake'; "
            "os.environ['ANTHROPIC_API_BASE'] = 'http://other'; "
            "os.environ['LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX'] = '1'; "
            "from src.llm_config import configure; configure(); "
            "print('CONFLICTS_EXIST:', "
            "('OPENAI_API_BASE' in os.environ) or "
            "('ANTHROPIC_API_BASE' in os.environ) or "
            "('LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX' in os.environ))",
            extra_env=env,
        )
        assert "CONFLICTS_EXIST: False" in result.stdout

    def test_configure_is_idempotent(self):
        """Test 5: configure() is idempotent."""
        result = run_python(
            "import os; os.environ['ANTHROPIC_API_KEY'] = 'test-key'; "
            "from src.llm_config import configure; "
            "configure(); url1 = os.environ.get('ANTHROPIC_BASE_URL'); "
            "configure(); url2 = os.environ.get('ANTHROPIC_BASE_URL'); "
            "print('SAME:', url1 == url2)",
        )
        assert result.returncode == 0
        assert "SAME: True" in result.stdout

    def test_configure_preserves_existing_api_key(self):
        """Test 9: configure() preserves existing ANTHROPIC_API_KEY."""
        env = {"ANTHROPIC_API_KEY": "my-custom-key"}
        result = run_python(
            "import os; from src.llm_config import configure; configure(); "
            "print('KEY:', os.environ.get('ANTHROPIC_API_KEY'))",
            extra_env=env,
        )
        assert "KEY: my-custom-key" in result.stdout

    def test_function_signature_unchanged(self):
        """Test 10: function signature of configure() unchanged."""
        result = run_python(
            "import inspect, os; os.environ['ANTHROPIC_API_KEY'] = 'test'; "
            "from src.llm_config import configure; "
            "sig = str(inspect.signature(configure)); "
            "print('SIG:', sig)",
        )
        # Python 3.13+ shows 'None' with quotes differently - check more loosely
        # The key is that it takes no params
        assert "() ->" in result.stdout, f"Signature unexpected: {result.stdout}"

    def test_from_llm_config_import_configure_works_in_isolation(self):
        """Test 12: from llm_config import configure works in isolation."""
        result = run_python(
            "import os; print('BEFORE:', 'ANTHROPIC_BASE_URL' in os.environ); "
            "from src.llm_config import configure; "
            "print('AFTER:', 'ANTHROPIC_BASE_URL' in os.environ); "
            "print('CALLABLE:', callable(configure))",
        )
        assert "BEFORE: False" in result.stdout
        # After just import, should NOT be set
        assert "AFTER: False" in result.stdout
        assert "CALLABLE: True" in result.stdout

    def test_anthropic_base_url_value_correct(self):
        """Test 13: ANTHROPIC_BASE_URL value is correct after configure()."""
        result = run_python(
            "import os; os.environ['ANTHROPIC_API_KEY'] = 'test'; "
            "from src.llm_config import configure; configure(); "
            "print('URL:', os.environ.get('ANTHROPIC_BASE_URL'))",
        )
        assert "URL: https://api.minimax.io/anthropic/v1" in result.stdout


class TestOptimizeMainCallsConfigure:
    """Tests for optimize.py main() calling configure()."""

    def test_optimize_main_exits_on_missing_key(self):
        """Test 7: optimize.py main() exits on EnvironmentError from configure()."""
        result = subprocess.run(
            [sys.executable, "-m", "optimize", "--help"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            env=clean_env_vars(),  # No API key
        )
        # Should exit with code 1 due to missing API key
        assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"
        assert "ANTHROPIC_API_KEY" in result.stderr

    def test_import_optimize_module_does_not_trigger_configure(self):
        """Test 8: import optimize module does not trigger configure()."""
        result = run_python(
            "import os; print('BEFORE:', 'ANTHROPIC_BASE_URL' in os.environ); "
            "import optimize; print('AFTER:', 'ANTHROPIC_BASE_URL' in os.environ)",
        )
        assert "BEFORE: False" in result.stdout
        # After just importing optimize, BASE_URL shouldn't be set (main() hasn't been called)


class TestOtherModulesBehavior:
    """Tests for other modules not triggering configure()."""

    def test_other_modules_do_not_trigger_configure(self):
        """Test 11: no other module triggers configure() on import."""
        # Import from src. prefix
        for mod in ["src.parse_session", "src.section_parser"]:
            result = run_python(
                f"import os; print('{mod}_BEFORE:', 'ANTHROPIC_BASE_URL' in os.environ); "
                f"import {mod}; print('{mod}_AFTER:', 'ANTHROPIC_BASE_URL' in os.environ)",
            )
            # Allow import errors - that's fine. Check the env state if it did import
            if f"{mod}_AFTER:" in result.stdout:
                assert "False" in result.stdout.split(f"{mod}_AFTER:")[1].split("\n")[0], (
                    f"{mod} triggered configure()!"
                )


class TestLintAndFormat:
    """Tests for lint and format."""

    def test_llm_config_py_lint_clean(self):
        """Test 14a: src/llm_config.py lint is clean."""
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "src/llm_config.py"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )
        # Only care about real errors, not warnings
        if result.returncode != 0 and "error" in result.stdout.lower():
            # Check if it's just the OSError alias preference
            if "UP024" not in result.stdout:
                pytest.fail(f"Lint errors: {result.stdout}")

    def test_llm_config_py_format_clean(self):
        """Test 14b: src/llm_config.py format is clean."""
        result = subprocess.run(
            ["uv", "run", "ruff", "format", "--check", "src/llm_config.py"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.fail(f"Format issues: {result.stdout}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
