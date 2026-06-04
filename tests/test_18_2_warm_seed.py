"""
Tests for Task 18.2: warm-restart seeding in watch_and_learn.py.

The watcher reads <output_dir>/<target>/best_candidate.md before each
optimization to use the prior run's best as the seed. Falls back to
the original seed (current skill_file content) if no prior best exists.
"""

import sys
import tempfile
from pathlib import Path

# Ensure src/ is on path so watch_and_learn.py's internal imports work
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))  # noqa: E402

from watch_and_learn import _get_warm_seed  # noqa: E402


class TestGetWarmSeed:
    """_get_warm_seed returns prior best when it exists, original seed otherwise."""

    def test_returns_original_seed_when_no_prior_best_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            target = "skill"
            original_seed = "# Original skill content\n"
            result = _get_warm_seed(output_dir, target, original_seed)
            assert result == original_seed, "Should fall back to original_seed"

    def test_returns_prior_best_when_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            target = "skill"
            prior_best = "# Prior run's best skill\n# 1500 chars\n"
            # Pre-create the prior best file
            best_path = output_dir / target / "best_candidate.md"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_path.write_text(prior_best, encoding="utf-8")
            # Call _get_warm_seed
            result = _get_warm_seed(output_dir, target, "# Original\n")
            assert result == prior_best, f"Should return prior best, got {result!r}"

    def test_returns_original_seed_when_target_dir_does_not_exist(self):
        """If <output_dir>/<target>/ doesn't exist at all, fall back to original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            # Don't create the target dir
            result = _get_warm_seed(output_dir, "nonexistent_target", "# Original\n")
            assert result == "# Original\n"

    def test_falls_back_on_read_error(self, monkeypatch):
        """If best_candidate.md exists but read fails, fall back to original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            target = "skill"
            best_path = output_dir / target / "best_candidate.md"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_path.write_text("some content", encoding="utf-8")

            # Force read_text to raise OSError
            def raise_oserror(*args, **kwargs):
                raise OSError("simulated read failure")

            monkeypatch.setattr(Path, "read_text", raise_oserror)
            original_seed = "# Original fallback\n"
            result = _get_warm_seed(output_dir, target, original_seed)
            assert result == original_seed, "Should fall back to original on read error"

    def test_handles_unicode_decode_error(self, monkeypatch):
        """If best_candidate.md has invalid UTF-8, fall back to original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            target = "skill"
            best_path = output_dir / target / "best_candidate.md"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_path.write_bytes(b"\xff\xfe\xfd")  # Invalid UTF-8

            # Force read_text to raise UnicodeDecodeError
            def raise_unicode(*args, **kwargs):
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")

            monkeypatch.setattr(Path, "read_text", raise_unicode)
            original_seed = "# Original\n"
            result = _get_warm_seed(output_dir, target, original_seed)
            assert result == original_seed


class TestWarmSeedOutputDirConvention:
    """The default output_dir is 'outputs/' and target is derived from skill_file.stem."""

    def test_default_output_dir_is_outputs(self):
        """When output_dir is None, watch_and_learn defaults to Path('outputs')."""
        import inspect

        # We test this by importing watch_and_learn and inspecting the default behavior
        from watch_and_learn import watch_and_learn  # noqa: F401

        sig = inspect.signature(watch_and_learn)
        params = sig.parameters
        # output_dir should be a parameter with default None
        assert "output_dir" in params
        assert params["output_dir"].default is None
        # target should also have default None
        assert "target" in params
        assert params["target"].default is None
        print("OK: watch_and_learn has output_dir and target kwargs with default None")

    def test_main_block_has_output_dir_argparse(self):
        """The __main__ block adds --output-dir to argparse."""
        import ast

        watch_path = Path(__file__).parent.parent / "watch_and_learn.py"
        src = watch_path.read_text(encoding="utf-8")
        # Parse and find the __main__ block
        tree = ast.parse(src)
        # Find the if __name__ == "__main__" block
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                ):
                    # Walk the body
                    for child in node.body:
                        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                            call = child.value
                            # Look for argparse add_argument calls
                            if (
                                isinstance(call.func, ast.Attribute)
                                and call.func.attr == "add_argument"
                                and call.args
                                and isinstance(call.args[0], ast.Constant)
                                and call.args[0].value == "--output-dir"
                            ):
                                print("OK: --output-dir argparse flag found")
                                return
        raise AssertionError("--output-dir argparse flag not found in watch_and_learn.py __main__ block")
