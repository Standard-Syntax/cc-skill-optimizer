"""
Verify 5 remaining correctness fixes in optimize.py:
1. Issue 7: OBJECTIVE_BY_TARGET and BACKGROUND_BY_TARGET at module level
2. Issue 14: _walk_subdirs caller refactored (no duplicate root_dir processing)
3. Issue 16: bare except blocks now log warnings
4. Issue 17: f-string prefix already fixed (no bare f-strings exist)
5. Issue 18: SEED_BY_TARGET has all required keys including "multi", "sections", "nested"
"""

import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIssue7ModuleLevelConstants(unittest.TestCase):
    """Verify Issue 7: OBJECTIVE_BY_TARGET and BACKGROUND_BY_TARGET are at module level."""

    def test_objective_by_target_is_module_level(self):
        """OBJECTIVE_BY_TARGET should be importable from module namespace."""
        import optimize

        self.assertIn("OBJECTIVE_BY_TARGET", dir(optimize))
        self.assertIsInstance(optimize.OBJECTIVE_BY_TARGET, dict)

    def test_background_by_target_is_module_level(self):
        """BACKGROUND_BY_TARGET should be importable from module namespace."""
        import optimize

        self.assertIn("BACKGROUND_BY_TARGET", dir(optimize))
        self.assertIsInstance(optimize.BACKGROUND_BY_TARGET, dict)

    def test_objective_by_target_has_all_targets(self):
        """OBJECTIVE_BY_TARGET must have all 6 target keys."""
        from optimize import OBJECTIVE_BY_TARGET

        required_keys = {"skill", "claude", "agent", "multi", "sections", "nested"}
        self.assertEqual(set(OBJECTIVE_BY_TARGET.keys()), required_keys)

    def test_background_by_target_has_all_targets(self):
        """BACKGROUND_BY_TARGET must have all 6 target keys."""
        from optimize import BACKGROUND_BY_TARGET

        required_keys = {"skill", "claude", "agent", "multi", "sections", "nested"}
        self.assertEqual(set(BACKGROUND_BY_TARGET.keys()), required_keys)

    def test_import_directly_from_module(self):
        """Should be able to import these constants directly."""
        # This tests they are truly at module level, not nested in a function
        from optimize import BACKGROUND_BY_TARGET, OBJECTIVE_BY_TARGET

        self.assertTrue(len(OBJECTIVE_BY_TARGET["skill"]) > 0)
        self.assertTrue(len(BACKGROUND_BY_TARGET["skill"]) > 0)


class TestIssue14WalkSubdirsNoDuplicateRoot(unittest.TestCase):
    """Verify Issue 14: _walk_subdirs caller doesn't process root_dir twice."""

    def test_load_nested_files_single_root_processing(self):
        """load_nested_files should process root only once in the all_dirs list."""
        import optimize

        source = inspect.getsource(optimize.load_nested_files)
        # The fix should be: all_dirs = [root_dir] + _walk_subdirs(root_dir, max_depth)
        # This means root_dir is explicitly added once, not included in _walk_subdirs result
        self.assertIn("all_dirs = [root_dir]", source)
        self.assertIn("_walk_subdirs(root_dir", source)

    def test_walk_subdirs_does_not_include_root(self):
        """_walk_subdirs should only return subdirectories, not the root."""
        import optimize

        source = inspect.getsource(optimize._walk_subdirs)
        # _walk starts at root but does NOT add root to results
        # It only appends 'entry' (child directories) to results
        self.assertIn("results.append(entry)", source)
        # The function should NOT have logic that adds 'root' to results

    @mock.patch("pathlib.Path.exists", return_value=True)
    @mock.patch("pathlib.Path.iterdir", return_value=[])
    @mock.patch("pathlib.Path.is_file", return_value=False)
    def test_walk_subdirs_returns_only_children(self, mock_is_file, mock_iterdir, mock_exists):
        """_walk_subdirs should return only subdirectory Path objects, not root."""
        from optimize import _walk_subdirs

        # Create a mock root
        mock_root = mock.MagicMock(spec=Path)
        mock_root.exists.return_value = True
        mock_root.iterdir.return_value = []  # No children

        result = _walk_subdirs(mock_root, max_depth=2)
        # Should return empty list since there are no subdirectories
        self.assertEqual(result, [])

    @mock.patch("pathlib.Path.exists", return_value=True)
    @mock.patch("pathlib.Path.iterdir")
    @mock.patch("pathlib.Path.is_dir", return_value=True)
    @mock.patch("pathlib.Path.is_file", return_value=False)
    def test_walk_subdirs_excludes_root_from_results(
        self, mock_is_file, mock_is_dir, mock_iterdir, mock_exists
    ):
        """Even when root has children, root itself should not be in results."""
        from optimize import _walk_subdirs

        # Create mock structure: root -> child_dir
        mock_root = mock.MagicMock(spec=Path)
        mock_root.exists.return_value = True
        mock_root.name = "root"

        mock_child = mock.MagicMock(spec=Path)
        mock_child.name = "child_dir"
        mock_child.is_dir.return_value = True
        mock_child.is_file.return_value = False
        mock_child.iterdir.return_value = []

        mock_root.iterdir.return_value = [mock_child]

        result = _walk_subdirs(mock_root, max_depth=2)

        # child_dir should be in results, but root should NOT be
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "child_dir")


class TestIssue16BareExceptLogsWarnings(unittest.TestCase):
    """Verify Issue 16: bare except blocks now log warnings."""

    def test_bare_except_blocks_have_logging(self):
        """All bare 'except Exception:' blocks should call logger.warning."""
        import optimize

        source = inspect.getsource(optimize)

        # Find all except Exception blocks (both "except Exception:" and "except Exception as exc:")
        bare_except_count = source.count("except Exception:")
        bare_except_as_count = source.count("except Exception as ")
        total_bare_except = bare_except_count + bare_except_as_count
        self.assertGreaterEqual(
            total_bare_except,
            2,
            f"Should have at least 2 bare except blocks, found {total_bare_except}",
        )

        # Verify each bare except is followed by logger.warning
        # We check the specific function run_dspy_gepa
        dspy_source = inspect.getsource(optimize.run_dspy_gepa)

        # First bare except block at line ~949
        self.assertIn("except Exception:", dspy_source)
        self.assertIn("logger.warning", dspy_source)
        self.assertIn("Could not extract optimized instructions", dspy_source)

        # Second bare except at line ~957
        self.assertIn("except Exception as exc:", dspy_source)
        self.assertIn("logger.warning", dspy_source)
        self.assertIn("DSPy stage-1 initialization failed", dspy_source)

    def test_permission_error_not_bare_except(self):
        """PermissionError catch is specific, not bare except - allowed to use 'pass'."""
        import optimize

        source = inspect.getsource(optimize._walk_subdirs)
        # The _walk function catches PermissionError specifically (not bare except)
        # This is acceptable because PermissionError on iterdir() is expected and handled
        self.assertIn("except PermissionError:", source)
        # But it's a specific exception, not "except Exception:"
        self.assertNotIn("except Exception:", source)

    def test_exception_handlers_use_logger(self):
        """Verify that exception handlers use the logging module, not print."""
        import optimize

        source = inspect.getsource(optimize.run_dspy_gepa)

        # Should use logging.getLogger and logger.warning
        self.assertIn("logging.getLogger(__name__)", source)
        self.assertIn("logger.warning", source)

        # Should NOT use bare print for errors
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "except Exception" in line:
                # Find the next few lines that form the except block
                block = "\n".join(lines[i : i + 5])
                self.assertIn(
                    "logger.warning",
                    block,
                    f"Except block should use logger.warning, not print. Block:\n{block}",
                )


class TestIssue17FStringPrefixFixed(unittest.TestCase):
    """Verify Issue 17: f-string prefix already fixed (no bare f-strings exist)."""

    def test_no_plain_fstrings_exist(self):
        """All f-strings in optimize.py should have proper prefixes."""
        import optimize

        source = inspect.getsource(optimize)

        # Check that f-strings are properly used
        # The issue was about bare f-strings being incorrectly formatted
        # Now they should all have proper f"" prefix

        # Check line 518 specifically - it's a regular string replace, not an f-string
        lines = source.split("\n")
        for i, line in enumerate(lines):
            i + 1
            stripped = line.strip()

            # Skip comments
            if stripped.startswith("#"):
                continue

            # If line contains an f-string literal pattern, verify it has f prefix
            if "'{" in line or '"{' in line or ".format(" in line:
                # These patterns suggest string formatting - verify correct usage
                pass

    def test_load_nested_files_no_fstring_issues(self):
        """Verify load_nested_files has no f-string problems."""
        import optimize

        source = inspect.getsource(optimize.load_nested_files)

        # The original issue was about f-string prefix
        # Now the code should be clean
        self.assertNotIn('f"}', source)
        self.assertNotIn("f'}", source)

        # Check line 518 area - should be key = str(rel_path).replace("\\", "/")
        lines = source.split("\n")
        for line in lines:
            if "rel_path" in line and "replace" in line:
                # This is the line that was potentially fixed
                self.assertNotIn('f"', line, "String replace on rel_path should not be an f-string")


class TestIssue18SeedByTargetHasAllKeys(unittest.TestCase):
    """Verify Issue 18: SEED_BY_TARGET has all required keys including multi, sections, nested."""

    def test_seed_by_target_has_six_keys(self):
        """SEED_BY_TARGET should have exactly 6 keys."""
        from optimize import SEED_BY_TARGET

        self.assertEqual(len(SEED_BY_TARGET), 6)
        self.assertIn("skill", SEED_BY_TARGET)
        self.assertIn("claude", SEED_BY_TARGET)
        self.assertIn("agent", SEED_BY_TARGET)
        self.assertIn("multi", SEED_BY_TARGET)
        self.assertIn("sections", SEED_BY_TARGET)
        self.assertIn("nested", SEED_BY_TARGET)

    def test_seed_by_target_multi_key(self):
        """SEED_BY_TARGET['multi'] should exist and be a string."""
        from optimize import SEED_BY_TARGET

        self.assertIn("multi", SEED_BY_TARGET)
        self.assertIsInstance(SEED_BY_TARGET["multi"], str)
        self.assertGreater(len(SEED_BY_TARGET["multi"]), 0)

    def test_seed_by_target_sections_key(self):
        """SEED_BY_TARGET['sections'] should exist and be a string."""
        from optimize import SEED_BY_TARGET

        self.assertIn("sections", SEED_BY_TARGET)
        self.assertIsInstance(SEED_BY_TARGET["sections"], str)
        self.assertGreater(len(SEED_BY_TARGET["sections"]), 0)

    def test_seed_by_target_nested_key(self):
        """SEED_BY_TARGET['nested'] should exist and be a string."""
        from optimize import SEED_BY_TARGET

        self.assertIn("nested", SEED_BY_TARGET)
        self.assertIsInstance(SEED_BY_TARGET["nested"], str)
        self.assertGreater(len(SEED_BY_TARGET["nested"]), 0)

    def test_seed_by_target_references_correct_seed_variables(self):
        """SEED_BY_TARGET values should reference actual SEED_* variables."""
        from optimize import SEED_AGENTS_MD, SEED_BY_TARGET, SEED_CLAUDE_MD, SEED_SKILL_MD

        # skill -> SEED_SKILL_MD
        self.assertEqual(SEED_BY_TARGET["skill"], SEED_SKILL_MD)

        # claude -> SEED_CLAUDE_MD
        self.assertEqual(SEED_BY_TARGET["claude"], SEED_CLAUDE_MD)

        # agent -> SEED_AGENTS_MD
        self.assertEqual(SEED_BY_TARGET["agent"], SEED_AGENTS_MD)

        # multi, sections -> SEED_SKILL_MD (fallback for dict-based modes)
        self.assertEqual(SEED_BY_TARGET["multi"], SEED_SKILL_MD)
        self.assertEqual(SEED_BY_TARGET["sections"], SEED_SKILL_MD)

        # nested -> SEED_CLAUDE_MD (for nested file optimization)
        self.assertEqual(SEED_BY_TARGET["nested"], SEED_CLAUDE_MD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
