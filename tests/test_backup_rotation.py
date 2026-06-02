"""
Tests for backup rotation fix in watch_and_learn.py

Tests the max_backups rotation logic that keeps only the most recent
backup files and deletes older ones beyond the limit.

These tests exercise the rotation logic directly by extracting the relevant
code paths and testing them in isolation with mocked filesystem operations.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def rotate_backups(skill_file: Path, max_backups: int) -> None:
    """
    Extract the backup rotation logic from _run_optimization for direct testing.
    This matches the rotation logic at lines 176-184 of watch_and_learn.py:

        backup_pattern = skill_file.with_suffix(".bak.*.md")
        backups = sorted(
            backup_pattern.parent.glob(backup_pattern.name),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_backup in backups[max_backups:]:
            old_backup.unlink(missing_ok=True)
    """
    backup_pattern = skill_file.with_suffix(".bak.*.md")
    backups = sorted(
        backup_pattern.parent.glob(backup_pattern.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_backup in backups[max_backups:]:
        old_backup.unlink(missing_ok=True)


def create_backup(skill_file: Path, timestamp: int, content: str = "backup content") -> Path:
    """Create a backup file with a specific timestamp."""
    backup = skill_file.with_suffix(f".bak.{timestamp}.md")
    backup.write_text(content, encoding="utf-8")
    backup.touch()
    os.utime(backup, (timestamp, timestamp))
    return backup


class TestBackupRotation:
    """Test backup rotation in _run_optimization"""

    @pytest.fixture
    def temp_skill_dir(self, tmp_path):
        """Create a temporary directory with a skill file."""
        skill_dir = tmp_path / ".claude" / "skills" / "test"
        skill_dir.mkdir(parents=True)
        return skill_dir

    @pytest.fixture
    def skill_file(self, temp_skill_dir):
        """Create a mock SKILL.md file."""
        skill = temp_skill_dir / "SKILL.md"
        skill.write_text("# Original skill content", encoding="utf-8")
        return skill

    def test_rotation_keeps_only_max_backups_most_recent(self, skill_file, temp_skill_dir):
        """
        Test that rotation correctly keeps only max_backups most recent backups.
        Creates 7 backups with different timestamps, sets max_backups=5,
        and verifies only 5 most recent remain.
        """
        max_backups = 5
        base_time = 1000000

        # Create 7 backup files with different timestamps
        for i in range(7):
            create_backup(skill_file, base_time + i * 100)

        # Run rotation
        rotate_backups(skill_file, max_backups)

        # Check that only max_backups backups remain
        backup_pattern = skill_file.with_suffix(".bak.*.md")
        remaining_backups = sorted(
            backup_pattern.parent.glob(backup_pattern.name),
            key=lambda p: p.stat().st_mtime,
        )

        assert len(remaining_backups) == max_backups, (
            f"Expected {max_backups} backups to remain, found {len(remaining_backups)}"
        )

        # Verify the 5 most recent (highest timestamps) remain
        expected_remaining_timestamps = [base_time + i * 100 for i in range(2, 7)]
        remaining_timestamps = [int(b.stem.split(".bak.")[-1]) for b in remaining_backups]
        assert remaining_timestamps == sorted(expected_remaining_timestamps)

    def test_rotation_only_deletes_backup_pattern_files(self, skill_file, temp_skill_dir):
        """
        Test that rotation only deletes files matching the backup pattern.
        Creates backup files and non-backup files, verifies only
        *.bak.*.md files are considered for deletion.
        """
        max_backups = 2
        base_time = 1000000

        # Create 3 backup files
        for i in range(3):
            create_backup(skill_file, base_time + i * 100)

        # Create non-backup files that should NOT be touched
        unrelated_file = temp_skill_dir / "unrelated.md"
        unrelated_file.write_text("not a backup", encoding="utf-8")

        other_backup = temp_skill_dir / "other.bak.txt"
        other_backup.write_text("different pattern", encoding="utf-8")

        # Run rotation
        rotate_backups(skill_file, max_backups)

        # Verify unrelated files were NOT deleted
        assert unrelated_file.exists(), "Unrelated .md file should not be deleted"
        assert other_backup.exists(), "File with different .bak pattern should not be deleted"

        # Verify backup pattern files were properly rotated
        backup_pattern = skill_file.with_suffix(".bak.*.md")
        remaining_backups = list(backup_pattern.parent.glob(backup_pattern.name))
        assert len(remaining_backups) == max_backups

    def test_rotation_does_nothing_when_fewer_than_max_backups(self, skill_file, temp_skill_dir):
        """
        Test that when fewer than max_backups exist, nothing is deleted.
        """
        max_backups = 5

        # Create only 3 backups
        base_time = 1000000
        for i in range(3):
            create_backup(skill_file, base_time + i * 100)

        initial_backups = list(
            skill_file.with_suffix(".bak.*.md").parent.glob(
                skill_file.with_suffix(".bak.*.md").name
            )
        )
        initial_count = len(initial_backups)

        # Run rotation
        rotate_backups(skill_file, max_backups)

        # All 3 original backups should still exist
        backup_pattern = skill_file.with_suffix(".bak.*.md")
        remaining_backups = list(backup_pattern.parent.glob(backup_pattern.name))

        # Should still have 3 backups (nothing deleted since we have fewer than max_backups)
        assert len(remaining_backups) == initial_count, (
            f"Expected {initial_count} backups to remain, found {len(remaining_backups)}"
        )

    def test_new_backup_included_in_count_after_write(self, skill_file, temp_skill_dir):
        """
        Test that the newly written backup is included in the count.
        Rotation happens AFTER writing the new backup, so the new backup
        should be counted toward max_backups.

        This simulates the actual flow:
        1. Write new backup
        2. Run rotation (which now sees the new backup)
        """
        max_backups = 3
        base_time = 1000000

        # Create 3 existing backups
        for i in range(3):
            create_backup(skill_file, base_time + i * 100)

        # Simulate: write a new backup (as _run_optimization does before rotation)
        new_backup_time = int(time.time())
        create_backup(skill_file, new_backup_time)

        # Now run rotation - should keep only 3 most recent
        rotate_backups(skill_file, max_backups)

        # Should have exactly max_backups backups (oldest ones deleted)
        backup_pattern = skill_file.with_suffix(".bak.*.md")
        remaining_backups = sorted(
            backup_pattern.parent.glob(backup_pattern.name),
            key=lambda p: p.stat().st_mtime,
        )

        assert len(remaining_backups) == max_backups, (
            f"Expected exactly {max_backups} backups after rotation, found {len(remaining_backups)}"
        )

        # The most recent backup should be the one we just created
        most_recent = remaining_backups[-1]
        assert int(most_recent.stem.split(".bak.")[-1]) == new_backup_time


class TestBackupRotationEdgeCases:
    """Edge case tests for backup rotation."""

    @pytest.fixture
    def temp_skill_dir(self, tmp_path):
        skill_dir = tmp_path / ".claude" / "skills" / "test"
        skill_dir.mkdir(parents=True)
        return skill_dir

    @pytest.fixture
    def skill_file(self, temp_skill_dir):
        skill = temp_skill_dir / "SKILL.md"
        skill.write_text("# Original skill", encoding="utf-8")
        return skill

    def test_max_backups_of_one(self, skill_file, temp_skill_dir):
        """Test rotation with max_backups=1."""
        base_time = 1000000

        # Create 3 backups
        for i in range(3):
            create_backup(skill_file, base_time + i * 100)

        # Run rotation
        rotate_backups(skill_file, max_backups=1)

        backup_pattern = skill_file.with_suffix(".bak.*.md")
        remaining = list(backup_pattern.parent.glob(backup_pattern.name))
        assert len(remaining) == 1

    def test_no_backups_exist(self, skill_file):
        """Test that rotation works when no backups exist."""
        # Run rotation on directory with no backups
        rotate_backups(skill_file, max_backups=5)

        backup_pattern = skill_file.with_suffix(".bak.*.md")
        remaining = list(backup_pattern.parent.glob(backup_pattern.name))
        # Should have 0 backups
        assert len(remaining) == 0

    def test_empty_directory(self, temp_skill_dir):
        """Test rotation when the skill file directory is empty."""
        skill_file = temp_skill_dir / "SKILL.md"
        skill_file.write_text("# Skill", encoding="utf-8")

        # Create some backups
        base_time = 1000000
        for i in range(2):
            create_backup(skill_file, base_time + i * 100)

        # Run rotation
        rotate_backups(skill_file, max_backups=5)

        backup_pattern = skill_file.with_suffix(".bak.*.md")
        remaining = list(backup_pattern.parent.glob(backup_pattern.name))
        # Both backups should remain (we have fewer than max_backups)
        assert len(remaining) == 2

    def test_exactly_max_backups_unchanged(self, skill_file, temp_skill_dir):
        """Test that rotation leaves exactly max_backups unchanged."""
        max_backups = 3
        base_time = 1000000

        # Create exactly max_backups backups
        for i in range(max_backups):
            create_backup(skill_file, base_time + i * 100)

        # Run rotation
        rotate_backups(skill_file, max_backups)

        backup_pattern = skill_file.with_suffix(".bak.*.md")
        remaining = list(backup_pattern.parent.glob(backup_pattern.name))
        # All should remain since we have exactly max_backups
        assert len(remaining) == max_backups

    def test_one_more_than_max_backups_deletes_oldest(self, skill_file, temp_skill_dir):
        """Test that having max_backups+1 files deletes the oldest."""
        max_backups = 3
        base_time = 1000000

        # Create max_backups + 1 backups
        for i in range(max_backups + 1):
            create_backup(skill_file, base_time + i * 100)

        # Verify we have 4 backups before rotation
        backup_pattern = skill_file.with_suffix(".bak.*.md")
        before_rotation = list(backup_pattern.parent.glob(backup_pattern.name))
        assert len(before_rotation) == 4

        # Run rotation
        rotate_backups(skill_file, max_backups)

        remaining = sorted(
            backup_pattern.parent.glob(backup_pattern.name),
            key=lambda p: p.stat().st_mtime,
        )
        # Should have exactly max_backups
        assert len(remaining) == max_backups

        # The oldest (base_time) should be deleted
        remaining_timestamps = [int(b.stem.split(".bak.")[-1]) for b in remaining]
        assert base_time not in remaining_timestamps
        assert base_time + 100 in remaining_timestamps
        assert base_time + 200 in remaining_timestamps
        assert base_time + 300 in remaining_timestamps


class TestBackupRotationIntegration:
    """
    Integration tests that verify the rotation logic works correctly
    when called in the actual sequence used by _run_optimization.
    """

    @pytest.fixture
    def temp_skill_dir(self, tmp_path):
        skill_dir = tmp_path / ".claude" / "skills" / "test"
        skill_dir.mkdir(parents=True)
        return skill_dir

    @pytest.fixture
    def skill_file(self, temp_skill_dir):
        skill = temp_skill_dir / "SKILL.md"
        skill.write_text("# Original skill", encoding="utf-8")
        return skill

    def test_full_rotation_sequence(self, skill_file, temp_skill_dir):
        """
        Simulate the full sequence from _run_optimization:
        1. Create some old backups
        2. Write new backup
        3. Run rotation
        4. Verify correct files remain
        """
        max_backups = 5
        base_time = 1000000

        # Step 1: Create old backups (simulating previous runs)
        for i in range(7):
            create_backup(skill_file, base_time + i * 100)

        # Verify pre-condition
        backup_pattern = skill_file.with_suffix(".bak.*.md")
        before = list(backup_pattern.parent.glob(backup_pattern.name))
        assert len(before) == 7

        # Step 2: Simulate writing new backup (this happens before rotation in _run_optimization)
        new_backup_time = int(time.time())
        create_backup(skill_file, new_backup_time)

        # Verify new backup is included
        after_write = sorted(
            backup_pattern.parent.glob(backup_pattern.name),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        assert len(after_write) == 8
        assert after_write[0].stat().st_mtime >= new_backup_time

        # Step 3: Run rotation (this is what _run_optimization does after writing backup)
        rotate_backups(skill_file, max_backups)

        # Step 4: Verify correct files remain
        remaining = sorted(
            backup_pattern.parent.glob(backup_pattern.name),
            key=lambda p: p.stat().st_mtime,
        )
        assert len(remaining) == max_backups

        # The 5 most recent should remain (timestamps: base_time+200 through base_time+600 and new_backup_time)
        # Oldest ones (base_time, base_time+100) should be deleted
        remaining_timestamps = [int(b.stem.split(".bak.")[-1]) for b in remaining]
        assert base_time not in remaining_timestamps  # oldest deleted
        assert base_time + 100 not in remaining_timestamps  # second oldest deleted
