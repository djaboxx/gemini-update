"""
Unit tests for the file access abstraction layer.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.file_access import (
    FileAccessFactory,
    FileInfo,
    GeminiFileAccessLayer,
    LocalFileAccessLayer,
)


class TestLocalFileAccessLayer(unittest.TestCase):
    """Test the local file access layer."""

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = Path(__file__).parent
        self.file_access = LocalFileAccessLayer(self.test_dir)

    def test_read_file(self):
        """Test reading a file."""
        # Create a test file
        test_file = self.test_dir / "test_read_file.txt"
        content = "Line 1\nLine 2\nLine 3\n"
        with open(test_file, "w") as f:
            f.write(content)

        try:
            # Test reading the entire file
            result = self.file_access.read_file(str(test_file))
            self.assertEqual(result, content)

            # Test reading specific lines
            result = self.file_access.read_file(str(test_file), start_line=1, end_line=1)
            self.assertEqual(result, "Line 2\n")
        finally:
            # Clean up
            if test_file.exists():
                os.remove(test_file)

    def test_list_directory(self):
        """Test listing a directory."""
        result = self.file_access.list_directory(str(self.test_dir))
        self.assertTrue(len(result) > 0)
        self.assertTrue("test_file_access.py" in result)

    def test_file_exists(self):
        """Test checking if a file exists."""
        self.assertTrue(self.file_access.file_exists(__file__))
        self.assertFalse(self.file_access.file_exists("nonexistent_file.txt"))

    def test_is_directory(self):
        """Test checking if a path is a directory."""
        self.assertTrue(self.file_access.is_directory(str(self.test_dir)))
        self.assertFalse(self.file_access.is_directory(__file__))

    def test_get_file_info(self):
        """Test getting file information."""
        # Test a file
        info = self.file_access.get_file_info(__file__)
        self.assertTrue(info.exists)
        self.assertFalse(info.is_dir)
        self.assertTrue(info.size > 0)
        
        # Test a directory
        info = self.file_access.get_file_info(str(self.test_dir))
        self.assertTrue(info.exists)
        self.assertTrue(info.is_dir)

    def test_validate_path_inside(self):
        """Test validating paths inside the project directory."""
        result = self.file_access.validate_path(__file__)
        self.assertEqual(result, str(Path(__file__).resolve()))

    def test_validate_path_outside(self):
        """Test validating paths outside the project directory."""
        with self.assertRaises(ValueError):
            self.file_access.validate_path("/tmp/outside_project.txt")


class TestGeminiFileAccessLayer(unittest.TestCase):
    """Test the Gemini file access layer."""

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = Path(__file__).parent
        self.mock_sync_manager = MagicMock()
        self.mock_sync_manager.handle_file_change = MagicMock(return_value="gemini://file/123")
        self.mock_sync_manager.state_manager.get_all_file_states = MagicMock(return_value=[])
        self.file_access = GeminiFileAccessLayer(self.mock_sync_manager, self.test_dir)

    def test_read_file(self):
        """Test reading a file through Gemini Files API."""
        # Create a test file
        test_file = self.test_dir / "test_gemini_read.txt"
        content = "Line 1\nLine 2\nLine 3\n"
        with open(test_file, "w") as f:
            f.write(content)

        try:
            # Test reading the entire file
            result = self.file_access.read_file(str(test_file))
            self.assertEqual(result, content)
            self.mock_sync_manager.handle_file_change.assert_called_with(Path(test_file).resolve())

            # Test reading specific lines
            result = self.file_access.read_file(str(test_file), start_line=1, end_line=1)
            self.assertEqual(result, "Line 2\n")
        finally:
            # Clean up
            if test_file.exists():
                os.remove(test_file)

    def test_file_exists(self):
        """Test checking if a file exists with Gemini integration."""
        self.assertTrue(self.file_access.file_exists(__file__))
        self.assertFalse(self.file_access.file_exists("nonexistent_file.txt"))


class TestFileAccessFactory(unittest.TestCase):
    """Test the file access factory."""

    def test_create_local_access_layer(self):
        """Test creating a local file access layer."""
        access_layer = FileAccessFactory.create_file_access_layer(Path(__file__).parent)
        self.assertIsInstance(access_layer, LocalFileAccessLayer)

    def test_create_gemini_access_layer(self):
        """Test creating a Gemini file access layer."""
        mock_sync_manager = MagicMock()
        access_layer = FileAccessFactory.create_file_access_layer(
            Path(__file__).parent, mock_sync_manager
        )
        self.assertIsInstance(access_layer, GeminiFileAccessLayer)


if __name__ == "__main__":
    unittest.main()
