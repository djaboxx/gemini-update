"""
Unit tests for the feature tools module with Gemini API Files support.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai import RunContext

from src.models.analysis import CodebaseContext
from src.models.file_access import FileAccessFactory, GeminiFileAccessLayer, LocalFileAccessLayer
from src.tools.feature_tools import identify_affected_files


class TestIdentifyAffectedFiles(unittest.TestCase):
    """Test the identify_affected_files function with both local and Gemini files."""

    def setUp(self):
        """Set up the test environment with a mock project structure."""
        self.test_dir = Path(__file__).parent / "test_project"
        self.test_dir.mkdir(exist_ok=True)
        
        # Create mock project structure
        src_dir = self.test_dir / "src"
        src_dir.mkdir(exist_ok=True)
        
        models_dir = src_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        controllers_dir = src_dir / "controllers"
        controllers_dir.mkdir(exist_ok=True)
        
        # Create some mock Python files
        main_file = self.test_dir / "main.py"
        with open(main_file, "w") as f:
            f.write("import src.models.user\nfrom src.controllers.auth_controller import login\n\n")
            f.write("def main():\n    print('Hello, world!')\n\n")
            f.write("if __name__ == '__main__':\n    main()\n")
        
        user_model = models_dir / "user.py"
        with open(user_model, "w") as f:
            f.write("class User:\n    def __init__(self, name, email):\n")
            f.write("        self.name = name\n        self.email = email\n\n")
            f.write("    def validate(self):\n        return '@' in self.email\n")
        
        auth_controller = controllers_dir / "auth_controller.py"
        with open(auth_controller, "w") as f:
            f.write("from src.models.user import User\n\n")
            f.write("def login(email, password):\n    user = User('Test User', email)\n")
            f.write("    if not user.validate():\n        return False\n")
            f.write("    # More authentication logic\n    return True\n")
        
        # Create mock config files
        package_json = self.test_dir / "package.json"
        with open(package_json, "w") as f:
            f.write('{"name": "test-project", "version": "1.0.0"}\n')

    def tearDown(self):
        """Clean up the test environment."""
        # Remove all files and directories from test_project
        if self.test_dir.exists():
            for file_path in self.test_dir.glob("**/*"):
                if file_path.is_file():
                    file_path.unlink()
            
            for dir_path in sorted(list(self.test_dir.glob("**/*")), key=lambda x: len(str(x)), reverse=True):
                if dir_path.is_dir():
                    dir_path.rmdir()
            
            self.test_dir.rmdir()

    @patch("src.tools.feature_tools.re.findall")
    @patch("src.tools.feature_tools.os.listdir")
    async def test_identify_affected_files_local(self, mock_listdir, mock_findall):
        """Test identifying affected files with local filesystem."""
        # Set up mock returns
        mock_findall.return_value = ["user", "authentication", "login"]
        mock_listdir.return_value = ["user.py", "other.py"]
        
        # Set up mock context
        ctx = MagicMock(spec=RunContext)
        ctx.deps = CodebaseContext(project_dir=self.test_dir)
        ctx.execute = AsyncMock()
        ctx.execute.return_value = {
            "imports": [{"name": "src.models.user"}],
            "from_imports": [
                {"module": "src.controllers.auth_controller", "names": ["login"]}
            ]
        }
        
        # Run the function
        result = await identify_affected_files(
            ctx, "Implement user authentication login feature"
        )
        
        # Verify results
        self.assertTrue(len(result) > 0)
        self.assertTrue(any("user.py" in path for path in result))
        self.assertTrue(any("auth_controller.py" in path for path in result))
        self.assertTrue("package.json" in result)

    @patch("src.tools.feature_tools.re.findall")
    async def test_identify_affected_files_gemini(self, mock_findall):
        """Test identifying affected files with Gemini Files API."""
        # Set up mock returns
        mock_findall.return_value = ["user", "authentication", "login"]
        
        # Create mock GeminiFileManager
        mock_sync_manager = MagicMock()
        mock_sync_manager.handle_file_change = MagicMock(return_value="gemini://file/123")
        mock_sync_manager.state_manager.get_all_file_states = MagicMock(return_value=[])
        
        # Create file access layer with mock sync manager
        file_access = GeminiFileAccessLayer(mock_sync_manager, self.test_dir)
        
        # Mock file access methods
        original_file_exists = file_access.file_exists
        original_is_directory = file_access.is_directory
        original_list_directory = file_access.list_directory
        
        file_access.file_exists = MagicMock(side_effect=lambda path: original_file_exists(path))
        file_access.is_directory = MagicMock(side_effect=lambda path: original_is_directory(path))
        file_access.list_directory = MagicMock(side_effect=lambda path: original_list_directory(path))
        
        # Set up mock context with Gemini file access
        ctx = MagicMock(spec=RunContext)
        ctx.deps = CodebaseContext(
            project_dir=self.test_dir,
            gemini_sync_manager=mock_sync_manager,
            file_access=file_access
        )
        ctx.execute = AsyncMock()
        ctx.execute.return_value = {
            "imports": [{"name": "src.models.user"}],
            "from_imports": [
                {"module": "src.controllers.auth_controller", "names": ["login"]}
            ]
        }
        
        # Run the function
        result = await identify_affected_files(
            ctx, "Implement user authentication login feature"
        )
        
        # Verify results
        self.assertTrue(len(result) > 0)
        self.assertTrue(any("user.py" in path for path in result))
        self.assertTrue(any("auth_controller.py" in path for path in result))
        self.assertTrue("package.json" in result)
        
        # Verify file access layer methods were called
        self.assertTrue(file_access.file_exists.called)
        self.assertTrue(file_access.is_directory.called)
        self.assertTrue(file_access.list_directory.called)


if __name__ == "__main__":
    unittest.main()
