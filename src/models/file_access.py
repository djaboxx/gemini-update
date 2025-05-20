"""
Models for abstracting file access between local filesystem and Gemini API Files.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Union

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """Information about a file, regardless of storage location."""
    
    path: str = Field(..., description="Path to the file (relative or absolute)")
    exists: bool = Field(True, description="Whether the file exists")
    is_dir: bool = Field(False, description="Whether the path refers to a directory")
    size: int = Field(0, description="Size of the file in bytes")
    mime_type: str = Field("text/plain", description="MIME type of the file")
    uri: Optional[str] = Field(None, description="URI for Gemini API Files")


class FileAccessLayer(ABC):
    """Abstract base class for file access operations."""
    
    @abstractmethod
    def read_file(self, file_path: str, start_line: Optional[int] = None, 
                  end_line: Optional[int] = None) -> str:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            start_line: Optional starting line number (0-based)
            end_line: Optional ending line number (0-based)
            
        Returns:
            The file content as a string
        """
        pass
        
    @abstractmethod
    def list_directory(self, dir_path: str) -> List[str]:
        """
        List files and directories within a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            List of file and directory names
        """
        pass
        
    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file exists, False otherwise
        """
        pass
        
    @abstractmethod
    def is_directory(self, path: str) -> bool:
        """
        Check if a path refers to a directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is a directory, False otherwise
        """
        pass
        
    @abstractmethod
    def get_file_info(self, path: str) -> FileInfo:
        """
        Get information about a file or directory.
        
        Args:
            path: Path to the file or directory
            
        Returns:
            FileInfo object with information about the file or directory
        """
        pass
        
    @abstractmethod
    def validate_path(self, path: str) -> str:
        """
        Validate that a path is within the allowed scope.
        
        Args:
            path: Path to validate
            
        Returns:
            Validated path
            
        Raises:
            ValueError: If the path is outside the allowed scope
        """
        pass


class LocalFileAccessLayer(FileAccessLayer):
    """File access layer for local filesystem."""
    
    def __init__(self, project_dir: Path):
        """
        Initialize the local file access layer.
        
        Args:
            project_dir: Base directory for file operations
        """
        self.project_dir = project_dir
    
    def read_file(self, file_path: str, start_line: Optional[int] = None, 
                  end_line: Optional[int] = None) -> str:
        """
        Read content from a file in the local filesystem.
        
        Args:
            file_path: Relative or absolute path to the file
            start_line: Optional starting line number (0-based)
            end_line: Optional ending line number (0-based)
            
        Returns:
            The file content as a string
        """
        abs_path = self.validate_path(file_path)
        
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            if start_line is None and end_line is None:
                # Read the entire file
                return f.read()
            else:
                # Read specified lines
                lines = f.readlines()
                
                if start_line is None:
                    start_line = 0
                if end_line is None:
                    end_line = len(lines) - 1
                
                # Ensure line numbers are within bounds
                start_line = max(0, min(start_line, len(lines) - 1))
                end_line = max(0, min(end_line, len(lines) - 1))
                
                # Return the specified lines
                return "".join(lines[start_line : end_line + 1])
    
    def list_directory(self, dir_path: str) -> List[str]:
        """
        List files and directories within a directory in the local filesystem.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            List of file and directory names
        """
        abs_path = self.validate_path(dir_path)
        return os.listdir(abs_path)
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the local filesystem.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file exists, False otherwise
        """
        try:
            abs_path = self.validate_path(file_path)
            return Path(abs_path).exists()
        except ValueError:
            return False
    
    def is_directory(self, path: str) -> bool:
        """
        Check if a path refers to a directory in the local filesystem.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is a directory, False otherwise
        """
        try:
            abs_path = self.validate_path(path)
            return Path(abs_path).is_dir()
        except ValueError:
            return False
    
    def get_file_info(self, path: str) -> FileInfo:
        """
        Get information about a file or directory in the local filesystem.
        
        Args:
            path: Path to the file or directory
            
        Returns:
            FileInfo object with information about the file or directory
        """
        try:
            abs_path = self.validate_path(path)
            path_obj = Path(abs_path)
            
            exists = path_obj.exists()
            is_dir = path_obj.is_dir() if exists else False
            size = path_obj.stat().st_size if exists and not is_dir else 0
            mime_type = "text/plain"  # Basic default
            
            return FileInfo(
                path=path,
                exists=exists,
                is_dir=is_dir,
                size=size,
                mime_type=mime_type
            )
        except ValueError:
            return FileInfo(
                path=path,
                exists=False,
                is_dir=False,
                size=0,
                mime_type="text/plain"
            )
    
    def validate_path(self, path: str) -> str:
        """
        Validate that a path is within the project directory.
        
        Args:
            path: Relative or absolute path
            
        Returns:
            Absolute path to the validated file or directory
            
        Raises:
            ValueError: If the path is outside the project directory
        """
        path_obj = Path(path)
        
        # Handle absolute paths
        if path_obj.is_absolute():
            abs_path = path_obj
        else:
            # Handle relative paths
            abs_path = (self.project_dir / path_obj).resolve()
            
        # Check if the path is within the project directory
        try:
            abs_path.relative_to(self.project_dir)
        except ValueError:
            raise ValueError(f"Path '{path}' is outside the project directory")
            
        return str(abs_path)


class GeminiFileAccessLayer(FileAccessLayer):
    """File access layer for Gemini API Files."""
    
    def __init__(self, sync_manager: Any, project_dir: Path):
        """
        Initialize the Gemini file access layer.
        
        Args:
            sync_manager: Gemini sync manager instance
            project_dir: Base directory for file operations
        """
        self.sync_manager = sync_manager
        self.project_dir = project_dir
        self._file_states = {}  # Cache of file states
        self._refresh_file_states()
        
    def _refresh_file_states(self):
        """Refresh the cache of file states."""
        if hasattr(self.sync_manager, "state_manager"):
            self._file_states = {
                fs.local_path: fs
                for fs in self.sync_manager.state_manager.get_all_file_states()
            }
    
    def read_file(self, file_path: str, start_line: Optional[int] = None, 
                  end_line: Optional[int] = None) -> str:
        """
        Read content from a file through Gemini API Files.
        
        For files accessed through Gemini API Files, we need to:
        1. Ensure the file is uploaded to Gemini
        2. Fall back to local file system for actual content reading
        
        Args:
            file_path: Relative or absolute path to the file
            start_line: Optional starting line number (0-based)
            end_line: Optional ending line number (0-based)
            
        Returns:
            The file content as a string
        """
        # Validate the path
        abs_path = self.validate_path(file_path)
        
        # Ensure the file is uploaded to Gemini
        path_obj = Path(abs_path)
        self.sync_manager.handle_file_change(path_obj)
        
        # Refresh file states to get the updated URI
        self._refresh_file_states()
        
        # Now read from local file system
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            if start_line is None and end_line is None:
                # Read the entire file
                return f.read()
            else:
                # Read specified lines
                lines = f.readlines()
                
                if start_line is None:
                    start_line = 0
                if end_line is None:
                    end_line = len(lines) - 1
                
                # Ensure line numbers are within bounds
                start_line = max(0, min(start_line, len(lines) - 1))
                end_line = max(0, min(end_line, len(lines) - 1))
                
                # Return the specified lines
                return "".join(lines[start_line : end_line + 1])
    
    def list_directory(self, dir_path: str) -> List[str]:
        """
        List files and directories.
        
        For Gemini files, we still need to scan the local filesystem,
        but we can indicate which files are synced with Gemini.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            List of file and directory names
        """
        abs_path = self.validate_path(dir_path)
        return os.listdir(abs_path)
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file exists, False otherwise
        """
        try:
            abs_path = self.validate_path(file_path)
            return Path(abs_path).exists()
        except ValueError:
            return False
    
    def is_directory(self, path: str) -> bool:
        """
        Check if a path refers to a directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is a directory, False otherwise
        """
        try:
            abs_path = self.validate_path(path)
            return Path(abs_path).is_dir()
        except ValueError:
            return False
    
    def get_file_info(self, path: str) -> FileInfo:
        """
        Get information about a file or directory.
        
        Args:
            path: Path to the file or directory
            
        Returns:
            FileInfo object with information about the file or directory
        """
        try:
            abs_path = self.validate_path(path)
            path_obj = Path(abs_path)
            
            exists = path_obj.exists()
            is_dir = path_obj.is_dir() if exists else False
            size = path_obj.stat().st_size if exists and not is_dir else 0
            
            # Determine the MIME type (simplified)
            mime_type = "text/plain"  # Default
            
            # Check if this file has been synced with Gemini
            uri = None
            rel_path = str(path_obj.relative_to(self.project_dir))
            if rel_path in self._file_states:
                uri = self._file_states[rel_path].gemini_uri
            
            return FileInfo(
                path=path,
                exists=exists,
                is_dir=is_dir,
                size=size,
                mime_type=mime_type,
                uri=uri
            )
        except ValueError:
            return FileInfo(
                path=path,
                exists=False,
                is_dir=False,
                size=0,
                mime_type="text/plain"
            )
    
    def validate_path(self, path: str) -> str:
        """
        Validate that a path is within the project directory.
        
        Args:
            path: Relative or absolute path
            
        Returns:
            Absolute path to the validated file or directory
            
        Raises:
            ValueError: If the path is outside the project directory
        """
        path_obj = Path(path)
        
        # Handle absolute paths
        if path_obj.is_absolute():
            abs_path = path_obj
        else:
            # Handle relative paths
            abs_path = (self.project_dir / path_obj).resolve()
            
        # Check if the path is within the project directory
        try:
            abs_path.relative_to(self.project_dir)
        except ValueError:
            raise ValueError(f"Path '{path}' is outside the project directory")
            
        return str(abs_path)


class FileAccessFactory:
    """Factory for creating file access layers."""
    
    @staticmethod
    def create_file_access_layer(project_dir: Path, gemini_sync_manager: Optional[Any] = None) -> FileAccessLayer:
        """
        Create an appropriate file access layer based on available components.
        
        Args:
            project_dir: Project directory path
            gemini_sync_manager: Optional Gemini sync manager
            
        Returns:
            FileAccessLayer implementation
        """
        if gemini_sync_manager is not None:
            return GeminiFileAccessLayer(gemini_sync_manager, project_dir)
        else:
            return LocalFileAccessLayer(project_dir)
