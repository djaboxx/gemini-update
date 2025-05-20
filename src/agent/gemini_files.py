"""
Gemini Files API integration module.
"""

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union

import google.generativeai as genai

from src.models import AgentConfig

logger = logging.getLogger("gemini_update")


class GeminiFileState:
    """State of a file synchronized with Gemini."""
    
    def __init__(
        self,
        local_path: str,
        gemini_uri: str,
        content_hash: str,
        sync_time: float,
    ):
        """Initialize a file state."""
        self.local_path = local_path
        self.gemini_uri = gemini_uri
        self.content_hash = content_hash
        self.sync_time = sync_time


class GeminiFileManager:
    """Manager for interacting with Gemini Files API."""
    
    def __init__(self, api_key: str, project_dir: Union[str, Path], max_file_size_mb: float = 4.0):
        """
        Initialize the Gemini file manager.
        
        Args:
            api_key: Gemini API key
            project_dir: Base directory for relative file paths
            max_file_size_mb: Maximum file size in MB (default: 4MB)
        """
        self.project_dir = Path(project_dir).resolve()
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.file_states: Dict[str, GeminiFileState] = {}
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute a hash of the file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash of the file content
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {str(e)}")
            return ""
    
    def _get_mime_type(self, file_path: Path) -> str:
        """
        Determine the MIME type of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        import mimetypes
        
        # Initialize mimetype detection
        if not mimetypes.inited:
            mimetypes.init()
        
        # Add common development file types
        dev_mime_types = {
            ".py": "text/x-python",
            ".java": "text/x-java",
            ".js": "application/javascript",
            ".ts": "application/typescript",
            ".html": "text/html",
            ".css": "text/css",
            ".json": "application/json",
            ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml",
            ".md": "text/markdown",
        }
        
        # Try to determine MIME type from extension
        ext = file_path.suffix.lower()
        if ext in dev_mime_types:
            return dev_mime_types[ext]
        
        # Fall back to Python's mimetype detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # If Python couldn't determine the type, check if it's likely a text file
        if mime_type is None:
            try:
                with open(file_path, "rb") as f:
                    content = f.read(1024)
                    # Try to decode as UTF-8 to check if it's text
                    content.decode("utf-8")
                    return "text/plain"
            except (UnicodeDecodeError, IOError):
                # If we can't decode as UTF-8, it's likely binary
                pass
        
        # Return the detected MIME type or default
        return mime_type or "application/octet-stream"
    
    def upload_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Upload a file to Gemini Files API.
        
        Args:
            file_path: Path to the file (absolute or relative to project_dir)
            
        Returns:
            URI of the uploaded file, or None if upload failed
        """
        try:
            # Normalize the file path
            path_obj = Path(file_path)
            if not path_obj.is_absolute():
                path_obj = (self.project_dir / path_obj).resolve()
            
            # Check if the file exists
            if not path_obj.exists() or not path_obj.is_file():
                logger.warning(f"Cannot upload non-existent file: {path_obj}")
                return None
            
            # Check file size
            file_size = path_obj.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.warning(
                    f"File too large to upload: {path_obj} "
                    f"({file_size / (1024 * 1024):.2f} MB > {self.max_file_size_bytes / (1024 * 1024):.2f} MB)"
                )
                return None
            
            # Compute file hash
            content_hash = self._compute_file_hash(path_obj)
            if not content_hash:
                logger.warning(f"Failed to compute hash for {path_obj}")
                return None
            
            # Check if the file has changed since the last upload
            rel_path = str(path_obj.relative_to(self.project_dir))
            if rel_path in self.file_states and self.file_states[rel_path].content_hash == content_hash:
                logger.debug(f"File unchanged since last upload: {path_obj}")
                return self.file_states[rel_path].gemini_uri
            
            # Determine MIME type
            mime_type = self._get_mime_type(path_obj)
            
            # Upload the file
            try:
                response = genai.upload_file(path=path_obj, mime_type=mime_type)
                file_uri = response.uri
                
                # Update the state
                self.file_states[rel_path] = GeminiFileState(
                    local_path=rel_path,
                    gemini_uri=file_uri,
                    content_hash=content_hash,
                    sync_time=time.time(),
                )
                
                logger.info(f"Uploaded {path_obj} to Gemini: {file_uri}")
                return file_uri
            except Exception as e:
                logger.error(f"Error uploading {path_obj} to Gemini: {str(e)}")
                return None
            
        except Exception as e:
            logger.exception(f"Unexpected error uploading {file_path}: {str(e)}")
            return None
    
    def delete_file(self, file_uri: str) -> bool:
        """
        Delete a file from Gemini Files API.
        
        Args:
            file_uri: URI of the file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Extract the file ID from the URI
            file_id = file_uri.split("/")[-1]
            
            # Delete the file
            genai.delete_file(name=file_id)
            
            # Find and remove from state
            for key, state in list(self.file_states.items()):
                if state.gemini_uri == file_uri:
                    del self.file_states[key]
            
            logger.info(f"Deleted file from Gemini: {file_uri}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from Gemini: {str(e)}")
            return False
    
    def get_file_uri(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Get the Gemini URI for a file if it has been uploaded.
        
        Args:
            file_path: Path to the file
            
        Returns:
            URI of the file in Gemini, or None if not uploaded
        """
        # Normalize the file path
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = (self.project_dir / path_obj).resolve()
        
        # Convert to relative path
        try:
            rel_path = str(path_obj.relative_to(self.project_dir))
            if rel_path in self.file_states:
                return self.file_states[rel_path].gemini_uri
        except ValueError:
            # File is outside project directory
            pass
        
        return None
    
    def clear_all_files(self) -> int:
        """
        Delete all uploaded files from Gemini and clear the state.
        
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        
        # Delete all files
        for state in list(self.file_states.values()):
            try:
                self.delete_file(state.gemini_uri)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {state.local_path} from Gemini: {str(e)}")
        
        # Clear the state
        self.file_states.clear()
        
        logger.info(f"Cleared {deleted_count} files from Gemini")
        return deleted_count
