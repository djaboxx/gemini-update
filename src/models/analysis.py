"""
Models for codebase analysis and feature implementation planning.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class LanguageStats(BaseModel):
    """Statistics of programming languages used in the codebase."""
    # Use model instead of Dict[str, int] to avoid additionalProperties issue with Gemini
    language_name: str = Field(..., description="Name of the programming language")
    file_count: int = Field(..., description="Number of files using this language")


class AnalysisResult(BaseModel):
    """Result of a codebase analysis."""
    project_type: str = Field(..., description="Identified project type")
    primary_language: str = Field(..., description="Identified primary language")
    description: str = Field(..., description="Brief analysis summary")
    frameworks: List[str] = Field(default_factory=list, description="Identified frameworks")
    files_analyzed: int = Field(..., description="Number of files analyzed")
    language_stats: List[LanguageStats] = Field(default_factory=list, description="Statistics of languages used")

    def to_markdown(self) -> str:
        """Convert the analysis result to markdown."""
        markdown = "# Codebase Analysis Report\n\n"
        markdown += f"**Project Type:** {self.project_type}\n\n"
        markdown += f"**Primary Language:** {self.primary_language}\n\n"
        
        if self.frameworks:
            markdown += "**Frameworks:**\n\n"
            for framework in self.frameworks:
                markdown += f"- {framework}\n"
            markdown += "\n"
            
        markdown += f"**Files Analyzed:** {self.files_analyzed}\n\n"
        
        markdown += "**Language Statistics:**\n\n"
        for lang_stat in self.language_stats:
            markdown += f"- {lang_stat.language_name}: {lang_stat.file_count} files\n"
        markdown += "\n"
        
        markdown += f"**Summary:** {self.description}\n"
        
        return markdown


class CodebaseFile(BaseModel):
    """Representation of a file in the codebase."""
    
    path: str = Field(..., description="Relative path to the file")
    file_type: str = Field(..., description="Type of file (Python, JavaScript, etc.)")
    content: Optional[str] = Field(None, description="File content if loaded")
    

class CodeDependency(BaseModel):
    """Representation of a dependency between files."""
    
    source: str = Field(..., description="Source file path")
    target: str = Field(..., description="Target file path")
    dependency_type: str = Field(..., description="Type of dependency (import, inheritance, etc.)")
    

class FeatureScope(BaseModel):
    """Representation of what a feature needs to modify."""
    
    affected_files: List[str] = Field(default_factory=list, description="Files that need to be modified")
    new_files: List[str] = Field(default_factory=list, description="New files that need to be created")
    dependencies_needed: List[str] = Field(default_factory=list, description="New dependencies that need to be added")
    config_changes: List[str] = Field(default_factory=list, description="Configuration changes needed")


class ChangeType(str, Enum):
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"


class CodeChange(BaseModel):
    """Representation of a specific code change."""
    
    file_path: str = Field(..., description="Path to the file that needs changes")
    change_type: ChangeType = Field(..., description="Type of change (add, modify, delete)")
    description: str = Field(..., description="Description of what changes are needed")
    code_snippet: Optional[str] = Field(None, description="Suggested code snippet")
    line_range: Optional[str] = Field(None, description="Line range for the change")


class ImplementationPlan(BaseModel):
    """Complete plan for implementing a feature."""
    
    feature_name: str = Field(..., description="Name of the feature")
    description: str = Field(..., description="Description of the feature")
    scope: FeatureScope = Field(..., description="Scope of the feature implementation")
    changes: List[CodeChange] = Field(..., description="Specific code changes needed")
    estimated_complexity: str = Field(..., description="Estimated complexity (Low, Medium, High)")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies between changes")
    generated_at: datetime = Field(default_factory=datetime.now, description="When the plan was generated")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def to_markdown(self) -> str:
        """
        Convert the implementation plan to a well-formatted markdown document.
            
        Returns:
            The markdown string
        """
        # Build the markdown content
        markdown = f"# Implementation Plan for {self.feature_name}\n\n"
        
        # Add timestamp
        markdown += f"*Generated at: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        # Add description
        markdown += f"## Description\n\n{self.description}\n\n"
        
        # Add complexity
        markdown += f"**Estimated Complexity**: {self.estimated_complexity}\n\n"
        
        # Add scope section
        markdown += "## Scope\n\n"
        
        if self.scope.affected_files:
            markdown += "### Files to Modify\n\n"
            for file in self.scope.affected_files:
                markdown += f"- `{file}`\n"
            markdown += "\n"
            
        if self.scope.new_files:
            markdown += "### New Files to Create\n\n"
            for file in self.scope.new_files:
                markdown += f"- `{file}`\n"
            markdown += "\n"
            
        if self.scope.dependencies_needed:
            markdown += "### Dependencies to Add\n\n"
            for dep in self.scope.dependencies_needed:
                markdown += f"- `{dep}`\n"
            markdown += "\n"
            
        if self.scope.config_changes:
            markdown += "### Configuration Changes\n\n"
            for change in self.scope.config_changes:
                markdown += f"- {change}\n"
            markdown += "\n"
        
        # Add changes section
        markdown += "## Implementation Steps\n\n"
        for i, change in enumerate(self.changes, 1):
            markdown += f"### Step {i}: {change.change_type.value.capitalize()} {change.file_path}\n\n"
            markdown += f"**Description:** {change.description}\n\n"
            
            if change.code_snippet:
                markdown += "**Code:**\n\n```\n"
                markdown += change.code_snippet
                markdown += "\n```\n\n"
                
            if change.line_range:
                markdown += f"**Location:** Lines {change.line_range}\n\n"
                
        # Add dependencies section if any
        if self.dependencies:
            markdown += "## Dependencies Between Changes\n\n"
            for dep in self.dependencies:
                markdown += f"- {dep}\n"
            markdown += "\n"
        
        return markdown


class CodebaseContext(BaseModel):
    """Context model containing information about the codebase."""
    
    project_dir: Path = Field(..., description="Path to the project directory")
    files: Dict[str, CodebaseFile] = Field(default_factory=dict, description="Map of files in the project")
    dependencies: List[CodeDependency] = Field(default_factory=list, description="Dependencies between files")
    project_type: Optional[str] = Field(None, description="Type of project (web app, CLI tool, etc.)")
    primary_language: Optional[str] = Field(None, description="Primary programming language")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks used in the project")
    gemini_sync_manager: Optional[Any] = Field(None, description="Optional Gemini sync manager for file operations")
    file_access: Optional[Any] = Field(None, description="File access layer for operations")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize file access layer if not provided
        if not self.file_access:
            from src.models.file_access import FileAccessFactory
            self.file_access = FileAccessFactory.create_file_access_layer(
                self.project_dir, self.gemini_sync_manager
            )
    
    def validate_file_path(self, file_path: str) -> Path:
        """
        Validate that a file path is within the project directory.
        
        Args:
            file_path: Relative or absolute path to a file
            
        Returns:
            Absolute path to the validated file
            
        Raises:
            ValueError: If the file path is outside the project directory
        """
        if self.file_access:
            validated_path = self.file_access.validate_path(file_path)
            return Path(validated_path)
        
        path = Path(file_path)
        
        # Handle absolute paths
        if path.is_absolute():
            abs_path = path
        else:
            # Handle relative paths
            abs_path = (self.project_dir / path).resolve()
            
        # Check if the path is within the project directory
        try:
            abs_path.relative_to(self.project_dir)
        except ValueError:
            raise ValueError(f"File path '{file_path}' is outside the project directory")
            
        return abs_path


class Settings(BaseModel):
    """Application settings."""
    
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    output_dir: Path = Field(default=Path.cwd(), description="Directory for output files")
    max_files_to_analyze: int = Field(default=100, description="Maximum number of files to analyze")
    max_file_size_mb: float = Field(default=4.0, description="Maximum file size in MB for Gemini Files uploads")
    file_extensions: List[str] = Field(
        default_factory=lambda: [".py", ".js", ".ts", ".go", ".java", ".html", ".css", ".md", ".yaml", ".yml", ".json"],
        description="File extensions to include in analysis"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["__pycache__", "node_modules", ".git", ".venv", "venv", "env", "build", "dist"],
        description="Patterns to exclude from analysis"
    )
    use_gemini_files: bool = Field(default=False, description="Enable Gemini Files API for code analysis")
    gemini_model: Optional[str] = Field(default=None, description="Gemini model to use")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create Settings from environment variables."""
        import os
        from dotenv import load_dotenv
        
        # Load environment variables from .env file if present
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")
            
        output_dir = os.getenv("GEMINI_UPDATE_OUTPUT_DIR", str(Path.cwd()))
        max_files = int(os.getenv("GEMINI_UPDATE_MAX_FILES", "100"))
        max_file_size = float(os.getenv("GEMINI_UPDATE_MAX_FILE_SIZE", "4.0"))
        use_gemini_files = os.getenv("GEMINI_UPDATE_USE_FILES", "false").lower() == "true"
        gemini_model = os.getenv("GEMINI_MODEL")
        
        return cls(
            gemini_api_key=api_key,
            output_dir=Path(output_dir),
            max_files_to_analyze=max_files,
            max_file_size_mb=max_file_size,
            use_gemini_files=use_gemini_files,
            gemini_model=gemini_model
        )
