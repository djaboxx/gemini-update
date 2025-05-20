"""
Pydantic-AI Tools for interacting with the codebase.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import logging
import fnmatch
import ast
from functools import wraps

from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai import ModelRetry

from src.models import CodebaseContext


logger = logging.getLogger("gemini_update")


def register_tools(agent: Agent[CodebaseContext, str], max_retries: int = 1) -> None:
    """Register codebase interaction tools with the agent."""
    
    def retry_on_error(func):
        """Decorator to add retry logic to tools."""
        @wraps(func)  # Preserve the original function's metadata
        async def wrapped_tool(*args, **kwargs):
            retries = 0
            last_error = None
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    retries += 1
                    if retries <= max_retries:
                        logger.warning(f"Tool {func.__name__} failed, attempt {retries}/{max_retries}: {str(e)}")
                        continue
                    raise ModelRetry(f"Error in {func.__name__}: {str(e)}")
            raise last_error
        return wrapped_tool

    @retry_on_error
    @agent.tool
    async def read_file(
        ctx: RunContext[CodebaseContext],
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """
        Read content from a file within the project directory.

        Args:
            ctx: The run context containing project filesystem access
            file_path: Relative or absolute path to the file within the project
            start_line: Optional starting line number (0-based)
            end_line: Optional ending line number (0-based)

        Returns:
            The file content as a string
        """
        try:
            # Use file access layer if available
            if ctx.deps.file_access:
                return ctx.deps.file_access.read_file(file_path, start_line, end_line)
                
            # Legacy implementation as fallback
            # Validate the file path is within the project
            abs_path = ctx.deps.validate_file_path(file_path)

            # Read the file
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

        except (FileNotFoundError, ValueError, PermissionError) as e:
            raise ModelRetry(f"Error reading file: {str(e)}")

    @retry_on_error
    @agent.tool
    async def list_directory(ctx: RunContext[CodebaseContext], dir_path: str) -> List[str]:
        """
        List files and directories within a directory in the project.

        Args:
            ctx: The run context containing project filesystem access
            dir_path: Path to the directory within the project

        Returns:
            List of file and directory names
        """
        try:
            # Use file access layer if available
            if ctx.deps.file_access:
                return ctx.deps.file_access.list_directory(dir_path)
                
            # Legacy implementation as fallback
            # Validate the directory path is within the project
            abs_path = ctx.deps.validate_file_path(dir_path)

            # Check if the path is a directory
            if not os.path.isdir(abs_path):
                raise ValueError(f"Path '{dir_path}' is not a directory")

            # List the directory
            return os.listdir(abs_path)

        except (FileNotFoundError, ValueError, PermissionError) as e:
            raise ModelRetry(f"Error listing directory: {str(e)}")

    @retry_on_error
    @agent.tool
    async def find_files(
        ctx: RunContext[CodebaseContext],
        pattern: str,
        base_dir: Optional[str] = None,
        recursive: bool = True,
    ) -> List[str]:
        """
        Find files matching a pattern within the project directory.

        Args:
            ctx: The run context containing project filesystem access
            pattern: Glob pattern to match files (e.g., "*.py", "**/*.js")
            base_dir: Optional base directory to start the search (relative to project root)
            recursive: Whether to search recursively (default: True)

        Returns:
            List of files matching the pattern
        """
        try:
            # Determine the base directory
            if base_dir:
                base_path = ctx.deps.validate_file_path(base_dir)
                if not Path(base_path).is_dir():
                    raise ModelRetry(f"Not a directory: {base_dir}")
            else:
                base_path = ctx.deps.project_dir

            # Find files matching the pattern
            result = []
            base_path_obj = Path(base_path)

            if recursive:
                walk_iter = os.walk(base_path)
            else:
                # If not recursive, only search the top-level directory
                walk_iter = [(base_path, [], os.listdir(base_path))]

            for root, _, files in walk_iter:
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(base_path_obj)
                    
                    # Check if the file matches the pattern
                    if fnmatch.fnmatch(str(rel_path), pattern):
                        result.append(str(rel_path))

            return result

        except (FileNotFoundError, ValueError, PermissionError) as e:
            raise ModelRetry(f"Error finding files: {str(e)}")

    @retry_on_error
    @agent.tool
    async def search_code(
        ctx: RunContext[CodebaseContext],
        query: str,
        file_patterns: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for a string pattern in code files.

        Args:
            ctx: The run context containing project filesystem access
            query: String or regex pattern to search for
            file_patterns: Optional list of glob patterns to restrict the search
            case_sensitive: Whether the search should be case-sensitive

        Returns:
            List of matches with file paths, line numbers, and matching lines
        """
        try:
            project_dir = ctx.deps.project_dir
            result = []

            # Compile regex pattern
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(query, flags)
            except re.error:
                # If not a valid regex, search for the literal text
                pattern = re.compile(re.escape(query), flags)

            # Get files to search
            files_to_search = []
            if file_patterns:
                for file_pattern in file_patterns:
                    pattern_files = await find_files(ctx, file_pattern)
                    files_to_search.extend(pattern_files)
            else:
                # Default to common code file extensions
                for ext in [".py", ".js", ".ts", ".java", ".go", ".c", ".cpp", ".h", ".hpp", ".cs", ".html", ".css"]:
                    pattern_files = await find_files(ctx, f"**/*{ext}")
                    files_to_search.extend(pattern_files)

            # Deduplicate files
            files_to_search = list(set(files_to_search))

            # Search each file
            for file_path in files_to_search:
                abs_path = project_dir / file_path
                try:
                    # Read the file content
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()

                    # Search for matches
                    for i, line in enumerate(lines):
                        if pattern.search(line):
                            result.append({
                                "file": str(file_path),
                                "line_number": i + 1,  # 1-based line number
                                "line": line.rstrip()
                            })
                except (UnicodeDecodeError, IOError):
                    # Skip files that can't be read as text
                    continue

            return result

        except Exception as e:
            raise ModelRetry(f"Error searching code: {str(e)}")

    @retry_on_error
    @agent.tool
    async def analyze_imports(ctx: RunContext[CodebaseContext], file_path: str) -> Dict[str, Any]:
        """
        Analyze imports and dependencies in a Python file.

        Args:
            ctx: The run context containing project filesystem access
            file_path: Relative or absolute path to the Python file

        Returns:
            Dictionary with import information
        """
        try:
            # Validate the file path is within the project
            abs_path = ctx.deps.validate_file_path(file_path)

            if not str(abs_path).endswith(".py"):
                raise ModelRetry(f"Not a Python file: {file_path}")

            # Read the file
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except Exception as e:
                raise ModelRetry(f"Error reading file: {str(e)}")

            # Parse the AST
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                raise ModelRetry(f"Syntax error in {file_path}: {str(e)}")

            # Analyze imports
            imports = []
            from_imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append({"name": name.name, "alias": name.asname})
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        from_imports.append({
                            "module": module,
                            "name": name.name,
                            "alias": name.asname
                        })

            return {
                "file": file_path,
                "imports": imports,
                "from_imports": from_imports
            }

        except Exception as e:
            raise ModelRetry(f"Error analyzing imports: {str(e)}")

    @retry_on_error
    @agent.tool
    async def get_project_info(ctx: RunContext[CodebaseContext]) -> Dict[str, Any]:
        """
        Get high-level information about the project.

        Args:
            ctx: The run context containing project filesystem access

        Returns:
            Dictionary with project information
        """
        try:
            project_dir = ctx.deps.project_dir
            result = {
                "project_dir": str(project_dir),
                "file_count": 0,
                "language_stats": {},
                "readme": None,
                "package_files": []
            }

            # Find common project files
            common_files = {
                "readme": ["README.md", "readme.md", "README.txt", "readme.txt"],
                "package_json": ["package.json"],
                "requirements": ["requirements.txt"],
                "setup_py": ["setup.py"],
                "pyproject_toml": ["pyproject.toml"],
                "cargo_toml": ["Cargo.toml"],
                "go_mod": ["go.mod"],
                "composer_json": ["composer.json"]
            }

            # Check for common files
            for file_type, file_names in common_files.items():
                for file_name in file_names:
                    if (project_dir / file_name).exists():
                        with open(project_dir / file_name, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        if file_type == "readme":
                            result["readme"] = content
                        else:
                            result["package_files"].append({
                                "type": file_type,
                                "path": file_name,
                                "content": content
                            })

            # Count files by extension
            extension_count = {}
            for root, _, files in os.walk(project_dir):
                for file in files:
                    result["file_count"] += 1
                    _, ext = os.path.splitext(file)
                    if ext:
                        ext = ext.lower()
                        extension_count[ext] = extension_count.get(ext, 0) + 1

            # Map extensions to languages
            extension_to_language = {
                ".py": "Python",
                ".js": "JavaScript",
                ".ts": "TypeScript",
                ".jsx": "React",
                ".tsx": "React/TypeScript",
                ".java": "Java",
                ".go": "Go",
                ".rb": "Ruby",
                ".php": "PHP",
                ".c": "C",
                ".cpp": "C++",
                ".cs": "C#",
                ".html": "HTML",
                ".css": "CSS",
                ".scss": "SCSS",
                ".less": "LESS",
                ".rs": "Rust",
                ".swift": "Swift",
                ".kt": "Kotlin",
                ".kts": "Kotlin",
                ".dart": "Dart",
                ".sh": "Shell",
                ".bash": "Shell",
                ".zsh": "Shell",
                ".md": "Markdown",
                ".json": "JSON",
                ".yml": "YAML",
                ".yaml": "YAML",
                ".toml": "TOML",
                ".xml": "XML",
                ".sql": "SQL"
            }

            # Compute language statistics
            for ext, count in extension_count.items():
                lang = extension_to_language.get(ext, "Other")
                if lang in result["language_stats"]:
                    result["language_stats"][lang] += count
                else:
                    result["language_stats"][lang] = count

            # Sort languages by file count
            result["language_stats"] = dict(
                sorted(result["language_stats"].items(), key=lambda x: x[1], reverse=True)
            )

            return result

        except Exception as e:
            raise ModelRetry(f"Error getting project info: {str(e)}")

    @retry_on_error
    @agent.tool
    async def analyze_codebase(ctx: RunContext[CodebaseContext]) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of the codebase.

        Args:
            ctx: The run context containing project filesystem access

        Returns:
            Dictionary with analysis results
        """
        try:
            # Get project info
            project_info = await get_project_info(ctx)
            
            # Determine primary language
            primary_language = next(iter(project_info["language_stats"]), "Unknown")
            
            # Identify framework based on package files
            framework = "Unknown"
            frameworks = []
            
            for pkg_file in project_info["package_files"]:
                if pkg_file["type"] == "package_json":
                    import json
                    try:
                        pkg_data = json.loads(pkg_file["content"])
                        dependencies = {
                            **pkg_data.get("dependencies", {}),
                            **pkg_data.get("devDependencies", {})
                        }
                        
                        # Check for common JavaScript frameworks
                        if "react" in dependencies:
                            frameworks.append("React")
                        if "vue" in dependencies:
                            frameworks.append("Vue.js")
                        if "angular" in dependencies or "@angular/core" in dependencies:
                            frameworks.append("Angular")
                        if "express" in dependencies:
                            frameworks.append("Express")
                        if "next" in dependencies:
                            frameworks.append("Next.js")
                    except json.JSONDecodeError:
                        pass
                        
                elif pkg_file["type"] in ["requirements", "setup_py", "pyproject_toml"]:
                    content = pkg_file["content"]
                    
                    # Check for common Python frameworks
                    if "django" in content.lower():
                        frameworks.append("Django")
                    if "flask" in content.lower():
                        frameworks.append("Flask") 
                    if "fastapi" in content.lower():
                        frameworks.append("FastAPI")
                    if "tornado" in content.lower():
                        frameworks.append("Tornado")
            
            # Determine project type
            project_type = "Unknown"
            if "package.json" in [f["path"] for f in project_info["package_files"]]:
                if any(f in frameworks for f in ["React", "Vue.js", "Angular", "Next.js"]):
                    project_type = "Web Frontend"
                elif "Express" in frameworks:
                    project_type = "Web Backend"
                else:
                    project_type = "Node.js Project"
            elif any(f["path"] in ["requirements.txt", "setup.py", "pyproject.toml"] for f in project_info["package_files"]):
                if any(f in frameworks for f in ["Django", "Flask", "FastAPI"]):
                    project_type = "Web Backend"
                else:
                    project_type = "Python Project"
            elif "go.mod" in [f["path"] for f in project_info["package_files"]]:
                project_type = "Go Project"
                
            # Find common project directories
            dirs = []
            common_dirs = ["src", "lib", "app", "api", "client", "server", "test", "tests", "docs"]
            for d in common_dirs:
                if (ctx.deps.project_dir / d).is_dir():
                    dirs.append(d)
                    
            return {
                "project_type": project_type,
                "primary_language": primary_language,
                "frameworks": frameworks,
                "directories": dirs,
                "file_count": project_info["file_count"],
                "language_stats": project_info["language_stats"]
            }
            
        except Exception as e:
            raise ModelRetry(f"Error analyzing codebase: {str(e)}")

    @retry_on_error
    @agent.tool
    async def file_exists(
        ctx: RunContext[CodebaseContext],
        file_path: str,
    ) -> bool:
        """
        Check if a file exists within the project directory.

        Args:
            ctx: The run context containing project filesystem access
            file_path: Relative or absolute path to the file within the project

        Returns:
            True if the file exists, False otherwise
        """
        try:
            # Use file access layer if available
            if ctx.deps.file_access:
                return ctx.deps.file_access.file_exists(file_path)
            
            # Legacy implementation as fallback
            try:
                abs_path = ctx.deps.validate_file_path(file_path)
                return os.path.exists(abs_path)
            except ValueError:
                return False
        except Exception as e:
            logger.warning(f"Error checking if file exists: {str(e)}")
            return False
