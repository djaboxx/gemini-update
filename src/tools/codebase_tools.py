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

from src.models import CodebaseContext, AnalysisResult, LanguageStats


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
        
    # Helper functions to avoid tools calling other tools
    def _get_project_info_impl(project_dir: Path) -> Dict[str, Any]:
        """
        Implementation of project info gathering logic.
        
        Args:
            project_dir: Path to the project directory
            
        Returns:
            Dictionary with project information
        """
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
            "composer_json": ["composer.json"],
            "terraform": ["main.tf", "variables.tf", "outputs.tf"],
            "dockerfile": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
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
            ".tf": "Terraform",
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
            return _get_project_info_impl(project_dir)
        except Exception as e:
            raise ModelRetry(f"Error getting project info: {str(e)}")

    @retry_on_error
    @agent.tool
    async def execute_code_analysis(
        ctx: RunContext[CodebaseContext],
        analysis_code: str,
        description: str = "Analyzing codebase..."
    ) -> Dict[str, Any]:
        """
        Execute custom Python code to analyze the codebase.
        
        This tool allows Gemini to write and execute Python code that inspects the project structure,
        analyzes imports, and detects patterns to determine project type and characteristics.
        
        The code will be executed in a context where it has access to:
        - project_dir: Path to the project directory
        - os, re, json, pathlib: Common Python modules
        - find_imports: Helper function to analyze imports in Python files
        - collect_file_samples: Helper function to get content from key files

        Args:
            ctx: The run context containing project filesystem access
            analysis_code: Python code to execute for analysis
            description: Description of what the analysis is doing
            
        Returns:
            Dictionary with analysis results
        """
        try:
            import json
            import os
            import re
            import tempfile
            import importlib.util
            from pathlib import Path
            import subprocess
            
            # Create a temporary directory for the analysis script
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create the analysis script
                script_path = Path(temp_dir) / "codebase_analysis.py"
                
                # Create the analysis script with common utilities and the provided code
                script_content = f'''
import os
import re
import json
import ast
import importlib
import fnmatch
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional

# Set up project directory
project_dir = Path(r"{ctx.deps.project_dir}")
result = {{}}

def find_imports(file_path):
    """Analyze imports in a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({{"name": name.name, "alias": name.asname}})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append({{"module": module, "name": name.name, "alias": name.asname}})
        return imports
    except Exception as e:
        return {{"error": str(e)}}

def collect_file_samples(extensions=None, max_samples_per_ext=5, max_lines=200):
    """Collect samples of files with specific extensions."""
    if extensions is None:
        extensions = {{".py", ".js", ".ts", ".html", ".css", ".go", ".java", ".md"}}
    
    samples = {{}}
    for ext in extensions:
        samples[ext] = []
        count = 0
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(ext) and count < max_samples_per_ext:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = "".join(f.readlines()[:max_lines])
                        rel_path = os.path.relpath(file_path, project_dir)
                        samples[ext].append({{"path": rel_path, "content": content}})
                        count += 1
                    except:
                        pass
    return samples

def find_files(pattern, base_dir=None, recursive=True):
    """Find files matching a pattern."""
    base_path = project_dir if base_dir is None else project_dir / base_dir
    result = []
    
    if recursive:
        for root, _, files in os.walk(base_path):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(project_dir)
                if fnmatch.fnmatch(str(rel_path), pattern):
                    result.append(str(rel_path))
    else:
        for item in base_path.iterdir():
            if item.is_file() and fnmatch.fnmatch(item.name, pattern):
                result.append(str(item.relative_to(project_dir)))
    
    return result

# User's analysis code starts here
try:
{analysis_code.replace("\\", "\\\\").replace("{", "{{").replace("}", "}}").rstrip().replace("\n", "\n    ")}
except Exception as e:
    result["error"] = str(e)

# Print the result as JSON
print(json.dumps(result, default=str))
'''

                with open(script_path, "w") as f:
                    f.write(script_content)
                
                # Execute the script and capture the output
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(ctx.deps.project_dir)
                )
                
                if proc.returncode != 0:
                    error_msg = proc.stderr.strip() or f"Process exited with code {proc.returncode}"
                    raise ModelRetry(f"Error executing analysis code: {error_msg}")
                
                # Parse the JSON output
                try:
                    return json.loads(proc.stdout)
                except json.JSONDecodeError as e:
                    raise ModelRetry(f"Error parsing analysis result: {str(e)}\nOutput: {proc.stdout}")
                
        except Exception as e:
            raise ModelRetry(f"Error executing code analysis: {str(e)}")
    
    @retry_on_error
    @agent.tool
    async def analyze_codebase(ctx: RunContext[CodebaseContext], include_dependencies: bool = False) -> AnalysisResult:
        """
        Perform a comprehensive analysis of the codebase.

        Args:
            ctx: The run context containing project filesystem access
            include_dependencies: Whether to include detailed dependency information

        Returns:
            AnalysisResult containing the analysis
        """
        try:
            # First collect basic project info
            project_info = _get_project_info_impl(ctx.deps.project_dir)
            
            # Use the execute_code_analysis tool to run a more sophisticated analysis
            analysis_code = """
# Detect project structure and characteristics
import os
import json
from pathlib import Path
import re

# Initialize result structure
result = {
    "project_type": "Unknown",
    "primary_language": None,
    "frameworks": [],
    "description": "",
    "key_files": [],
    "key_directories": [],
    "has_tests": False,
    "has_documentation": False,
    "entry_points": [],
    "code_patterns": {}
}

# Add the language statistics to the result
language_stats = {}
extension_count = {}
for root, _, files in os.walk(project_dir):
    for file in files:
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
    ".tf": "Terraform",
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
    if lang in language_stats:
        language_stats[lang] += count
    else:
        language_stats[lang] = count

# Sort languages by file count
result["language_stats"] = dict(sorted(language_stats.items(), key=lambda x: x[1], reverse=True))

# Determine primary language
if result["language_stats"]:
    result["primary_language"] = next(iter(result["language_stats"]), "Unknown")

# Look for common project files
package_json_path = project_dir / "package.json"
requirements_txt_path = project_dir / "requirements.txt"
setup_py_path = project_dir / "setup.py"
pyproject_toml_path = project_dir / "pyproject.toml" 
go_mod_path = project_dir / "go.mod"
cargo_toml_path = project_dir / "Cargo.toml"
composer_json_path = project_dir / "composer.json"
build_gradle_path = project_dir / "build.gradle"
pom_xml_path = project_dir / "pom.xml"
gemfile_path = project_dir / "Gemfile"
dockerfile_path = project_dir / "Dockerfile"
docker_compose_path = project_dir / "docker-compose.yml"
docker_compose_yaml_path = project_dir / "docker-compose.yaml"
makefile_path = project_dir / "Makefile"

# Check for config files
config_files = []
for file in ["config.json", "config.yaml", "config.yml", ".env", ".env.example", ".gitignore", 
             "tsconfig.json", "tox.ini", "pytest.ini", "jest.config.js", ".eslintrc"]:
    if (project_dir / file).exists():
        config_files.append(file)
result["config_files"] = config_files

# Check for common directories
directories = []
important_dirs = []
for d in os.listdir(project_dir):
    if os.path.isdir(os.path.join(project_dir, d)) and not d.startswith('.') and d not in ["node_modules", "__pycache__", "venv", "env", ".venv"]:
        directories.append(d)
        # Mark important directories
        if d.lower() in ["src", "app", "lib", "server", "client", "api", "test", "tests", "docs", "bin", "cmd", "pkg", "internal"]:
            important_dirs.append(d)

result["directories"] = directories
result["key_directories"] = important_dirs

# Look for test directories and files
result["has_tests"] = any(d.lower() in ["test", "tests", "__tests__"] for d in directories) or \
                       bool(find_files("test_*.py")) or \
                       bool(find_files("*.test.js")) or \
                       bool(find_files("*.spec.js"))

# Look for documentation
result["has_documentation"] = any(d.lower() in ["docs", "documentation"] for d in directories) or \
                               any((project_dir / f).exists() for f in ["README.md", "readme.md", "docs.md", "API.md"])

# Detect Python frameworks and libraries
if result["primary_language"] == "Python":
    python_frameworks = []
    library_imports = set()
    cli_patterns = False
    api_patterns = False
    agent_patterns = False
    data_science_patterns = False
    
    # Check for Python framework patterns in imports
    python_files = find_files("**/*.py")
    for file_path in python_files[:30]:  # Limit to prevent analyzing too many files
        try:
            imports = find_imports(project_dir / file_path)
            for imp in imports:
                import_name = imp.get("name", "")
                module_name = imp.get("module", "")
                
                library = import_name.split(".")[0] if "." in import_name else import_name
                if module_name:
                    library = module_name.split(".")[0] 
                
                if library:
                    library_imports.add(library)
                
                # Web frameworks
                if any(fw in (import_name, module_name) for fw in ["django", "flask", "fastapi", "tornado", "bottle", "pyramid"]):
                    if fw := next((fw for fw in ["django", "flask", "fastapi", "tornado", "bottle", "pyramid"] 
                                  if fw in (import_name, module_name)), None):
                        python_frameworks.append(fw.capitalize())
                
                # CLI patterns
                if any(cli in (import_name, module_name) for cli in ["click", "typer", "argparse", "docopt", "fire"]):
                    cli_patterns = True
                
                # API patterns 
                if any(api in (import_name, module_name) for api in ["requests", "httpx", "aiohttp", "urllib3"]):
                    api_patterns = True
                
                # Data science patterns
                if any(ds in (import_name, module_name) for ds in ["pandas", "numpy", "scipy", "scikit-learn", 
                                                               "matplotlib", "tensorflow", "torch", "keras"]):
                    data_science_patterns = True
                    
                # Agent patterns
                if any(agent in (import_name, module_name, library) for agent in ["openai", "langchain", "llama", "gemini", "anthropic", 
                                                                            "vertexai", "pydantic_ai", "semantic_kernel"]):
                    agent_patterns = True
                    
            # Check for specific code patterns in content
            try:
                with open(project_dir / file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    if "def main" in content:
                        result["entry_points"].append(file_path)
                    if "@app.route" in content or "class APIView" in content or "@api_view" in content:
                        api_patterns = True
                    if "__name__ == '__main__'" in content:
                        result["entry_points"].append(file_path)
            except:
                pass
        except:
            continue
    
    # Update frameworks
    result["frameworks"] = python_frameworks
    result["library_imports"] = list(library_imports)
    
    # Determine Python project type
    if python_frameworks:
        result["project_type"] = "Web Backend"
    elif data_science_patterns:
        result["project_type"] = "Data Science"
    elif agent_patterns:
        result["project_type"] = "AI Agent"
    elif cli_patterns:
        result["project_type"] = "Command Line Tool"
    elif api_patterns:
        result["project_type"] = "API Client/Server"
    else:
        result["project_type"] = "Python Library/Application"
    
    # Check for setup tools to determine if it's a library
    for file in [setup_py_path, pyproject_toml_path]:
        if file.exists():
            try:
                with open(file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().lower()
                    if "setup(" in content or "setuptools" in content or "poetry" in content:
                        if "project_type" not in result or result["project_type"] == "Python Library/Application":
                            result["project_type"] = "Python Package"
            except:
                pass

# Detect JavaScript/TypeScript frameworks
elif result["primary_language"] in ["JavaScript", "TypeScript"]:
    js_frameworks = []
    
    # Read package.json
    if package_json_path.exists():
        try:
            with open(package_json_path, "r", encoding="utf-8") as f:
                package_data = json.loads(f.read())
            
            dependencies = {
                **package_data.get("dependencies", {}),
                **package_data.get("devDependencies", {})
            }
            
            # Check for common JavaScript frameworks
            if "react" in dependencies:
                js_frameworks.append("React")
            if "vue" in dependencies:
                js_frameworks.append("Vue.js")
            if "angular" in dependencies or "@angular/core" in dependencies:
                js_frameworks.append("Angular")
            if "express" in dependencies:
                js_frameworks.append("Express")
            if "next" in dependencies:
                js_frameworks.append("Next.js")
            if "gatsby" in dependencies:
                js_frameworks.append("Gatsby")
            if "nuxt" in dependencies:
                js_frameworks.append("Nuxt.js")
            if "electron" in dependencies:
                js_frameworks.append("Electron")
            
            # Check for type
            result["frameworks"] = js_frameworks
            
            # Check entry points
            if "main" in package_data:
                result["entry_points"].append(package_data["main"])
                
            # Determine JS/TS project type
            if any(fw in js_frameworks for fw in ["React", "Vue.js", "Angular", "Next.js", "Gatsby", "Nuxt.js"]):
                result["project_type"] = "Web Frontend"
            elif "Express" in js_frameworks:
                result["project_type"] = "Web Backend"
            elif "Electron" in js_frameworks:
                result["project_type"] = "Desktop Application"
            elif "react-native" in dependencies:
                result["project_type"] = "Mobile Application"
            elif "type": "module" in package_data:
                result["project_type"] = "JavaScript/TypeScript Package"
            else:
                result["project_type"] = "JavaScript/TypeScript Application"
        except:
            result["project_type"] = "JavaScript/TypeScript Application"

# Detect Go projects
elif result["primary_language"] == "Go":
    if go_mod_path.exists():
        result["project_type"] = "Go Project"
        # Try to determine if it's a library or application by looking for main packages
        main_files = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(".go"):
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            content = f.read()
                            if "package main" in content and "func main" in content:
                                main_files.append(os.path.relpath(os.path.join(root, file), project_dir))
                    except:
                        continue
        
        if main_files:
            result["entry_points"] = main_files
            result["project_type"] = "Go Application"
        else:
            result["project_type"] = "Go Library"

# Generate summary description
if result["project_type"] != "Unknown" and result["primary_language"]:
    description = f"A {result['project_type'].lower()} written primarily in {result['primary_language']}"
    if result["frameworks"]:
        description += f" using {', '.join(result['frameworks'])}"
    description += "."
    result["description"] = description

# Check for Docker to determine if it's containerized
if dockerfile_path.exists() or docker_compose_path.exists() or docker_compose_yaml_path.exists():
    result["is_containerized"] = True
    result["description"] += " The application is containerized using Docker."
else:
    result["is_containerized"] = False

# Detect repository type
if (project_dir / ".git").exists():
    result["has_git"] = True
else:
    result["has_git"] = False
"""            
            analysis_result = await execute_code_analysis(ctx, analysis_code)
            
            # Handle errors in execution
            if "error" in analysis_result:
                logger.warning(f"Error in custom analysis code: {analysis_result['error']}")
                # Fall back to basic analysis
                return await self._legacy_analyze_codebase(ctx, project_info)
            
            # Use the AI analysis results to create the AnalysisResult object
            language_stats_list = []
            for lang_name, file_count in analysis_result.get("language_stats", {}).items():
                language_stats_list.append(LanguageStats(
                    language_name=lang_name,
                    file_count=file_count
                ))
            
            return AnalysisResult(
                project_type=analysis_result.get("project_type", "Unknown"),
                primary_language=analysis_result.get("primary_language", "Unknown"),
                description=analysis_result.get("description", ""),
                frameworks=analysis_result.get("frameworks", []),
                files_analyzed=project_info["file_count"],
                language_stats=language_stats_list
            )
            
        except Exception as e:
            raise ModelRetry(f"Error analyzing codebase: {str(e)}")
    
    async def _legacy_analyze_codebase(self, ctx: RunContext[CodebaseContext], project_info: Dict[str, Any]) -> AnalysisResult:
        """Legacy implementation of codebase analysis as a fallback."""
        # Determine primary language and frameworks
        primary_language = next(iter(project_info["language_stats"]), "Unknown")
        frameworks = []
        
        # Extract frameworks from package files
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
                
        # Generate summary description
        description = f"A {project_type.lower()} written primarily in {primary_language}"
        if frameworks:
            description += f" using {', '.join(frameworks)}"
        description += "."
            
        # Convert dictionary-based language stats to a list of LanguageStats objects
        language_stats_list = []
        for lang_name, file_count in project_info["language_stats"].items():
            language_stats_list.append(LanguageStats(
                language_name=lang_name,
                file_count=file_count
            ))
        
        # Create the AnalysisResult
        return AnalysisResult(
            project_type=project_type,
            primary_language=primary_language,
            description=description,
            frameworks=frameworks,
            files_analyzed=project_info["file_count"],
            language_stats=language_stats_list
        )

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

    @retry_on_error
    @agent.tool
    async def custom_codebase_analysis(
        ctx: RunContext[CodebaseContext],
        analysis_goal: str,
        analysis_code: str
    ) -> Dict[str, Any]:
        """
        Execute highly customized Python code for specific codebase analysis goals.
        
        This advanced tool is meant for targeted analysis of specific aspects of the codebase.
        Unlike execute_code_analysis, which has a predetermined template, this tool allows for
        completely custom code to be executed for specialized analysis tasks.
        
        Some example uses:
        - Detecting architectural patterns (MVC, microservices, etc.)
        - Finding circular dependencies
        - Analyzing inheritance hierarchies
        - Identifying code smells or anti-patterns
        - Extracting domain-specific information
        
        Args:
            ctx: The run context containing project filesystem access
            analysis_goal: A clear description of what the analysis is trying to determine
            analysis_code: Custom Python code to execute for the analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # We'll use the same execution environment as execute_code_analysis
            # but with fewer constraints on the structure of the analysis code
            return await execute_code_analysis(ctx, analysis_code, f"Custom analysis: {analysis_goal}")
        except Exception as e:
            raise ModelRetry(f"Error in custom codebase analysis: {str(e)}")
