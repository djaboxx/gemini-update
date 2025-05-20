"""
Code execution tools for Gemini-driven codebase analysis.
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import traceback

from pydantic_ai import Agent, RunContext, ModelRetry
from functools import wraps

from src.models import CodebaseContext, AnalysisResult, LanguageStats
from src.models.code_execution import (
    ExecutionStatus,
    CodeExecutionError,
    CodeExecutionResult,
    CodeExecutionRequest,
    CodeSuggestion,
    CodeFixRequest
)

logger = logging.getLogger("gemini_update")


def register_code_execution_tools(agent: Agent[CodebaseContext, str], max_retries: int = 1) -> None:
    """Register code execution tools with the agent."""
    
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
    
    def execute_code(request: CodeExecutionRequest) -> CodeExecutionResult:
        """
        Execute Python code safely using dynamic imports and context managers for isolation.
        """
        start_time = time.time()
        import sys
        import importlib.util
        import resource
        from io import StringIO
        
        # Create StringIO objects for capturing output
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Prepare result
        execution_time = 0.0
        memory_usage = None
        
        try:
            # Create a temporary directory for the script
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create the module path
                module_name = f"gemini_dynamic_code_{int(time.time() * 1000)}"
                script_path = Path(temp_dir) / f"{module_name}.py"
                
                # Write the code directly to a file
                with open(script_path, "w") as f:
                    f.write(request.code)
                
                # Set up environment variables if specified
                if request.environment_variables:
                    original_env = os.environ.copy()
                    os.environ.update(request.environment_variables)
                
                # Create the module spec and load the module
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                if not spec:
                    raise ImportError(f"Could not load module spec from {script_path}")
                
                module = importlib.util.module_from_spec(spec)
                
                # Set up timer for execution
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Execution timed out after {request.timeout_seconds} seconds")
                
                # Save old handlers and set new ones
                old_stdout, old_stderr = sys.stdout, sys.stderr
                old_handler = signal.getsignal(signal.SIGALRM)
                
                try:
                    # Set timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(request.timeout_seconds)
                    
                    # Redirect stdout/stderr
                    sys.stdout, sys.stderr = stdout_capture, stderr_capture
                    
                    # Execute the module
                    spec.loader.exec_module(module)
                    
                    # If there's an analyze function, call it
                    output = module.analyze()
                    
                    # Get memory usage
                    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB to MB
                    execution_time = time.time() - start_time
                    
                    # Reset alarm
                    signal.alarm(0)
                    
                    # Process the output
                    return CodeExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        output=stdout_capture.getvalue() if output is None else output,
                        execution_time=execution_time,
                        memory_usage=memory_usage
                    )
                    
                except Exception as e:
                    # Reset alarm
                    signal.alarm(0)
                    
                    # Handle execution error
                    execution_time = time.time() - start_time
                    return CodeExecutionResult(
                        status=ExecutionStatus.ERROR,
                        errors=[CodeExecutionError(
                            error_type=type(e).__name__,
                            message=str(e),
                            traceback=traceback.format_exc()
                        )],
                        execution_time=execution_time,
                        memory_usage=memory_usage
                    )
                    
                finally:
                    # Reset stdout/stderr
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    
                    # Reset signal handler
                    signal.signal(signal.SIGALRM, old_handler)
                    
                    # Reset environment variables if they were modified
                    if request.environment_variables:
                        os.environ.clear()
                        os.environ.update(original_env)
                
                # Write the script to the temporary file
                with open(script_path, "w") as f:
                    f.write(wrapper_code)
                
                # Set up environment for the subprocess
                env = os.environ.copy()
                if request.environment_variables:
                    env.update(request.environment_variables)
                
                # Execute the script in a subprocess
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=request.timeout_seconds)
                    execution_time = time.time() - start_time
                    
                    # Process the output
                    if process.returncode == 0:
                        try:
                            # Parse the JSON result from stdout
                            result_data = json.loads(stdout)
                            
                            if result_data.get("status") == "error":
                                # Handle error case
                                errors = _parse_error(result_data.get("stderr", ""))
                                return CodeExecutionResult(
                                    status=ExecutionStatus.ERROR,
                                    errors=errors,
                                    execution_time=execution_time,
                                    memory_usage=result_data.get("memory_usage")
                                )
                            else:
                                # Handle success case
                                return CodeExecutionResult(
                                    status=ExecutionStatus.SUCCESS,
                                    output=result_data.get("output", ""),
                                    execution_time=execution_time,
                                    memory_usage=result_data.get("memory_usage")
                                )
                        except json.JSONDecodeError:
                            # Handle case where output isn't valid JSON
                            return CodeExecutionResult(
                                status=ExecutionStatus.ERROR,
                                errors=[CodeExecutionError(
                                    error_type="OutputParsingError",
                                    message="Could not parse execution output as JSON",
                                    traceback=f"stdout: {stdout}\nstderr: {stderr}"
                                )],
                                execution_time=execution_time
                            )
                    else:
                        # Handle subprocess error
                        errors = _parse_error(stderr)
                        return CodeExecutionResult(
                            status=ExecutionStatus.ERROR,
                            errors=errors,
                            execution_time=execution_time
                        )
                    
                except subprocess.TimeoutExpired:
                    # Handle timeout
                    process.kill()
                    execution_time = time.time() - start_time
                    return CodeExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        errors=[CodeExecutionError(
                            error_type="TimeoutError",
                            message=f"Execution took longer than {request.timeout_seconds} seconds and was terminated"
                        )],
                        execution_time=execution_time
                    )
                except Exception as e:
                    # Handle any other exceptions
                    execution_time = time.time() - start_time
                    return CodeExecutionResult(
                        status=ExecutionStatus.ERROR,
                        errors=[CodeExecutionError(
                            error_type=type(e).__name__,
                            message=str(e),
                            traceback=traceback.format_exc()
                        )],
                        execution_time=execution_time
                    )
        
        except Exception as e:
            # Handle setup exceptions
            execution_time = time.time() - start_time
            return CodeExecutionResult(
                status=ExecutionStatus.ERROR,
                errors=[CodeExecutionError(
                    error_type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc()
                )],
                execution_time=execution_time
            )

    def _parse_error(error_text: str) -> List[CodeExecutionError]:
        """
        Parse error output from Python execution into structured errors.
        """
        errors = []
        
        if not error_text:
            return errors
            
        # Try to extract traceback information
        tb_parts = error_text.strip().split("Traceback (most recent call last):")
        
        if len(tb_parts) > 1:
            # We have a traceback
            for i in range(1, len(tb_parts)):
                error_part = tb_parts[i].strip()
                if not error_part:
                    continue
                    
                error_lines = error_part.split('\n')
                
                # The last line usually contains the error type and message
                if error_lines:
                    last_line = error_lines[-1]
                    
                    # Extract error type and message
                    parts = last_line.split(':', 1)
                    if len(parts) >= 1:
                        error_type = parts[0].strip()
                        message = parts[1].strip() if len(parts) > 1 else ""
                        
                        # Try to extract line number
                        line_number = None
                        column = None
                        
                        for line in error_lines:
                            if "line" in line and ", in " in line:
                                try:
                                    # Format like: File "script.py", line 42, in <module>
                                    parts = line.split("line", 1)[1].split(",", 1)[0]
                                    line_number = int(parts.strip())
                                    break
                                except (ValueError, IndexError):
                                    pass
                        
                        errors.append(CodeExecutionError(
                            error_type=error_type,
                            message=message,
                            traceback=f"Traceback (most recent call last):{error_part}",
                            line_number=line_number,
                            column=column
                        ))
        else:
            # No traceback format, just use the whole error
            errors.append(CodeExecutionError(
                error_type="ExecutionError",
                message=error_text.strip(),
                traceback=None
            ))
        
        return errors or [CodeExecutionError(
            error_type="UnknownError",
            message="An error occurred but could not be parsed",
            traceback=error_text
        )]

    @retry_on_error
    @agent.tool
    async def execute_gemini_code(
        ctx: RunContext[CodebaseContext],
        code: str,
        description: str = "Executing code",
        timeout_seconds: int = 30,
        environment_variables: Optional[Dict[str, str]] = None
    ) -> CodeExecutionResult:
        """
        Execute Python code written by Gemini to analyze the codebase.
        
        This tool allows Gemini to write and execute custom Python code with access to:
        - project_dir: Path to the project directory for file exploration
        - Common Python modules like os, re, json, pathlib, ast, etc.
        - Helper utilities for finding imports, analyzing code patterns, etc.
        
        Args:
            ctx: The run context containing project filesystem access
            code: The Python code to execute
            description: Description of what the code is doing
            timeout_seconds: Maximum execution time in seconds
            environment_variables: Optional environment variables to set
            
        Returns:
            CodeExecutionResult object with execution status, output, errors, and timing information
        """
        try:
            # Create a default environment with the project path available
            env = {"PYTHONPATH": str(ctx.deps.project_dir)}
            if environment_variables:
                env.update(environment_variables)
                
            # Create the execution request
            request = CodeExecutionRequest(
                code=code,
                timeout_seconds=timeout_seconds,
                description=description,
                environment_variables=env
            )
            
            # Execute the code
            return execute_code(request)
            
        except Exception as e:
            logger.error(f"Error executing Gemini code: {str(e)}")
            return CodeExecutionResult(
                status=ExecutionStatus.ERROR,
                errors=[CodeExecutionError(
                    error_type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc()
                )],
                execution_time=0.0
            )

    @retry_on_error
    @agent.tool
    async def analyze_codebase_with_gemini(
        ctx: RunContext[CodebaseContext],
        custom_analysis_instructions: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze the codebase using Gemini-generated Python code.
        
        This tool allows Gemini to analyze the codebase by writing and executing its own
        custom Python code. It enables Gemini to identify project type, language, frameworks,
        and other characteristics by inspecting file patterns, imports, and code structures.
        
        IMPORTANT: Your code must define an 'analyze()' function that returns a dictionary with the following structure:
        {
            "project_type": str,           # Type of project (e.g., "Web Backend", "CLI Tool", "AI Agent")
            "primary_language": str,       # Main programming language used
            "description": str,            # Brief description of the project
            "frameworks": List[str],       # List of frameworks used
            "files_analyzed": int,         # Number of files analyzed
            "language_stats": Dict[str, int]  # Dictionary mapping language names to file counts
        }
        
        Args:
            ctx: The run context containing project filesystem access
            custom_analysis_instructions: Optional instructions for customizing the analysis
            
        Returns:
            AnalysisResult object with project information
        """
        try:
            # Generate the analysis code
            analysis_code = """
import os
import json
import re
from pathlib import Path
import ast
import fnmatch
import importlib
from collections import Counter, defaultdict

# Set up result structure
result = {
    "project_type": "Unknown",
    "primary_language": "Unknown",
    "frameworks": [],
    "description": "",
    "language_stats": {},
    "files_analyzed": 0
}

# Get the project directory
project_dir = Path(os.getcwd())

# Define helper functions
def find_files(pattern, base_dir=None):
    """Find files matching a pattern."""
    matches = []
    base_path = project_dir if base_dir is None else Path(base_dir)
    
    for root, dirnames, filenames in os.walk(base_path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    
    return matches

def analyze_imports(file_path):
    """Analyze imports in a Python file."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({"name": name.name, "alias": name.asname})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append({
                        "module": module,
                        "name": name.name,
                        "alias": name.asname
                    })
    except:
        pass
    return imports

# Count files by extension to determine language statistics
language_stats = defaultdict(int)
extension_to_language = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "React/JavaScript",
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
    ".rs": "Rust",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".md": "Markdown",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
    ".sql": "SQL"
}

file_count = 0
for root, dirs, files in os.walk(project_dir):
    # Skip hidden directories and common exclusions
    dirs[:] = [d for d in dirs if not d.startswith('.') and 
               d not in ['node_modules', 'venv', '.venv', '__pycache__', 'dist', 'build']]
    
    for file in files:
        file_count += 1
        _, ext = os.path.splitext(file)
        if ext:
            ext = ext.lower()
            lang = extension_to_language.get(ext, "Other")
            language_stats[lang] += 1

# Sort languages by file count
result["language_stats"] = dict(sorted(language_stats.items(), key=lambda x: x[1], reverse=True))
result["files_analyzed"] = file_count

# Determine primary language
if result["language_stats"]:
    result["primary_language"] = next(iter(result["language_stats"]))

# Detect common directories and files
key_directories = []
common_dirs = ["src", "app", "lib", "server", "client", "test", "tests", "docs"]
for d in common_dirs:
    if os.path.isdir(os.path.join(project_dir, d)):
        key_directories.append(d)

# Check for tests and docs
has_tests = any(d in ["test", "tests"] for d in key_directories) or bool(find_files("test_*.py")) or bool(find_files("*.test.js"))
has_documentation = "docs" in key_directories or os.path.exists(os.path.join(project_dir, "README.md"))

# Look for containerization
is_containerized = os.path.exists(os.path.join(project_dir, "Dockerfile")) or \
                   os.path.exists(os.path.join(project_dir, "docker-compose.yml")) or \
                   os.path.exists(os.path.join(project_dir, "docker-compose.yaml"))

# Identify project type and frameworks based on primary language
if result["primary_language"] == "Python":
    # Look for specific files/directories
    has_setup_py = os.path.exists(os.path.join(project_dir, "setup.py"))
    has_pyproject_toml = os.path.exists(os.path.join(project_dir, "pyproject.toml"))
    has_requirements_txt = os.path.exists(os.path.join(project_dir, "requirements.txt"))
    
    # Check for Python frameworks by analyzing imports
    python_frameworks = set()
    library_imports = set()
    web_framework_patterns = False
    cli_patterns = False
    ai_agent_patterns = False
    data_science_patterns = False
    
    # Framework mapping
    framework_indicators = {
        "flask": "Flask",
        "django": "Django",
        "fastapi": "FastAPI",
        "tornado": "Tornado",
        "bottle": "Bottle",
        "pyramid": "Pyramid",
    }
    
    # AI/ML library mapping
    ai_ml_indicators = {
        "tensorflow", "torch", "keras", "sklearn", "scikit-learn", "pandas", "numpy",
        "matplotlib", "seaborn", "openai", "huggingface", "transformers", "langchain",
        "llama", "anthropic", "vertexai", "pydantic_ai"
    }
    
    # Find Python files
    python_files = find_files("*.py")
    for file_path in python_files[:30]:  # Limit analysis to first 30 Python files
        # Check for entry points
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            if "__name__ == '__main__'" in content or "def main(" in content:
                pass  # Found an entry point
        
        # Analyze imports
        imports = analyze_imports(file_path)
        for imp in imports:
            module = imp.get("module", "")
            name = imp.get("name", "")
            
            # Extract library name
            lib_name = name.split(".")[0] if "." in name else name
            if module:
                lib_name = module.split(".")[0]
            
            if lib_name:
                library_imports.add(lib_name)
                
                # Check for web frameworks
                for indicator, framework in framework_indicators.items():
                    if indicator.lower() in [module.lower(), name.lower(), lib_name.lower()]:
                        python_frameworks.add(framework)
                        web_framework_patterns = True
                        
                # Check for CLI libraries
                if lib_name.lower() in ["click", "typer", "argparse", "fire"]:
                    cli_patterns = True
                    
                # Check for AI/ML libraries
                if lib_name.lower() in ai_ml_indicators:
                    data_science_patterns = True
                    
                # Check for AI agent libraries
                if lib_name.lower() in ["openai", "langchain", "llama", "anthropic", "vertexai", "pydantic_ai"]:
                    ai_agent_patterns = True
    
    # Set frameworks
    result["frameworks"] = list(python_frameworks)
    
    # Determine Python project type
    if web_framework_patterns:
        result["project_type"] = "Web Backend"
    elif ai_agent_patterns:
        result["project_type"] = "AI Agent"
    elif data_science_patterns:
        result["project_type"] = "Data Science" 
    elif cli_patterns:
        result["project_type"] = "Command Line Tool"
    elif has_setup_py or has_pyproject_toml:
        result["project_type"] = "Python Package"
    else:
        result["project_type"] = "Python Application"
        
elif result["primary_language"] in ["JavaScript", "TypeScript"]:
    # Check for package.json
    js_frameworks = []
    if os.path.exists(os.path.join(project_dir, "package.json")):
        with open(os.path.join(project_dir, "package.json"), 'r', encoding='utf-8') as f:
            try:
                package_data = json.load(f)
                dependencies = {}
                if "dependencies" in package_data:
                    dependencies.update(package_data["dependencies"])
                if "devDependencies" in package_data:
                    dependencies.update(package_data["devDependencies"])
                
                # Detect frameworks
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
                if "electron" in dependencies:
                    js_frameworks.append("Electron")
                    
                # Add frameworks to result
                result["frameworks"] = js_frameworks
                
                # Determine JS/TS project type
                if any(fw in js_frameworks for fw in ["React", "Vue.js", "Angular", "Next.js", "Gatsby"]):
                    result["project_type"] = "Web Frontend"
                elif "Express" in js_frameworks:
                    result["project_type"] = "Web Backend"
                elif "Electron" in js_frameworks:
                    result["project_type"] = "Desktop Application"
                elif "react-native" in dependencies:
                    result["project_type"] = "Mobile Application"
                else:
                    result["project_type"] = f"{result['primary_language']} Application"
            except:
                result["project_type"] = f"{result['primary_language']} Application"
    
# Generate a descriptive summary
if result["project_type"] != "Unknown" and result["primary_language"]:
    description = f"A {result['project_type'].lower()} written primarily in {result['primary_language']}"
    if result["frameworks"]:
        description += f" using {', '.join(result['frameworks'])}"
    if has_tests:
        description += " with tests"
    if is_containerized:
        description += ", containerized with Docker"
    description += "."
    result["description"] = description

def analyze():
    """Analyze the codebase and return results."""
    return result

# Output for direct execution
if __name__ == "__main__":
    print(json.dumps(analyze()))
"""
            
            # Apply custom analysis instructions if provided
            if custom_analysis_instructions:
                analysis_code += f"\n# Custom analysis instructions:\n# {custom_analysis_instructions}\n"
                
            # Execute the analysis code
            execution_result = await execute_gemini_code(
                ctx,
                analysis_code,
                description="Analyzing codebase structure and determining project type",
                timeout_seconds=60
            )
            
            # If execution was successful, parse the results
            if execution_result.status == ExecutionStatus.SUCCESS:
                try:
                    # Parse the JSON output
                    analysis_data = json.loads(execution_result.output) if isinstance(execution_result.output, str) else execution_result.output
                    
                    # Convert language stats to the expected format for AnalysisResult
                    language_stats_list = []
                    for lang_name, file_count in analysis_data.get("language_stats", {}).items():
                        language_stats_list.append(LanguageStats(
                            language_name=lang_name,
                            file_count=file_count
                        ))
                    
                    # Create and return the AnalysisResult
                    return AnalysisResult(
                        project_type=analysis_data.get("project_type", "Unknown"),
                        primary_language=analysis_data.get("primary_language", "Unknown"),
                        description=analysis_data.get("description", "A software project."),
                        frameworks=analysis_data.get("frameworks", []),
                        files_analyzed=analysis_data.get("files_analyzed", 0),
                        language_stats=language_stats_list
                    )
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    logger.error(f"Error parsing analysis results: {str(e)}")
                    return AnalysisResult(
                        project_type="Unknown",
                        primary_language="Unknown",
                        description="Failed to parse analysis results.",
                        frameworks=[],
                        files_analyzed=0,
                        language_stats=[]
                    )
            else:
                # Handle execution error
                error_msg = execution_result.errors[0].message if execution_result.errors else "Unknown error"
                logger.error(f"Error executing analysis code: {error_msg}")
                return AnalysisResult(
                    project_type="Unknown",
                    primary_language="Unknown", 
                    description=f"Analysis failed: {error_msg}",
                    frameworks=[],
                    files_analyzed=0,
                    language_stats=[]
                )
                
        except Exception as e:
            logger.error(f"Error analyzing codebase with Gemini: {str(e)}")
            return AnalysisResult(
                project_type="Unknown",
                primary_language="Unknown",
                description=f"Analysis failed with error: {str(e)}",
                frameworks=[],
                files_analyzed=0,
                language_stats=[]
            )
            analysis_code = """
import os
import json
import re
from pathlib import Path
import ast
import fnmatch
import importlib
from collections import Counter, defaultdict

# Set up result structure
result = {
    "project_type": "Unknown",
    "primary_language": "Unknown",
    "frameworks": [],
    "description": "",
    "language_stats": {},
    "key_files": [],
    "key_directories": [],
    "entry_points": [],
    "has_tests": False,
    "has_documentation": False,
    "is_containerized": False,
    "has_git": False
}

# Get the project directory
project_dir = Path(os.getcwd())

# Define helper functions
def find_files(pattern, base_dir=None):
    """Find files matching a pattern."""
    matches = []
    base_path = project_dir if base_dir is None else Path(base_dir)
    
    for root, dirnames, filenames in os.walk(base_path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    
    return matches

def analyze_imports(file_path):
    """Analyze imports in a Python file."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({"name": name.name, "alias": name.asname})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append({
                        "module": module,
                        "name": name.name,
                        "alias": name.asname
                    })
    except:
        pass
    return imports

# Count files by extension to determine language statistics
language_stats = defaultdict(int)
extension_to_language = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "React/JavaScript",
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
    ".rs": "Rust",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".md": "Markdown",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
    ".sql": "SQL"
}

file_count = 0
for root, dirs, files in os.walk(project_dir):
    # Skip hidden directories and common exclusions
    dirs[:] = [d for d in dirs if not d.startswith('.') and 
               d not in ['node_modules', 'venv', '.venv', '__pycache__', 'dist', 'build']]
    
    for file in files:
        file_count += 1
        _, ext = os.path.splitext(file)
        if ext:
            ext = ext.lower()
            lang = extension_to_language.get(ext, "Other")
            language_stats[lang] += 1

# Sort languages by file count
result["language_stats"] = dict(sorted(language_stats.items(), key=lambda x: x[1], reverse=True))
result["files_analyzed"] = file_count

# Determine primary language
if result["language_stats"]:
    result["primary_language"] = next(iter(result["language_stats"]))

# Detect common directories and files
common_dirs = ["src", "app", "lib", "server", "client", "test", "tests", "docs"]
key_directories = []
for d in common_dirs:
    if os.path.isdir(os.path.join(project_dir, d)):
        key_directories.append(d)
result["key_directories"] = key_directories

# Check for tests and docs
result["has_tests"] = any(d in ["test", "tests"] for d in key_directories) or bool(find_files("test_*.py")) or bool(find_files("*.test.js"))
result["has_documentation"] = "docs" in key_directories or os.path.exists(os.path.join(project_dir, "README.md"))

# Look for containerization
result["is_containerized"] = os.path.exists(os.path.join(project_dir, "Dockerfile")) or \
                             os.path.exists(os.path.join(project_dir, "docker-compose.yml")) or \
                             os.path.exists(os.path.join(project_dir, "docker-compose.yaml"))

# Check for git repository
result["has_git"] = os.path.isdir(os.path.join(project_dir, ".git"))

# Identify project type and frameworks based on primary language
if result["primary_language"] == "Python":
    # Look for specific files/directories
    has_setup_py = os.path.exists(os.path.join(project_dir, "setup.py"))
    has_pyproject_toml = os.path.exists(os.path.join(project_dir, "pyproject.toml"))
    has_requirements_txt = os.path.exists(os.path.join(project_dir, "requirements.txt"))
    
    # Check for Python frameworks by analyzing imports
    python_frameworks = set()
    library_imports = set()
    web_framework_patterns = False
    cli_patterns = False
    ai_agent_patterns = False
    data_science_patterns = False
    
    # Framework mapping
    framework_indicators = {
        "flask": "Flask",
        "django": "Django",
        "fastapi": "FastAPI",
        "tornado": "Tornado",
        "bottle": "Bottle",
        "pyramid": "Pyramid",
    }
    
    # AI/ML library mapping
    ai_ml_indicators = {
        "tensorflow", "torch", "keras", "sklearn", "scikit-learn", "pandas", "numpy",
        "matplotlib", "seaborn", "openai", "huggingface", "transformers", "langchain",
        "llama", "anthropic", "vertexai", "pydantic_ai"
    }
    
    # Find Python files
    python_files = find_files("*.py")
    for file_path in python_files[:30]:  # Limit analysis to first 30 Python files
        # Check for entry points
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            if "__name__ == '__main__'" in content or "def main(" in content:
                result["entry_points"].append(os.path.relpath(file_path, project_dir))
        
        # Analyze imports
        imports = analyze_imports(file_path)
        for imp in imports:
            module = imp.get("module", "")
            name = imp.get("name", "")
            
            # Extract library name
            lib_name = name.split(".")[0] if "." in name else name
            if module:
                lib_name = module.split(".")[0]
            
            if lib_name:
                library_imports.add(lib_name)
                
                # Check for web frameworks
                for indicator, framework in framework_indicators.items():
                    if indicator.lower() in [module.lower(), name.lower(), lib_name.lower()]:
                        python_frameworks.add(framework)
                        web_framework_patterns = True
                        
                # Check for CLI libraries
                if lib_name.lower() in ["click", "typer", "argparse", "fire"]:
                    cli_patterns = True
                    
                # Check for AI/ML libraries
                if lib_name.lower() in ai_ml_indicators:
                    data_science_patterns = True
                    
                # Check for AI agent libraries
                if lib_name.lower() in ["openai", "langchain", "llama", "anthropic", "vertexai", "pydantic_ai"]:
                    ai_agent_patterns = True
    
    # Set frameworks
    result["frameworks"] = list(python_frameworks)
    
    # Determine Python project type
    if web_framework_patterns:
        result["project_type"] = "Web Backend"
    elif ai_agent_patterns:
        result["project_type"] = "AI Agent"
    elif data_science_patterns:
        result["project_type"] = "Data Science" 
    elif cli_patterns:
        result["project_type"] = "Command Line Tool"
    elif has_setup_py or has_pyproject_toml:
        result["project_type"] = "Python Package"
    else:
        result["project_type"] = "Python Application"
        
elif result["primary_language"] in ["JavaScript", "TypeScript"]:
    # Check for package.json
    js_frameworks = []
    if os.path.exists(os.path.join(project_dir, "package.json")):
        with open(os.path.join(project_dir, "package.json"), 'r', encoding='utf-8') as f:
            try:
                package_data = json.load(f)
                dependencies = {}
                if "dependencies" in package_data:
                    dependencies.update(package_data["dependencies"])
                if "devDependencies" in package_data:
                    dependencies.update(package_data["devDependencies"])
                
                # Detect frameworks
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
                if "electron" in dependencies:
                    js_frameworks.append("Electron")
                    
                # Add frameworks to result
                result["frameworks"] = js_frameworks
                
                # Determine JS/TS project type
                if any(fw in js_frameworks for fw in ["React", "Vue.js", "Angular", "Next.js", "Gatsby"]):
                    result["project_type"] = "Web Frontend"
                elif "Express" in js_frameworks:
                    result["project_type"] = "Web Backend"
                elif "Electron" in js_frameworks:
                    result["project_type"] = "Desktop Application"
                elif "react-native" in dependencies:
                    result["project_type"] = "Mobile Application"
                else:
                    result["project_type"] = f"{result['primary_language']} Application"
            except:
                result["project_type"] = f"{result['primary_language']} Application"
    
# Generate a descriptive summary
if result["project_type"] != "Unknown" and result["primary_language"]:
    description = f"A {result['project_type'].lower()} written primarily in {result['primary_language']}"
    if result["frameworks"]:
        description += f" using {', '.join(result['frameworks'])}"
    if result["has_tests"]:
        description += " with tests"
    if result["is_containerized"]:
        description += ", containerized with Docker"
    description += "."
    result["description"] = description

# Output the results
print(json.dumps(result))
"""
            
            # Execute the analysis code
            execution_result = await execute_gemini_code(
                ctx,
                analysis_code,
                description="Analyzing codebase structure and determining project type",
                timeout_seconds=60
            )
            
            # If execution was successful, parse the results
            if execution_result.status == ExecutionStatus.SUCCESS:
                try:
                    # Parse the JSON output
                    analysis_data = json.loads(execution_result.output) if execution_result.output else {}
                    
                    # Convert language stats to the expected format for AnalysisResult
                    language_stats_list = []
                    for lang_name, file_count in analysis_data.get("language_stats", {}).items():
                        language_stats_list.append(LanguageStats(
                            language_name=lang_name,
                            file_count=file_count
                        ))
                    
                    # Create and return the AnalysisResult
                    return AnalysisResult(
                        project_type=analysis_data.get("project_type", "Unknown"),
                        primary_language=analysis_data.get("primary_language", "Unknown"),
                        description=analysis_data.get("description", "A software project."),
                        frameworks=analysis_data.get("frameworks", []),
                        files_analyzed=analysis_data.get("files_analyzed", 0),
                        language_stats=language_stats_list
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing analysis results: {str(e)}")
                    return AnalysisResult(
                        project_type="Unknown",
                        primary_language="Unknown",
                        description="Failed to parse analysis results.",
                        frameworks=[],
                        files_analyzed=0,
                        language_stats=[]
                    )
            else:
                # Handle execution error
                error_msg = execution_result.errors[0].message if execution_result.errors else "Unknown error"
                logger.error(f"Error executing analysis code: {error_msg}")
                return AnalysisResult(
                    project_type="Unknown",
                    primary_language="Unknown", 
                    description=f"Analysis failed: {error_msg}",
                    frameworks=[],
                    files_analyzed=0,
                    language_stats=[]
                )
                
        except Exception as e:
            logger.error(f"Error analyzing codebase with Gemini: {str(e)}")
            return AnalysisResult(
                project_type="Unknown",
                primary_language="Unknown",
                description=f"Analysis failed with error: {str(e)}",
                frameworks=[],
                files_analyzed=0,
                language_stats=[]
            )

    @retry_on_error
    @agent.tool
    async def fix_gemini_code(
        ctx: RunContext[CodebaseContext],
        code: str,
        execution_result: CodeExecutionResult,
        fix_description: Optional[str] = None
    ) -> CodeSuggestion:
        """
        Fix Python code that encountered execution errors.
        
        This tool helps Gemini improve code that had errors during execution.
        It provides detailed information about what went wrong and generates
        a suggestion for how to fix the code.
        
        Args:
            ctx: The run context containing project filesystem access
            code: The original Python code with errors
            execution_result: The result of executing the code (with errors)
            fix_description: Optional description of what needs to be fixed
            
        Returns:
            CodeSuggestion with fixed code and explanation
        """
        try:
            # Create a fix request
            fix_request = CodeFixRequest(
                code=code,
                execution_result=execution_result,
                fix_description=fix_description
            )
            
            # Ask Gemini to fix the code
            fix_prompt = fix_request.to_prompt()
            
            # Use the agent itself to generate a fix
            result = await ctx.agent.run_without_tools(
                prompt=fix_prompt,
                deps=ctx.deps
            )
            
            # Extract the fixed code from the response
            response_text = result.output
            
            # Try to extract the code block
            fixed_code = code  # Default to original code
            explanation = "Could not determine how to fix the code"
            
            # Look for code blocks in the response
            code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL)
            if code_blocks:
                fixed_code = code_blocks[0].strip()
                
                # Try to extract the explanation
                explanation_match = re.search(r"(.*?)```", response_text, re.DOTALL)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    # Look after the code block
                    after_code = response_text.split("```")[-1].strip()
                    if after_code:
                        explanation = after_code
            else:
                # No code block found, use the whole response as explanation
                explanation = response_text
            
            # Calculate a simple confidence score based on how different the fixed code is
            from difflib import SequenceMatcher
            confidence = min(0.95, SequenceMatcher(None, code, fixed_code).ratio())
            
            # Return the suggestion
            return CodeSuggestion(
                original_code=code,
                suggested_code=fixed_code,
                explanation=explanation,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error fixing code: {str(e)}")
            return CodeSuggestion(
                original_code=code,
                suggested_code=code,  # Return original code as we couldn't fix it
                explanation=f"Could not fix code: {str(e)}",
                confidence=0.0
            )
