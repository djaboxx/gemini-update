"""
Pydantic-AI Tools for interacting with the codebase.
"""

from functools import wraps
import os
import ast
import fnmatch
import re
import json
import sys
import traceback
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agent.agent import Agent, RunContext
from pydantic_ai import ModelRetry
from ..models.analysis import AnalysisResult, LanguageStats
from ..models.file_access import CodebaseContext
from ..utils.log import logger


def _get_project_info_impl(project_dir: Path) -> Dict[str, Any]:
    # This is a placeholder for the actual implementation
    # In a real scenario, this function would analyze the project
    # and return a dictionary with project information.
    return {
        "project_name": project_dir.name,
        "files": [],
        "directories": [],
        "language_stats": {},
        "primary_language": "Unknown",
    }


def register_tools(
    agent: Agent[CodebaseContext, str], max_retries: int = 1
) -> None:
    """Register codebase interaction tools with the agent."""

    def retry_on_error(func):
        """Decorator to add retry logic to tools."""

        @wraps(func)
        async def wrapped_tool(*args, **kwargs):
            retries = 0
            last_error = None
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except ModelRetry as e:
                    logger.warning(
                        f"ModelRetry: {e}. Retrying "
                        f"({retries + 1}/{max_retries + 1})"
                    )
                    last_error = e
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Max retries reached for {func.__name__}."
                        )
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            if last_error:
                raise last_error
            # This line should ideally not be reached if logic is correct
            raise Exception(
                f"Tool {func.__name__} failed after multiple retries."
            )

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
            file_path: Relative or absolute path to the file
            start_line: Optional starting line number (0-based)
            end_line: Optional ending line number (0-based)

        Returns:
            The file content as a string
        """
        try:
            if ctx.deps.file_access:
                return ctx.deps.file_access.read_file(
                    file_path, start_line, end_line
                )

            abs_path = ctx.deps.validate_file_path(file_path)
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                if start_line is None and end_line is None:
                    return f.read()
                else:
                    lines = f.readlines()
                    if start_line is None:
                        start_line = 0
                    if end_line is None:
                        end_line = len(lines) - 1
                    start_line = max(0, min(start_line, len(lines) - 1))
                    end_line = max(0, min(end_line, len(lines) - 1))
                    return "".join(lines[start_line : end_line + 1])
        except (FileNotFoundError, ValueError, PermissionError) as e:
            raise ModelRetry(f"Error reading file: {str(e)}")

    @retry_on_error
    @agent.tool
    async def list_directory(
        ctx: RunContext[CodebaseContext], dir_path: str
    ) -> List[str]:
        """
        List files and directories within a directory in the project.

        Args:
            ctx: The run context containing project filesystem access
            dir_path: Path to the directory within the project

        Returns:
            List of file and directory names
        """
        try:
            if ctx.deps.file_access:
                return ctx.deps.file_access.list_directory(dir_path)
            abs_path = ctx.deps.validate_file_path(dir_path)
            if not os.path.isdir(abs_path):
                raise ValueError(f"Path '{dir_path}' is not a directory")
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
            pattern: Glob pattern to match files (e.g., "*.py")
            base_dir: Optional base directory to start the search
                      (relative to project root)
            recursive: Whether to search recursively (default: True)

        Returns:
            List of files matching the pattern
        """
        try:
            if base_dir:
                base_path = ctx.deps.validate_file_path(base_dir)
                if not Path(base_path).is_dir():
                    raise ModelRetry(f"Not a directory: {base_dir}")
            else:
                base_path = ctx.deps.project_dir

            result = []
            base_path_obj = Path(base_path)

            if recursive:
                walk_iter = os.walk(base_path)
            else:
                walk_iter = [(base_path, [], os.listdir(base_path))]

            for root, _, files in walk_iter:
                for file in files:
                    file_path_obj = Path(root) / file
                    # Ensure we are dealing with files only before relative_to
                    if file_path_obj.is_file():
                        try:
                            rel_path = file_path_obj.relative_to(base_path_obj)
                            if fnmatch.fnmatch(str(rel_path), pattern):
                                result.append(str(rel_path))
                        except ValueError:
                            # This can happen if file_path_obj is not under base_path_obj
                            # which might indicate a symlink loop or unusual structure.
                            # For now, we log and skip.
                            logger.debug(f"Skipping {file_path_obj} as it's not relative to {base_path_obj}")

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
            file_patterns: Optional list of glob patterns to restrict search
            case_sensitive: Whether the search should be case-sensitive

        Returns:
            List of matches with file paths, line numbers, and matching lines
        """
        try:
            project_dir = ctx.deps.project_dir
            result_matches = []  # Renamed to avoid conflict

            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                regex_pattern = re.compile(query, flags)
            except re.error:
                regex_pattern = re.compile(re.escape(query), flags)

            files_to_search = []
            if file_patterns:
                for file_pattern_item in file_patterns:
                    pattern_files = await find_files(ctx, file_pattern_item)
                    files_to_search.extend(pattern_files)
            else:
                for ext in [
                    ".py", ".js", ".ts", ".java", ".go", ".c", ".cpp", ".h",
                    ".hpp", ".cs", ".html", ".css",
                ]:
                    pattern_files = await find_files(ctx, f"**/*{ext}")
                    files_to_search.extend(pattern_files)

            files_to_search = list(set(files_to_search))

            for file_path_str in files_to_search:
                abs_path = project_dir / file_path_str
                try:
                    with open(
                        abs_path, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        lines = f.readlines()
                    for i, line_content in enumerate(lines):
                        if regex_pattern.search(line_content):
                            result_matches.append(
                                {
                                    "file_path": file_path_str,
                                    "line_number": i + 1, # 1-based
                                    "line_content": line_content.strip(),
                                }
                            )
                except (UnicodeDecodeError, IOError):
                    continue
            return result_matches
        except Exception as e:
            raise ModelRetry(f"Error searching code: {str(e)}")

    @retry_on_error
    @agent.tool
    async def analyze_imports(
        ctx: RunContext[CodebaseContext], file_path: str
    ) -> Dict[str, Any]:
        """
        Analyze imports and dependencies in a Python file.

        Args:
            ctx: The run context containing project filesystem access
            file_path: Relative or absolute path to the Python file

        Returns:
            Dictionary with import information
        """
        try:
            abs_path = ctx.deps.validate_file_path(file_path)
            if not str(abs_path).endswith(".py"):
                raise ModelRetry(f"Not a Python file: {file_path}")
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except Exception as e:
                raise ModelRetry(f"Error reading file: {str(e)}")
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                raise ModelRetry(f"Syntax error in {file_path}: {str(e)}")

            imports_list = []  # Renamed
            from_imports_list = []  # Renamed

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name_node in node.names: # Renamed variable
                        imports_list.append(
                            {"name": name_node.name, "alias": name_node.asname}
                        )
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or "" # Renamed variable
                    for name_node in node.names: # Renamed variable
                        from_imports_list.append(
                            {
                                "module": module_name,
                                "name": name_node.name,
                                "alias": name_node.asname,
                            }
                        )
            return {
                "file": file_path,
                "imports": imports_list,
                "from_imports": from_imports_list,
            }
        except Exception as e:
            raise ModelRetry(f"Error analyzing imports: {str(e)}")

    @retry_on_error
    @agent.tool
    async def get_project_info(
        ctx: RunContext[CodebaseContext]
    ) -> Dict[str, Any]:
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
        description: str = "Analyzing codebase...",
    ) -> Dict[str, Any]:
        """
        Execute custom Python code to analyze the codebase.

        This tool allows Gemini to write and execute Python code that inspects
        the project structure, analyzes imports, and detects patterns to
        determine project type and characteristics.

        The code will be executed in a context where it has access to:
        - project_dir: Path to the project directory (as a pathlib.Path object)
        - result: A dictionary that the analysis_code should populate.
        - os, re, json, ast, fnmatch, Path, sys, typing: Common Python modules.

        The `analysis_code` must include all necessary logic. Gemini is
        responsible for writing the complete analysis script, including any
        file system operations or import parsing if needed.

        Args:
            ctx: The run context containing project filesystem access.
            analysis_code: Python code to execute for analysis.
            description: Description of what the analysis is doing.

        Returns:
            Dictionary with analysis results.
        """
        try:
            project_dir_path = Path(ctx.deps.project_dir).resolve()
            exec_globals = {
                "project_dir": project_dir_path,
                "result": {}, 
                "os": os,
                "re": re,
                "json": json,
                "ast": ast,
                "fnmatch": fnmatch,
                "Path": Path,
                "sys": sys,
                "typing": typing,
            }
            exec(analysis_code, exec_globals)
            analysis_output = exec_globals.get("result")
            if not isinstance(analysis_output, dict):
                logger.warning(
                    f"Analysis code did not produce a dict result. "
                    f"Got: {type(analysis_output)}"
                )
                return {
                    "error": f"Analysis code did not produce a dict result. "
                             f"Type: {type(analysis_output)}"
                }
            return analysis_output
        except SyntaxError as e:
            error_msg = (
                f"Syntax error in analysis code (line {e.lineno}, "
                f"offset {e.offset}): {e.msg}"
            )
            logger.error(f"{error_msg}\nProblematic code near error: {e.text}")
            raise ModelRetry(error_msg)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"Error executing analysis code: {type(e).__name__}: {str(e)}\n"
                f"Traceback:\n{tb_str}"
            )
            raise ModelRetry(
                f"Error during analysis code execution: {type(e).__name__}: {str(e)}"
            )

    @retry_on_error
    @agent.tool
    async def analyze_codebase(
        ctx: RunContext[CodebaseContext], include_dependencies: bool = False
    ) -> AnalysisResult:
        """
        Perform a comprehensive analysis of the codebase.

        Args:
            ctx: The run context containing project filesystem access
            include_dependencies: Whether to include detailed dependency info

        Returns:
            AnalysisResult containing the analysis
        """
        try:
            project_info_dict = _get_project_info_impl(ctx.deps.project_dir)

            analysis_code = """
# This is a placeholder for Gemini-generated analysis code.
# Gemini will write Python code here to analyze the project.
# The code will have access to 'project_dir', 'os', 're', 'json', etc.
# and should populate the 'result' dictionary.

result = {
    "project_type": "Unknown",
    "primary_language": project_info_dict.get("primary_language", "Unknown"),
    "frameworks": [],
    "description": "",
    "key_files": [],
    "key_directories": [],
    "has_tests": False,
    "has_documentation": False,
    "entry_points": [],
    "code_patterns": {},
    "language_stats": project_info_dict.get("language_stats", {}),
    "is_containerized": False, # Example: detect Dockerfile
    "has_git": (project_dir / ".git").exists() # Example: detect .git folder
}

# Example: Detect Python project
if result["primary_language"] == "Python":
    if (project_dir / "requirements.txt").exists() or \
       (project_dir / "setup.py").exists() or \
       (project_dir / "pyproject.toml").exists():
        result["project_type"] = "Python Application/Library"

# Gemini would add more sophisticated detection logic here.

# Ensure description is populated
if not result["description"] and result["project_type"] != "Unknown":
    result["description"] = f"A {result['project_type']} project."
            """
            # Pass project_info_dict to the exec_globals for the analysis_code
            exec_globals_for_analysis = {
                "project_dir": Path(ctx.deps.project_dir).resolve(),
                "result": {}, 
                "os": os,
                "re": re,
                "json": json,
                "ast": ast,
                "fnmatch": fnmatch,
                "Path": Path,
                "sys": sys,
                "typing": typing,
                "project_info_dict": project_info_dict # Make it available
            }

            analysis_result_dict = await execute_code_analysis(
                ctx, analysis_code, "Performing detailed codebase analysis..."
            )

            if "error" in analysis_result_dict:
                raise ModelRetry(
                    f"Error in custom analysis code: {analysis_result_dict['error']}"
                )
            
            # Convert dict to AnalysisResult Pydantic model
            # Ensure all required fields for AnalysisResult are present or have defaults
            lang_stats_data = analysis_result_dict.get("language_stats", {})
            language_stats_list = [
                LanguageStats(language_name=lang_name, file_count=file_count)
                for lang_name, file_count in lang_stats_data.items()
            ]

            return AnalysisResult(
                project_name=project_info_dict.get(
                    "project_name", ctx.deps.project_dir.name
                ),
                project_type=analysis_result_dict.get("project_type", "Unknown"),
                primary_language=analysis_result_dict.get(
                    "primary_language", "Unknown"
                ),
                frameworks=analysis_result_dict.get("frameworks", []),
                description=analysis_result_dict.get("description", ""),
                key_files=analysis_result_dict.get("key_files", []),
                key_directories=analysis_result_dict.get("key_directories", []),
                has_tests=analysis_result_dict.get("has_tests", False),
                has_documentation=analysis_result_dict.get(
                    "has_documentation", False
                ),
                entry_points=analysis_result_dict.get("entry_points", []),
                code_patterns=analysis_result_dict.get("code_patterns", {}),
                language_stats=language_stats_list,
                dependencies=[],  # Placeholder, to be implemented if needed
                is_containerized=analysis_result_dict.get("is_containerized", False),
                has_git=analysis_result_dict.get("has_git", False),
            )

        except Exception as e:
            logger.error(f"Error in analyze_codebase: {str(e)}")
            # Fallback to a basic AnalysisResult if detailed analysis fails
            return AnalysisResult(
                project_name=ctx.deps.project_dir.name,
                project_type="Unknown",
                primary_language="Unknown",
                description=f"Failed to perform detailed analysis: {str(e)}",
            )


# Placeholder for a more advanced analysis tool if needed in the future
# @retry_on_error
# @agent.tool
# async def advanced_code_analysis(
#     ctx: RunContext[CodebaseContext], analysis_goal: str, analysis_code: str
# ) -> Dict[str, Any]:
#     """
#     Execute highly customized Python code for specific codebase analysis goals.
#     This advanced tool is meant for targeted analysis of specific aspects of the codebase.
#     Unlike execute_code_analysis, which has a predetermined template, this tool allows for
#     fully custom code execution with minimal boilerplate.
#     Args:
#         ctx: The run context.
#         analysis_goal: A clear description of what the analysis is trying to determine.
#         analysis_code: The Python code to execute.
#     Returns:
#         A dictionary containing the results of the analysis.
#     """
#     try:
#         project_dir_path = Path(ctx.deps.project_dir).resolve()
#         exec_globals = {
#             "project_dir": project_dir_path,
#             "result": {},
#             "os": os,
#             "re": re,
#             "json": json,
#             "ast": ast,
#             "fnmatch": fnmatch,
#             "Path": Path,
#             "sys": sys,
#             "typing": typing,
#         }
#         exec(analysis_code, exec_globals)
#         return exec_globals.get("result", {"error": "No result produced"})
#     except Exception as e:
#         logger.error(f"Error in advanced_code_analysis: {str(e)}")
#         raise ModelRetry(f"Error executing advanced analysis: {str(e)}")
