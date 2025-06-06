"""
Code execution tools for Gemini-driven codebase analysis.
"""

import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from pydantic_ai import Agent, ModelRetry, RunContext

from src.config import get_gemini_api_key
from src.models import AnalysisResult, CodebaseContext, LanguageStats
from src.models.code_execution import (
    CodeExecutionError,
    CodeExecutionRequest,
    CodeExecutionResult,
    CodeFixRequest,
    CodeSuggestion,
    ExecutionStatus,
)

logger = logging.getLogger("gemini_update")


def register_code_execution_tools(
    agent: Agent[CodebaseContext, str], max_retries: int = 1
) -> None:
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
                        logger.warning(
                            f"Tool {func.__name__} failed, attempt {retries}/{max_retries}: {str(e)}"
                        )
                        continue
                    raise ModelRetry(f"Error in {func.__name__}: {str(e)}")
            raise last_error

        return wrapped_tool

    @retry_on_error
    @agent.tool
    async def execute_gemini_code(
        ctx: RunContext[CodebaseContext],
        code: str,
        description: str = "Executing code",
        timeout_seconds: int = 30,
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> CodeExecutionResult:
        """
        Execute Python code written by Gemini to analyze the codebase.

        IMPORTANT: The code MUST define an 'analyze()' function. This function will be called after dynamic import and should return the result of the analysis (as a dict or JSON-serializable object).

        This tool allows Gemini to write and execute custom Python code with access to:
        - project_dir: Path to the project directory for file exploration
        - Common Python modules like os, re, json, pathlib, ast, etc.
        - Helper utilities for finding imports, analyzing code patterns, etc.

        Args:
            ctx: The run context containing project filesystem access
            code: The Python code to execute (must define analyze())
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
                environment_variables=env,
            )

            # Execute the code (subprocess-based, always calls analyze())
            return execute_code_subprocess(request)

        except Exception as e:
            logger.error(f"Error executing Gemini code: {str(e)}")
            return CodeExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                memory_usage=None,
                errors=[
                    CodeExecutionError(
                        error_type=type(e).__name__,
                        message=str(e),
                        traceback=traceback.format_exc(),
                        line_number=None,
                        column=None,
                    )
                ],
                execution_time=0.0,
            )

    def execute_code_subprocess(request: CodeExecutionRequest) -> CodeExecutionResult:
        """
        Execute Python code in a subprocess. The code must define an analyze() function.
        The subprocess will import the code and call analyze(), returning its result as JSON.
        """
        start_time = time.time()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                module_name = f"gemini_dynamic_code_{int(time.time() * 1000)}"
                script_path = Path(temp_dir) / f"{module_name}.py"

                # Write the code to the file
                with open(script_path, "w") as f:
                    f.write(request.code)

                # Write a wrapper script that imports and calls analyze()
                wrapper_path = Path(temp_dir) / "run_wrapper.py"
                wrapper_code = f"""
import sys
import json
import importlib.util
spec = importlib.util.spec_from_file_location('{module_name}', '{script_path}')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    result = mod.analyze()
    print(json.dumps(result, default=str))
except Exception as e:
    import traceback
    print(json.dumps({{"status": "error", "stderr": traceback.format_exc()}}))
"""
                with open(wrapper_path, "w") as f:
                    f.write(wrapper_code)

                env = os.environ.copy()
                if (
                    hasattr(request, "environment_variables")
                    and request.environment_variables
                ):
                    for var in request.environment_variables:
                        env[var.name] = var.value

                process = subprocess.Popen(
                    [sys.executable, str(wrapper_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                try:
                    stdout, stderr = process.communicate(
                        timeout=request.timeout_seconds
                    )
                    execution_time = time.time() - start_time
                    if process.returncode == 0:
                        try:
                            result_data = json.loads(stdout)
                            if result_data.get("status") == "error":
                                errors = _parse_error(result_data.get("stderr", ""))
                                return CodeExecutionResult(
                                    status=ExecutionStatus.ERROR,
                                    output=None,
                                    memory_usage=None,
                                    errors=errors,
                                    execution_time=execution_time,
                                )
                            else:
                                return CodeExecutionResult(
                                    status=ExecutionStatus.SUCCESS,
                                    output=result_data,
                                    memory_usage=None,
                                    errors=[],
                                    execution_time=execution_time,
                                )
                        except json.JSONDecodeError:
                            return CodeExecutionResult(
                                status=ExecutionStatus.ERROR,
                                output=None,
                                memory_usage=None,
                                errors=[
                                    CodeExecutionError(
                                        error_type="OutputParsingError",
                                        message=("Could not parse output as JSON"),
                                        traceback=(
                                            f"stdout: {stdout}\n" f"stderr: {stderr}"
                                        ),
                                        line_number=None,
                                        column=None,
                                    )
                                ],
                                execution_time=execution_time,
                            )
                    else:
                        errors = _parse_error(stderr)
                        return CodeExecutionResult(
                            status=ExecutionStatus.ERROR,
                            output=None,
                            memory_usage=None,
                            errors=errors,
                            execution_time=execution_time,
                        )
                except subprocess.TimeoutExpired:
                    process.kill()
                    execution_time = time.time() - start_time
                    return CodeExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        output=None,
                        memory_usage=None,
                        errors=[
                            CodeExecutionError(
                                error_type="TimeoutError",
                                message=(
                                    f"Execution took longer than "
                                    f"{request.timeout_seconds} seconds and was terminated"
                                ),
                                traceback=None,
                                line_number=None,
                                column=None,
                            )
                        ],
                        execution_time=execution_time,
                    )
                except Exception as e:
                    execution_time = time.time() - start_time
                    return CodeExecutionResult(
                        status=ExecutionStatus.ERROR,
                        output=None,
                        memory_usage=None,
                        errors=[
                            CodeExecutionError(
                                error_type=type(e).__name__,
                                message=str(e),
                                traceback=traceback.format_exc(),
                                line_number=None,
                                column=None,
                            )
                        ],
                        execution_time=execution_time,
                    )
        except Exception as e:
            execution_time = time.time() - start_time
            return CodeExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                memory_usage=None,
                errors=[
                    CodeExecutionError(
                        error_type=type(e).__name__,
                        message=str(e),
                        traceback=traceback.format_exc(),
                        line_number=None,
                        column=None,
                    )
                ],
                execution_time=execution_time,
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

                error_lines = error_part.split("\n")

                # The last line usually contains the error type and message
                if error_lines:
                    last_line = error_lines[-1]

                    # Extract error type and message
                    parts = last_line.split(":", 1)
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

                        errors.append(
                            CodeExecutionError(
                                error_type=error_type,
                                message=message,
                                traceback=f"Traceback (most recent call last):{error_part}",
                                line_number=line_number,
                                column=column,
                            )
                        )
        else:
            # No traceback format, just use the whole error
            errors.append(
                CodeExecutionError(
                    error_type="ExecutionError",
                    message=error_text.strip(),
                    traceback=None,
                    line_number=None,
                    column=None,
                )
            )

        return errors or [
            CodeExecutionError(
                error_type="UnknownError",
                message="An error occurred but could not be parsed",
                traceback=error_text,
                line_number=None,
                column=None,
            )
        ]

    @retry_on_error
    @agent.tool
    async def analyze_codebase_with_gemini(
        ctx: RunContext[CodebaseContext],
        custom_analysis_instructions: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Analyze the codebase using Gemini-generated Python code.

        This tool instructs Gemini to generate Python code that defines an 'analyze()' function,
        which will be executed to analyze the codebase. The analyze() function must return a
        dictionary with the following structure:
        {
            "project_type": str,           # Type of project (e.g., "CLI Tool", "AI Agent")
            "primary_language": str,       # Main programming language used
            "description": str,            # Brief description of the project
            "frameworks": List[str],       # List of frameworks used
            "files_analyzed": int,         # Number of files analyzed
            "language_stats": Dict[str, int]  # Dictionary mapping language names to file counts
        }
        The analyze() function will be called after dynamic import.
        """
        try:
            # Build a prompt for Gemini to generate the analysis code
            prompt = (
                "Write Python code to analyze the codebase in the current working directory. "
                "Your code MUST define an analyze() function that returns a dictionary with the following keys: "
                "project_type, primary_language, description, frameworks, files_analyzed, language_stats.\n\n"
                "IMPORTANT: Focus on identifying the core structure of the project. It's essential to correctly "
                "identify whether this is a web application (with backend/frontend), a CLI tool, a Python library, "
                "or an AI agent. Look for evidence in the file structure, entry points, and imports.\n\n"
                "Look for patterns in the code to identify the project_type accurately:\n"
                "- CLI tools typically have argument parsing and command processors\n"
                "- Web applications usually have routes, controllers, and templates\n"
                "- Libraries will have importable modules but no clear entry point\n"
                "- AI agents will have code related to models, prompts, etc.\n\n"
                "You may use os, pathlib, ast, json, fnmatch, and other standard libraries. "
            )
            if custom_analysis_instructions:
                prompt += f"\nAdditional instructions: {custom_analysis_instructions}\n"
            prompt += (
                "\nDo not print or output anything except from the analyze() function. "
                "The code will be executed in a subprocess and analyze() will be called."
            )

            # Ask Gemini to generate the code
            result = await ctx.agent.run_without_tools(prompt, deps=ctx.deps)
            code = result.output

            # Execute the generated code
            execution_result = await execute_gemini_code(
                ctx,
                code,
                description="Analyzing codebase structure and determining project type",
                timeout_seconds=60,
            )

            # If execution was successful, parse the results
            if execution_result.status == ExecutionStatus.SUCCESS:
                try:
                    analysis_data = (
                        json.loads(execution_result.output)
                        if isinstance(execution_result.output, str)
                        else execution_result.output
                    )
                    language_stats_list = []
                    for lang_name, file_count in analysis_data.get(
                        "language_stats", {}
                    ).items():
                        language_stats_list.append(
                            LanguageStats(
                                language_name=lang_name, file_count=file_count
                            )
                        )
                    return AnalysisResult(
                        project_type=analysis_data.get("project_type", "Unknown"),
                        primary_language=analysis_data.get(
                            "primary_language", "Unknown"
                        ),
                        description=analysis_data.get(
                            "description", "A software project."
                        ),
                        frameworks=analysis_data.get("frameworks", []),
                        files_analyzed=analysis_data.get("files_analyzed", 0),
                        language_stats=language_stats_list,
                    )
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    logger.error(f"Error parsing analysis results: {str(e)}")
                    return AnalysisResult(
                        project_type="Unknown",
                        primary_language="Unknown",
                        description="Failed to parse analysis results.",
                        frameworks=[],
                        files_analyzed=0,
                        language_stats=[],
                    )
            else:
                error_msg = (
                    execution_result.errors[0].message
                    if execution_result.errors
                    else "Unknown error"
                )
                logger.error(f"Error executing analysis code: {error_msg}")
                return AnalysisResult(
                    project_type="Unknown",
                    primary_language="Unknown",
                    description=f"Analysis failed: {error_msg}",
                    frameworks=[],
                    files_analyzed=0,
                    language_stats=[],
                )
        except Exception as e:
            logger.error(f"Error analyzing codebase with Gemini: {str(e)}")
            return AnalysisResult(
                project_type="Unknown",
                primary_language="Unknown",
                description=f"Analysis failed with error: {str(e)}",
                frameworks=[],
                files_analyzed=0,
                language_stats=[],
            )

    # Function to execute code directly without requiring an analyze function
    @retry_on_error
    @agent.tool
    async def execute_code(
        ctx: RunContext[CodebaseContext],
        code: str,
        description: str = "Executing code",
        timeout_seconds: int = 10,
        environment_variables: Optional[Dict[str, str]] = None,
        context_data: Optional[Any] = None,
    ) -> Tuple[Any, Optional[str]]:
        """
        Execute a string containing Python code safely and return the result.

        The code should define an analyze() function that takes an optional context_data parameter.
        This function will be called after the code is executed, and its return value will be returned.

        Args:
            ctx: The run context containing project filesystem access
            code: The Python code to execute
            description: Description of what the code is doing
            timeout_seconds: Maximum execution time in seconds
            environment_variables: Optional environment variables to set
            context_data: Optional data to pass to the analyze function

        Returns:
            Tuple of (result of execution, error message or None)
        """
        if not code:
            return None, "No code provided"

        # Create a temporary Python module to execute the code
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "analysis_code.py"

            # Write the code to a file
            with open(temp_file, "w") as f:
                f.write(code)

            # Create the execution environment
            env = os.environ.copy()
            if environment_variables:
                env.update(environment_variables)

            # Set up the Python path
            env["PYTHONPATH"] = f"{ctx.deps.project_dir}:{env.get('PYTHONPATH', '')}"

            try:
                # Import the module
                module_name = "analysis_code"
                spec = importlib.util.spec_from_file_location(module_name, temp_file)
                if spec is None:
                    error_msg = f"Could not create a module spec for {temp_file}"
                    logger.error(error_msg)
                    return None, error_msg
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                if spec.loader is not None:
                    spec.loader.exec_module(module)
                else:
                    error_msg = f"Module spec loader is None for {temp_file}"
                    logger.error(error_msg)
                    return None, error_msg

                # Look for an analyze function
                if hasattr(module, "analyze"):
                    try:
                        result = module.analyze(context_data)
                        return result, None
                    except Exception as e:
                        error_msg = f"Error executing analyze() function: {str(e)}"
                        logger.error(error_msg)
                        return None, error_msg
                else:
                    error_msg = "Code does not contain an analyze() function"
                    logger.error(error_msg)
                    return None, error_msg

            except Exception as e:
                error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                return None, error_msg

    @retry_on_error
    @agent.tool
    async def fix_gemini_code(
        ctx: RunContext[CodebaseContext],
        code: str,
        execution_result: CodeExecutionResult,
        fix_description: Optional[str] = None,
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
                fix_description=fix_description,
            )

            # Ask Gemini to fix the code
            fix_prompt = fix_request.to_prompt()

            # Use the agent itself to generate a fix
            result = await agent.run_without_tools(prompt=fix_prompt, deps=ctx.deps)

            # Extract the fixed code from the response
            response_text = result.output

            # Try to extract the code block
            fixed_code = code  # Default to original code
            explanation = "Could not determine how to fix the code"

            # Look for code blocks in the response
            code_blocks = re.findall(
                r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL
            )
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
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error fixing code: {str(e)}")
            return CodeSuggestion(
                original_code=code,
                suggested_code=code,  # Return original code as we couldn't fix it
                explanation=f"Could not fix code: {str(e)}",
                confidence=0.0,
            )
