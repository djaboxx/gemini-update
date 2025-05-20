"""
Utilities for safely executing code and handling results.
"""

import sys
import os
import time
import traceback
import tempfile
import json
import subprocess
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import resource

from src.models.code_execution import (
    ExecutionStatus,
    CodeExecutionError,
    CodeExecutionResult,
    CodeExecutionRequest,
    CodeFixRequest,
    CodeSuggestion
)


def _parse_error(error_text: str) -> List[CodeExecutionError]:
    """
    Parse error output from Python execution into structured errors.
    
    Args:
        error_text: The error text from stderr
        
    Returns:
        List of parsed CodeExecutionError objects
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


def execute_code(request: CodeExecutionRequest) -> CodeExecutionResult:
    """
    Execute Python code safely in a subprocess with timeout and resource limits.
    
    Args:
        request: The code execution request
        
    Returns:
        CodeExecutionResult with execution details
    """
    start_time = time.time()
    
    try:
        # Create a temporary directory for the script
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "script.py"
            
            # Modify code to capture and format the output
            wrapper_code = f"""
import sys
import json
import traceback
import resource

# Redirect stdout/stderr to capture output
from io import StringIO
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

result = {{"status": "success", "output": None, "errors": None}}

try:
    # Execute the user code
{request.code.rstrip().replace(chr(10), chr(10) + '    ')}
    
    # Capture the output
    sys.stdout.flush()
    result["output"] = stdout_capture.getvalue()
    
    # Get peak memory usage
    result["memory_usage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB to MB
    
except Exception as e:
    result["status"] = "error"
    sys.stderr.write(traceback.format_exc())
    
finally:
    # Reset stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Include stderr in the result
    result["stderr"] = stderr_capture.getvalue()
    
    # Print the result as JSON for the parent process to parse
    print(json.dumps(result, default=str))
"""
            
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


def execute_code_string(code: str, timeout: int = 10) -> CodeExecutionResult:
    """
    Convenience function to execute a code string directly.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        CodeExecutionResult with execution details
    """
    request = CodeExecutionRequest(
        code=code,
        timeout_seconds=timeout,
        description="Code execution"
    )
    return execute_code(request)


def fix_code_errors(original_code: str, execution_result: CodeExecutionResult, fix_description: Optional[str] = None) -> CodeFixRequest:
    """
    Create a request to fix code based on execution errors.
    
    Args:
        original_code: The code with errors
        execution_result: The execution result containing errors
        fix_description: Optional description of what needs fixing
        
    Returns:
        CodeFixRequest object suitable for prompting an AI to fix the code
    """
    return CodeFixRequest(
        code=original_code,
        execution_result=execution_result, 
        fix_description=fix_description
    )
