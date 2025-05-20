"""
File access tools for Gemini-driven codebase analysis.
"""

from pathlib import Path
from typing import List, Optional, Union


async def read_file(
    path: Union[str, Path],
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """
    Read the contents of a file with optional line range.

    Args:
        path: Path to the file (string or Path object)
        start_line: First line to read (0-indexed, inclusive)
        end_line: Last line to read (0-indexed, inclusive)

    Returns:
        String contents of the file or requested line range

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the line range is invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path_obj.is_file():
        raise ValueError(f"Path is not a file: {path}")

    with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if start_line is None and end_line is None:
        return "".join(lines)

    # Validate line range
    if start_line is not None and (start_line < 0 or start_line >= len(lines)):
        nl = len(lines)
        ve = f"Start line {start_line} is out of range (file has {nl} lines)"
        raise ValueError(ve)

    if end_line is not None and (end_line < 0 or end_line >= len(lines)):
        raise ValueError(
            f"End line {end_line} is out of range (file has {len(lines)} lines)"
        )

    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError(f"Start line {start_line} is greater than end line {end_line}")

    # Default values if None
    start = 0 if start_line is None else start_line
    end = len(lines) - 1 if end_line is None else end_line

    return "".join(lines[start : end + 1])


async def write_file(path: Union[str, Path], content: str, mode: str = "w") -> None:
    """
    Write content to a file.

    Args:
        path: Path to the file (string or Path object)
        content: Content to write
        mode: File open mode ('w' for write, 'a' for append)

    Raises:
        ValueError: If the mode is not 'w' or 'a'
    """
    if mode not in ["w", "a"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'w' or 'a'.")

    path_obj = Path(path)

    # Create directory if it doesn't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, mode, encoding="utf-8") as f:
        f.write(content)


async def list_directory(path: Union[str, Path]) -> List[str]:
    """
    List files and directories in a directory.

    Args:
        path: Path to the directory

    Returns:
        List of files and directories in the given directory

    Raises:
        FileNotFoundError: If the directory doesn't exist
        NotADirectoryError: If the path is not a directory
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not path_obj.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    return [str(item.name) for item in path_obj.iterdir()]
