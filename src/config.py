"""
Configuration module for Gemini Update.
"""

import os
from typing import Optional

def get_gemini_api_key() -> Optional[str]:
    """
    Get the Gemini API key from the environment.
    
    Returns:
        The API key if available, None otherwise
    """
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
