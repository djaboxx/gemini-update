"""
Models for agent completion markers.
"""

from pydantic import BaseModel, Field


class AnalysisCompletionMarker(BaseModel):
    """Simple marker to indicate codebase analysis completion."""
    completed: bool = Field(True, description="Always True to indicate completion")
    project_type: str = Field(..., description="Identified project type")
    primary_language: str = Field(..., description="Identified primary language")
    description: str = Field(..., description="Brief analysis summary")
