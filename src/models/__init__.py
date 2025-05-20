"""
Package initialization for models.
"""

from src.models.analysis import (
    CodebaseFile, 
    CodeDependency, 
    FeatureScope, 
    ChangeType,
    CodeChange, 
    ImplementationPlan,
    CodebaseContext,
    Settings,
    AnalysisResult,
    LanguageStats
)

from src.models.code_execution import (
    ExecutionStatus,
    CodeExecutionError,
    CodeExecutionResult,
    CodeExecutionRequest,
    CodeSuggestion,
    CodeFixRequest
)

from src.models.feature import (
    FeatureType,
    Priority,
    RequirementType,
    Requirement,
    FeatureSpec,
    FeaturePrompt
)

from src.models.gemini import (
    AgentConfig,
    GeminiResponse
)

from src.models.completion_marker import (
    AnalysisCompletionMarker
)

__all__ = [
    # Analysis models
    "CodebaseFile",
    "CodeDependency",
    "FeatureScope",
    "ChangeType",
    "CodeChange",
    "ImplementationPlan",
    "CodebaseContext",
    "AnalysisResult",
    "LanguageStats",
    # Analysis settings
    "Settings",
    
    # Code execution models
    "ExecutionStatus",
    "CodeExecutionError",
    "CodeExecutionResult",
    "CodeExecutionRequest",
    "CodeSuggestion",
    "CodeFixRequest",
    
    # Feature models
    "FeatureType",
    "Priority",
    "RequirementType",
    "Requirement", 
    "FeatureSpec",
    "FeaturePrompt",
    
    # Gemini models
    "AgentConfig",
    "GeminiResponse",
    
    # Completion markers
    "AnalysisCompletionMarker"
]
