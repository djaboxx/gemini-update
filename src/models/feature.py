"""
Models for feature specification and project requirements.
"""

from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class FeatureType(str, Enum):
    CORE = "core"
    UI = "ui"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    OTHER = "other"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RequirementType(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non-functional"
    TECHNICAL = "technical"
    USER_STORY = "user-story"


class Requirement(BaseModel):
    """A specific requirement for a feature."""
    
    id: str = Field(..., description="Unique identifier for the requirement")
    type: RequirementType = Field(..., description="Type of requirement")
    description: str = Field(..., description="Description of the requirement")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Acceptance criteria")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other requirements")


class FeatureSpec(BaseModel):
    """Detailed specification for a feature."""
    
    name: str = Field(..., description="Feature name")
    description: str = Field(..., description="Feature description")
    feature_type: FeatureType = Field(..., description="Type of feature")
    priority: Priority = Field(..., description="Priority level")
    requirements: List[Requirement] = Field(..., description="Requirements for this feature")
    user_personas: List[str] = Field(default_factory=list, description="User personas affected")
    success_metrics: List[str] = Field(default_factory=list, description="Metrics to measure success")
    technical_notes: Optional[str] = Field(None, description="Technical notes or considerations")
    
    def to_markdown(self) -> str:
        """
        Convert the feature specification to a markdown document.
        
        Returns:
            The formatted markdown string
        """
        markdown = f"# Feature Specification: {self.name}\n\n"
        
        markdown += f"**Type:** {self.feature_type.value}\n"
        markdown += f"**Priority:** {self.priority.value}\n\n"
        
        markdown += f"## Description\n\n{self.description}\n\n"
        
        if self.user_personas:
            markdown += "## User Personas\n\n"
            for persona in self.user_personas:
                markdown += f"- {persona}\n"
            markdown += "\n"
        
        markdown += "## Requirements\n\n"
        for req in self.requirements:
            markdown += f"### {req.id}: {req.type.value}\n\n"
            markdown += f"{req.description}\n\n"
            
            if req.acceptance_criteria:
                markdown += "**Acceptance Criteria:**\n\n"
                for criteria in req.acceptance_criteria:
                    markdown += f"- {criteria}\n"
                markdown += "\n"
                
            if req.dependencies:
                markdown += "**Dependencies:**\n\n"
                for dep in req.dependencies:
                    markdown += f"- {dep}\n"
                markdown += "\n"
        
        if self.success_metrics:
            markdown += "## Success Metrics\n\n"
            for metric in self.success_metrics:
                markdown += f"- {metric}\n"
            markdown += "\n"
            
        if self.technical_notes:
            markdown += f"## Technical Notes\n\n{self.technical_notes}\n\n"
            
        return markdown


class FeaturePrompt(BaseModel):
    """Model for generating feature specifications from prompts."""
    
    feature_description: str = Field(..., description="Natural language description of the feature")
    project_context: str = Field(..., description="Context about the project")
    constraints: Optional[List[str]] = Field(None, description="Any constraints to consider")
    examples: Optional[List[str]] = Field(None, description="Example use cases")
    related_features: Optional[List[str]] = Field(None, description="Related existing features")
    
    def to_prompt(self) -> str:
        """
        Convert to a prompt string for Gemini.
        
        Returns:
            Formatted prompt string
        """
        prompt = f"# Feature Request\n\n{self.feature_description}\n\n"
        
        prompt += f"## Project Context\n\n{self.project_context}\n\n"
        
        if self.constraints:
            prompt += "## Constraints\n\n"
            for constraint in self.constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"
            
        if self.examples:
            prompt += "## Examples\n\n"
            for example in self.examples:
                prompt += f"- {example}\n"
            prompt += "\n"
            
        if self.related_features:
            prompt += "## Related Features\n\n"
            for feature in self.related_features:
                prompt += f"- {feature}\n"
            prompt += "\n"
            
        return prompt
