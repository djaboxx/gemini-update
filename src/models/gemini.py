"""
Models for Gemini agent configuration and responses.
"""

from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field, ConfigDict


class AgentConfig(BaseModel):
    """Configuration for the Gemini agent."""
    
    model_name: str = Field(..., description="Gemini model name to use")
    temperature: float = Field(0.2, description="Temperature for text generation")
    top_p: float = Field(0.95, description="Top-p for text generation")
    top_k: int = Field(40, description="Top-k for text generation")
    max_output_tokens: int = Field(8192, description="Maximum output tokens")
    enable_search: bool = Field(False, description="Enable Google search grounding")
    
    model_config = ConfigDict(extra="allow")


class GeminiResponse(BaseModel):
    """Response from the Gemini API."""
    
    text: str = Field(..., description="Generated text response")
    model: str = Field(..., description="Model used for generation")
    created: int = Field(..., description="Creation timestamp")
    finish_reason: Literal["STOP", "MAX_TOKEN", "SAFETY", "OTHER"] = Field(..., description="Reason for completion")
    safety_ratings: List[Dict[str, Any]] = Field(default_factory=list, description="Safety rating information")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage metadata")
    
    def get_token_count(self) -> Dict[str, int]:
        """
        Extract token usage information if available.
        
        Returns:
            Dictionary with token count information
        """
        if not self.usage_metadata:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
        return {
            "prompt_tokens": self.usage_metadata.get("prompt_token_count", 0),
            "completion_tokens": self.usage_metadata.get("candidates_token_count", 0),
            "total_tokens": self.usage_metadata.get("total_token_count", 0)
        }
