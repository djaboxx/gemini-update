"""
Documentation and library search tools using Gemini's grounded search capabilities.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from functools import wraps

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai import ModelRetry
from pydantic_ai.models.gemini import GeminiModelSettings
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from src.models import CodebaseContext


class SourceReference(BaseModel):
    """Reference to a source document from grounded search."""
    title: str = Field(default="Unknown source", description="Title of the source document")
    url: Optional[str] = Field(None, description="URL of the source document")


class LibraryDocumentationResponse(BaseModel):
    """Response model for library documentation search."""
    library: str = Field(..., description="Name of the library searched")
    query: str = Field(..., description="Original query about the library")
    documentation: str = Field(..., description="Documentation text retrieved")
    sources: List[SourceReference] = Field(default_factory=list, 
                                         description="Sources referenced in the documentation")


class ImplementationPatternsResponse(BaseModel):
    """Response model for implementation patterns search."""
    feature_description: str = Field(..., description="Description of the feature")
    languages: List[str] = Field(default_factory=list, 
                               description="Programming languages relevant to the search")
    frameworks: List[str] = Field(default_factory=list, 
                                description="Frameworks relevant to the search")
    implementation_patterns: str = Field(..., description="Implementation patterns text")
    sources: List[SourceReference] = Field(default_factory=list, 
                                         description="Sources referenced in the patterns")


logger = logging.getLogger("gemini_update")


def register_documentation_tools(agent: Agent[CodebaseContext, str], max_retries: int = 1) -> None:
    """Register documentation search tools with the agent."""
    
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
    
    @retry_on_error
    @agent.tool
    async def search_library_documentation(
        ctx: RunContext[CodebaseContext],
        library_name: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Search for documentation and best practices for a library using grounded search.
        
        Args:
            ctx: The run context with agent dependencies
            library_name: The name of the library to search for (e.g., 'fastapi', 'django', 'react')
            query: The specific question or topic to search for within the library's documentation
            
        Returns:
            Dictionary containing search results, documentation, and best practices
        """
        try:
            if not ctx.deps.settings.gemini_api_key:
                raise ModelRetry("Gemini API key is required for library documentation search")
                
            # Format an effective search prompt
            search_prompt = f"""
            I need detailed, current, and accurate documentation about the '{library_name}' library regarding '{query}'.
            Please provide:
            1. A brief overview of the relevant feature or capability
            2. Code examples showing best practices for implementation 
            3. Common pitfalls to avoid
            4. Current version information and compatibility notes
            5. Links to official documentation for this specific feature
            
            Use grounded search to ensure the information is up-to-date and accurate.
            """
            
            # Configure the model for grounded search
            # Note: Search grounding is automatically enabled in Gemini 1.5+ models
            # We don't need to specifically enable it, just use the model correctly
            model_name = ctx.deps.settings.gemini_model or "gemini-1.5-pro"
            
            safety_settings = {
                "HARASSMENT": "block_none",
                "HATE": "block_none",
                "SEXUAL": "block_none",
                "DANGEROUS": "block_none",
            }
            
            generation_config = GenerationConfig(
                temperature=0.2,  # Low temperature for factual responses
                top_p=0.95,
                top_k=40,
                max_output_tokens=4096,
                response_mime_type="text/plain"
            )
            
            logger.info(f"Searching for documentation on '{library_name}' regarding '{query}'")
            
            # Use a direct model call to ensure we get grounding
            genai.configure(api_key=ctx.deps.settings.gemini_api_key)
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config, safety_settings=safety_settings)
            
            response = model.generate_content(search_prompt)
            
            # Create a Pydantic model instance for the response
            result = LibraryDocumentationResponse(
                library=library_name,
                query=query,
                documentation=response.text,
                sources=[]
            )
            
            # Extract sources if available
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        sources = []
                        for segment in candidate.grounding_metadata.segments:
                            for source in segment.sources:
                                sources.append(SourceReference(
                                    title=source.title if hasattr(source, 'title') else "Unknown source",
                                    url=source.uri if hasattr(source, 'uri') else None
                                ))
                        result.sources = sources
                        logger.info(f"Found {len(sources)} sources for {library_name} documentation")
            except Exception as e:
                logger.warning(f"Error extracting grounding sources: {str(e)}")
            
            # Return the model as a dictionary
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Error searching for library documentation: {str(e)}")
            raise ModelRetry(f"Error searching for library documentation: {str(e)}")
    
    @retry_on_error
    @agent.tool
    async def search_implementation_patterns(
        ctx: RunContext[CodebaseContext],
        feature_description: str,
        languages: Optional[List[str]] = None,
        frameworks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for current best practices and implementation patterns for a specific feature.
        
        Args:
            ctx: The run context with agent dependencies
            feature_description: Description of the feature to be implemented
            languages: Optional list of programming languages relevant to the search (e.g., ['python', 'typescript'])
            frameworks: Optional list of frameworks relevant to the search (e.g., ['django', 'react'])
            
        Returns:
            Dictionary containing implementation patterns, best practices, and code examples
        """
        try:
            if not ctx.deps.settings.gemini_api_key:
                raise ModelRetry("Gemini API key is required for implementation patterns search")
            
            # Format languages and frameworks for the search prompt
            lang_str = f"in {', '.join(languages)}" if languages else ""
            framework_str = f"using {', '.join(frameworks)}" if frameworks else ""
            
            # Format an effective search prompt
            search_prompt = f"""
            I need to implement the following feature: '{feature_description}' {lang_str} {framework_str}.
            
            Please provide:
            1. Common design patterns for this type of feature
            2. Best practices for implementation
            3. Specific code examples showing modern implementation approaches
            4. Architectural considerations
            5. Common pitfalls to avoid
            
            Use grounded search to ensure the information represents current (as of {ctx.args.get('current_date', 'today')}) 
            best practices and patterns.
            """
            
            # Configure the model for grounded search
            model_name = ctx.deps.settings.gemini_model or "gemini-1.5-pro"
            
            safety_settings = {
                "HARASSMENT": "block_none",
                "HATE": "block_none",
                "SEXUAL": "block_none",
                "DANGEROUS": "block_none",
            }
            
            generation_config = GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=4096,
                response_mime_type="text/plain"
            )
            
            logger.info(f"Searching for implementation patterns for: '{feature_description}'")
            
            # Use a direct model call
            genai.configure(api_key=ctx.deps.settings.gemini_api_key)
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config, safety_settings=safety_settings)
            
            response = model.generate_content(search_prompt)
            
            # Create a Pydantic model instance for the response
            result = ImplementationPatternsResponse(
                feature_description=feature_description,
                languages=languages or [],
                frameworks=frameworks or [],
                implementation_patterns=response.text,
                sources=[]
            )
            
            # Extract sources if available
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        sources = []
                        for segment in candidate.grounding_metadata.segments:
                            for source in segment.sources:
                                sources.append(SourceReference(
                                    title=source.title if hasattr(source, 'title') else "Unknown source",
                                    url=source.uri if hasattr(source, 'uri') else None
                                ))
                        result.sources = sources
                        logger.info(f"Found {len(sources)} sources for implementation patterns")
            except Exception as e:
                logger.warning(f"Error extracting grounding sources: {str(e)}")
            
            # Return the model as a dictionary
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Error searching for implementation patterns: {str(e)}")
            raise ModelRetry(f"Error searching for implementation patterns: {str(e)}")
