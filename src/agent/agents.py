"""
Specialized agent implementations for codebase analysis and feature updates.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from src.models import (
    CodebaseContext, 
    FeatureSpec, 
    ImplementationPlan,
    Settings,
    AnalysisResult
)

from src.tools.codebase_tools import register_tools
from src.tools.feature_tools import register_feature_tools
from src.tools.documentation_tools import register_documentation_tools
from src.tools.code_execution_tools import register_code_execution_tools
from src.agent.common import CommonGeminiTools
from src.prompts.codebase_analyzer_prompt import get_codebase_analyzer_system_prompt, get_codebase_analyzer_prompt
from src.prompts.feature_spec_prompt import get_feature_spec_system_prompt, get_feature_spec_prompt
from src.prompts.implementation_planner_prompt import get_implementation_planner_system_prompt, get_implementation_planner_prompt
from src.prompts.codebase_analyzer_prompt import get_codebase_analyzer_system_prompt, get_codebase_analyzer_prompt
from src.prompts.feature_spec_prompt import get_feature_spec_system_prompt, get_feature_spec_prompt
from src.prompts.implementation_planner_prompt import get_implementation_planner_system_prompt, get_implementation_planner_prompt


logger = logging.getLogger("gemini_update")
console = Console()


class CodebaseAnalyzerAgent:
    """Agent specialized for analyzing codebases."""
    
    def __init__(self, settings: Settings, common_tools: CommonGeminiTools, model_name: Optional[str] = None):
        """Initialize the codebase analyzer agent."""
        self.settings = settings
        self.common_tools = common_tools
        
        system_prompt = get_codebase_analyzer_system_prompt()

        token_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192
        }
        
        self.agent = self.common_tools.create_pydantic_agent(
            model_name=model_name or "gemini-1.5-pro",
            token_config=token_config,
            deps_type=CodebaseContext,
            output_type=AnalysisResult,
            system_prompt_str=system_prompt
        )
        
        register_tools(self.agent)
        register_code_execution_tools(self.agent)

    async def analyze(self, context: CodebaseContext) -> CodebaseContext:
        """
        Analyze a codebase and update the context with findings.
        
        Args:
            context: CodebaseContext to analyze and update
            
        Returns:
            Updated CodebaseContext with analysis results
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing codebase...", total=None)
            
            prompt = get_codebase_analyzer_prompt()
            
            try:
                result = await self.agent.run(prompt, deps=context)
                
                # Access the output property which contains the AnalysisResult
                analysis_result = result.output
                
                if not isinstance(analysis_result, AnalysisResult):
                    raise TypeError(f"Expected AnalysisResult, got {type(analysis_result)}")
                
                # Update context with findings
                context.project_type = analysis_result.project_type
                context.primary_language = analysis_result.primary_language
                
                progress.update(task, description="Codebase analysis complete!")
                return context
                
            except Exception as e:
                logger.error(f"Error analyzing codebase: {str(e)}")
                progress.update(task, description=f"Error: {str(e)}")
                raise


class FeatureSpecGeneratorAgent:
    """Agent specialized for generating feature specifications."""
    
    def __init__(self, settings: Settings, common_tools: CommonGeminiTools, model_name: Optional[str] = None):
        """Initialize the feature specification generator agent."""
        self.settings = settings
        self.common_tools = common_tools
        
        system_prompt = get_feature_spec_system_prompt()
        
        token_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192
        }
        
        self.agent = self.common_tools.create_pydantic_agent(
            model_name=model_name or "gemini-1.5-pro",
            token_config=token_config,
            deps_type=CodebaseContext,
            output_type=FeatureSpec,
            system_prompt_str=system_prompt
        )
        
        register_feature_tools(self.agent)

    async def generate(self, feature_description: str, context: CodebaseContext) -> FeatureSpec:
        """
        Generate a feature specification.
        
        Args:
            feature_description: Natural language description of the feature
            context: CodebaseContext with information about the codebase
            
        Returns:
            Detailed FeatureSpec
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating feature specification...", total=None)
            
            prompt = get_feature_spec_prompt(feature_description)
            
            try:
                result = await self.agent.run(prompt, deps=context)
                
                # Access the output property which contains the FeatureSpec
                feature_spec = result.output
                
                if not isinstance(feature_spec, FeatureSpec):
                    raise TypeError(f"Expected FeatureSpec, got {type(feature_spec)}")
                
                progress.update(task, description="Feature specification generated!")
                return feature_spec
                
            except Exception as e:
                logger.error(f"Error generating feature specification: {str(e)}")
                progress.update(task, description=f"Error: {str(e)}")
                raise


class ImplementationPlannerAgent:
    """Agent specialized for planning feature implementations."""
    
    def __init__(self, settings: Settings, common_tools: CommonGeminiTools, model_name: Optional[str] = None):
        """Initialize the implementation planner agent."""
        self.settings = settings
        self.common_tools = common_tools
        
        system_prompt = get_implementation_planner_system_prompt()
        
        token_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192
        }
        
        self.agent = self.common_tools.create_pydantic_agent(
            model_name=model_name or "gemini-1.5-pro",
            token_config=token_config,
            deps_type=CodebaseContext,
            output_type=ImplementationPlan,
            system_prompt_str=system_prompt
        )
        
        register_documentation_tools(self.agent)

    async def plan(self, feature_spec: FeatureSpec, context: CodebaseContext) -> ImplementationPlan:
        """
        Plan the implementation of a feature.
        
        Args:
            feature_spec: Detailed feature specification
            context: CodebaseContext with information about the codebase
            
        Returns:
            Detailed ImplementationPlan
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Planning implementation...", total=None)
            
            prompt = get_implementation_planner_prompt(feature_spec)
            
            try:
                result = await self.agent.run(prompt, deps=context)
                
                # Access the output property which contains the ImplementationPlan
                implementation_plan = result.output
                
                if not isinstance(implementation_plan, ImplementationPlan):
                    raise TypeError(f"Expected ImplementationPlan, got {type(implementation_plan)}")
                
                progress.update(task, description="Implementation plan generated!")
                return implementation_plan
                
            except Exception as e:
                logger.error(f"Error planning implementation: {str(e)}")
                progress.update(task, description=f"Error: {str(e)}")
                raise
