"""
Main agent implementation for Gemini Update.
"""

import logging
import asyncio
import re
import sys
import os
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

import google.generativeai as genai
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.gemini import GeminiModelSettings
from pydantic_ai.usage import UsageLimits
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.models import CodebaseContext, ImplementationPlan, FeatureSpec, Settings
from src.tools.codebase_tools import register_tools
from src.tools.feature_tools import register_feature_tools
from src.tools.documentation_tools import register_documentation_tools


logger = logging.getLogger("gemini_update")
console = Console()


class GeminiUpdateAgent:
    """Agent for analyzing codebases and planning feature implementations."""

    def __init__(self, settings: Settings, model_name: Optional[str] = None, gemini_sync_manager: Optional[Any] = None):
        """
        Initialize the update agent with settings and create the Gemini agent.

        Args:
            settings: Application settings including API keys
            model_name: Optional name of the Gemini model to use. If None, will select automatically.
            gemini_sync_manager: Optional Gemini sync manager for Files API integration
        """
        self.settings = settings
        self.gemini_sync_manager = gemini_sync_manager
        
        # Initialize Google AI SDK
        genai.configure(api_key=settings.gemini_api_key)
        
        # Select the best available model if not specified
        self.model_name = model_name if model_name else self._select_best_model()
        
        # Create the Gemini agent
        self.agent = Agent[CodebaseContext, str](
            self.model_name,
            deps_type=CodebaseContext,
            settings=GeminiModelSettings(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
        )
        
        # Register tools with the agent
        register_tools(self.agent)
        register_feature_tools(self.agent)
        register_documentation_tools(self.agent)
        
    def _select_best_model(self) -> str:
        """
        Select the best available Gemini model.
        
        Returns:
            Name of the best available model
        """
        # Define models in order of preference
        preferred_models = [
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.0-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
        
        # Get available models
        try:
            available_models = [m.name for m in genai.list_models()]
            
            # Find the best available model
            for model in preferred_models:
                if model in available_models:
                    logger.info(f"Selected model: {model}")
                    return model
                    
            # If none of the preferred models are available, use the first available Gemini model
            for model in available_models:
                if "gemini" in model:
                    logger.info(f"Selected model: {model}")
                    return model
                    
            raise ValueError("No suitable Gemini models available.")
            
        except Exception as e:
            logger.error(f"Error selecting model: {str(e)}")
            # Default to gemini-1.5-pro as a fallback
            logger.info("Defaulting to gemini-1.5-pro")
            return "gemini-1.5-pro"
            
    async def analyze_codebase(self, project_dir: Union[str, Path]) -> CodebaseContext:
        """
        Analyze a codebase and create a CodebaseContext.
        
        Args:
            project_dir: Path to the project directory
            
        Returns:
            CodebaseContext with information about the codebase
        """
        project_path = Path(project_dir).resolve()
        if not project_path.exists() or not project_path.is_dir():
            raise ValueError(f"Project directory '{project_dir}' does not exist or is not a directory")
            
        logger.info(f"Analyzing codebase at {project_path}")
        
        # Create initial context with Gemini sync manager if available
        context = CodebaseContext(
            project_dir=project_path,
            gemini_sync_manager=self.gemini_sync_manager
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Add a task for the analysis
            task = progress.add_task("Analyzing codebase...", total=None)
            
            # Set up a prompt for the agent to analyze the codebase
            prompt = """
            Analyze the codebase and provide the following information:
            1. Project type (web app, CLI tool, library, etc.)
            2. Primary programming language
            3. Frameworks and libraries used
            4. Key files and their purposes
            5. Dependencies between files
            
            Use the available tools to explore the codebase.
            """
            
            # Run the agent with the analysis prompt
            try:
                await self.agent.run(prompt, deps=context)
                
                # Update task description when complete
                progress.update(task, description="Codebase analysis complete!")
                
            except Exception as e:
                logger.error(f"Error analyzing codebase: {str(e)}")
                progress.update(task, description=f"Error: {str(e)}")
                raise
                
        return context
        
    async def generate_feature_spec(
        self, 
        feature_description: str, 
        context: CodebaseContext
    ) -> FeatureSpec:
        """
        Generate a detailed feature specification from a description.
        
        Args:
            feature_description: Natural language description of the feature
            context: CodebaseContext with information about the codebase
            
        Returns:
            FeatureSpec with detailed requirements
        """
        logger.info(f"Generating feature specification for: {feature_description}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating feature specification...", total=None)
            
            # Set up a prompt for the agent to generate a feature specification
            prompt = f"""
            Based on the following feature description and the analyzed codebase,
            generate a detailed feature specification with requirements:
            
            Feature: {feature_description}
            
            The specification should include:
            - Feature name and description
            - Feature type and priority
            - Detailed requirements with acceptance criteria
            - User personas affected
            - Success metrics
            - Technical notes or considerations
            
            Format the response as a structured FeatureSpec.
            """
            
            # Run the agent with the feature specification prompt
            try:
                result = await self.agent.run(prompt, deps=context)
                
                # Parse the result as a FeatureSpec
                feature_spec = FeatureSpec.model_validate_json(result)
                
                progress.update(task, description="Feature specification generated!")
                
                return feature_spec
                
            except Exception as e:
                logger.error(f"Error generating feature specification: {str(e)}")
                progress.update(task, description=f"Error: {str(e)}")
                raise
                
    async def plan_implementation(
        self,
        feature_spec: FeatureSpec,
        context: CodebaseContext
    ) -> ImplementationPlan:
        """
        Plan the implementation of a feature in the codebase.
        
        Args:
            feature_spec: Detailed feature specification
            context: CodebaseContext with information about the codebase
            
        Returns:
            ImplementationPlan with detailed steps
        """
        logger.info(f"Planning implementation for feature: {feature_spec.name}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Planning implementation...", total=None)
            
            # Set up a prompt for the agent to create an implementation plan
            prompt = f"""
            Based on the following feature specification and the analyzed codebase,
            create a detailed implementation plan:
            
            {feature_spec.to_markdown()}
            
            The implementation plan should include:
            - Files that need to be modified
            - New files that need to be created
            - Specific code changes with descriptions and snippets
            - Dependencies between changes
            - Estimated complexity
            
            Format the response as a structured ImplementationPlan.
            """
            
            # Run the agent with the implementation planning prompt
            try:
                result = await self.agent.run(prompt, deps=context)
                
                # Parse the result as an ImplementationPlan
                plan = ImplementationPlan.model_validate_json(result)
                
                progress.update(task, description="Implementation plan generated!")
                
                return plan
                
            except Exception as e:
                logger.error(f"Error planning implementation: {str(e)}")
                progress.update(task, description=f"Error: {str(e)}")
                raise
                
    async def perform_feature_update(
        self,
        feature_description: str,
        project_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[FeatureSpec, ImplementationPlan]:
        """
        End-to-end process to analyze a codebase, generate a feature spec, and plan implementation.
        
        Args:
            feature_description: Natural language description of the feature
            project_dir: Path to the project directory
            output_dir: Optional directory to save output files
            
        Returns:
            Tuple of (FeatureSpec, ImplementationPlan)
        """
        # Determine output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.settings.output_dir
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Analyze the codebase
        console.print("\n[bold blue]Step 1: Analyzing codebase...[/bold blue]")
        context = await self.analyze_codebase(project_dir)
        
        # 2. Generate feature specification
        console.print("\n[bold blue]Step 2: Generating feature specification...[/bold blue]")
        feature_spec = await self.generate_feature_spec(feature_description, context)
        
        # Save feature specification to file
        feature_spec_path = output_path / f"{feature_spec.name.lower().replace(' ', '_')}_spec.md"
        with open(feature_spec_path, "w") as f:
            f.write(feature_spec.to_markdown())
        console.print(f"Feature specification saved to: [green]{feature_spec_path}[/green]")
        
        # 3. Plan implementation
        console.print("\n[bold blue]Step 3: Planning implementation...[/bold blue]")
        implementation_plan = await self.plan_implementation(feature_spec, context)
        
        # Save implementation plan to file
        implementation_plan_path = output_path / f"{feature_spec.name.lower().replace(' ', '_')}_plan.md"
        with open(implementation_plan_path, "w") as f:
            f.write(implementation_plan.to_markdown())
        console.print(f"Implementation plan saved to: [green]{implementation_plan_path}[/green]")
        
        return feature_spec, implementation_plan
