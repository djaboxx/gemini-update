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


logger = logging.getLogger("gemini_update")
console = Console()


class CodebaseAnalyzerAgent:
    """Agent specialized for analyzing codebases."""
    
    def __init__(self, settings: Settings, common_tools: CommonGeminiTools, model_name: Optional[str] = None):
        """Initialize the codebase analyzer agent."""
        self.settings = settings
        self.common_tools = common_tools
        
        system_prompt = """
        You are a specialized code analysis agent. Your role is to analyze codebases and identify:
        1. Project type (web app, CLI tool, library, agent, etc.)
        2. Primary programming language
        3. Frameworks and libraries used
        4. Key files and their purposes
        5. Dependencies between files
        
        Use the available tools to explore the codebase systematically:
        - First, use get_project_info to get basic stats and language distribution
        - Use execute_gemini_code to write and run custom Python code that analyzes the project structure
          * This tool allows you to write and execute your own Python code to analyze the codebase
          * You can write code to inspect file patterns, imports, and framework usage
          * It's more flexible than the older execute_code_analysis tool
          
        - Use analyze_codebase_with_gemini to perform a complete codebase analysis by writing Python code
        - Analyze the codebase at multiple levels (structure, imports, code patterns)
        - Write custom code to detect patterns specific to different project types
        - For Python projects, analyze imports to differentiate between web apps, CLI tools, data science, and AI agents
        - Pay special attention to import statements, entry points, and framework usage patterns
        - Document your findings in the returned AnalysisResult
        
        The execute_gemini_code tool allows you complete freedom to run custom Python code to analyze the codebase.
        This is the preferred way to gain deep insights into the codebase structure and patterns.
        If your code has errors, you'll get detailed feedback and can fix and retry your solution.
        """

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
            
            prompt = """
            Analyze this codebase thoroughly to determine its structure and purpose:
            
            1. First, gather basic project information using get_project_info to understand language distribution and identify key files
            
            2. Write and execute custom analysis code with execute_gemini_code to:
               - Inspect file structures and directory organization
               - Analyze import statements and dependencies
               - Detect framework usage and code patterns
               - Identify specific patterns that indicate project types
               
               For example, you can write Python code that:
               - Scans directory structures to detect framework-specific patterns
               - Examines Python imports to identify key libraries and frameworks
               - Searches for configuration files that indicate specific project types
               - Analyzes entry point files to determine application structure
               
            3. You can use analyze_codebase_with_gemini for a complete end-to-end analysis
               - This tool lets you write a complete analysis script that examines the entire codebase
               - It's useful for handling complex project type detection
            
            4. Pay special attention to distinguishing between different project types:
               - For Python projects, differentiate between web apps, CLI tools, data science, and AI agents
               - Look beyond simple file extensions to identify actual usage patterns
               - Examine imports, class structures, and coding patterns
            
            5. Identify entry points and key structural components
            
            Use these insights to accurately classify the project type and return an AnalysisResult object.
            """
            
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
        
        system_prompt = """
        You are a specialized feature specification agent. Your role is to:
        1. Analyze feature descriptions and codebase context
        2. Generate detailed, actionable feature specifications
        3. Define clear requirements and acceptance criteria
        4. Identify affected user personas
        5. Specify success metrics
        
        Your outputs must be structured FeatureSpec objects with:
        - Feature name and clear description
        - Feature type and priority level
        - Detailed requirements list
        - Acceptance criteria for each requirement
        - Relevant user personas
        - Measurable success metrics
        - Technical notes considering the codebase context
        """
        
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
            
            prompt = f"""
            Based on the following feature description and the analyzed codebase,
            generate a detailed feature specification with requirements:
            
            Feature: {feature_description}
            
            Include:
            - Feature name and description
            - Feature type and priority
            - Detailed requirements with acceptance criteria
            - User personas affected
            - Success metrics
            - Technical notes or considerations
            
            Respond with a valid FeatureSpec object.
            """
            
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
        
        system_prompt = """
        You are a specialized implementation planning agent. Your role is to:
        1. Analyze feature specifications in the context of the codebase
        2. Plan detailed implementation steps
        3. Identify files requiring changes
        4. Generate specific code changes
        5. Define dependencies between changes
        
        Your outputs must be structured ImplementationPlan objects with:
        - Clear scope of changes (files, dependencies)
        - Detailed code changes with snippets
        - Dependencies between changes
        - Complexity estimates
        - Implementation steps in the correct order
        """
        
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
            
            prompt = f"""
            Based on the following feature specification and the analyzed codebase,
            create a detailed implementation plan:
            
            {feature_spec.to_markdown()}
            
            Include:
            - Files that need to be modified
            - New files that need to be created
            - Specific code changes with descriptions and snippets
            - Dependencies between changes
            - Estimated complexity
            
            Respond with a valid ImplementationPlan object.
            """
            
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
