"""
Command-line interface for the Gemini Update agent.
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from src.agent.agent import GeminiUpdateAgent
from src.models import Settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("gemini_update")
console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Gemini Update - Analyze codebases and plan feature implementations"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--gemini-api-key",
        type=str,
        help="Google Gemini API key (defaults to GEMINI_API_KEY env var)",
    )
    common_parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save output files (defaults to current directory)",
    )
    common_parser.add_argument(
        "--use-gemini-files",
        action="store_true",
        help="Enable Gemini Files API for code analysis"
    )
    common_parser.add_argument(
        "--max-file-size",
        type=float,
        default=4.0,
        help="Maximum file size in MB for Gemini Files uploads (default: 4.0)",
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a codebase",
        parents=[common_parser],
    )
    analyze_parser.add_argument(
        "--project-dir",
        type=str,
        required=True,
        help="Path to the project directory to analyze",
    )
    
    # Feature command
    feature_parser = subparsers.add_parser(
        "feature",
        help="Generate a feature specification and implementation plan",
        parents=[common_parser],
    )
    feature_parser.add_argument(
        "--project-dir",
        type=str,
        required=True,
        help="Path to the project directory to analyze",
    )
    feature_parser.add_argument(
        "--feature-description",
        type=str,
        required=True,
        help="Description of the feature to implement",
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )
    
    return parser


def get_settings(args: argparse.Namespace) -> Settings:
    """Get application settings from args and environment variables."""
    try:
        # Start with environment-based settings
        settings = Settings.from_env()
        
        # Override with command-line arguments if provided
        if hasattr(args, "gemini_api_key") and args.gemini_api_key:
            settings.gemini_api_key = args.gemini_api_key
            
        if hasattr(args, "output_dir") and args.output_dir:
            settings.output_dir = Path(args.output_dir).resolve()
            
        if hasattr(args, "use_gemini_files"):
            settings.use_gemini_files = args.use_gemini_files
            
        if hasattr(args, "max_file_size"):
            settings.max_file_size_mb = args.max_file_size
            
        if hasattr(args, "model") and args.model:
            settings.gemini_model = args.model
        
        # Create output directory if it doesn't exist
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        
        return settings
        
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print("Please provide the Gemini API key with --gemini-api-key or set the GEMINI_API_KEY environment variable.")
        sys.exit(1)


async def handle_analyze_command(args: argparse.Namespace) -> int:
    """Handle 'analyze' command."""
    settings = get_settings(args)
    
    # Initialize Gemini sync manager if enabled
    gemini_sync_manager = None
    if settings.use_gemini_files:
        from src.agent.gemini_files import GeminiFileManager
        console.print("[green]Initializing Gemini Files API integration...[/green]")
        project_dir = Path(args.project_dir).resolve()
        gemini_sync_manager = GeminiFileManager(
            api_key=settings.gemini_api_key,
            project_dir=project_dir,
            max_file_size_mb=settings.max_file_size_mb
        )
    
    # Create the agent
    agent = GeminiUpdateAgent(
        settings=settings,
        model_name=settings.gemini_model,
        gemini_sync_manager=gemini_sync_manager
    )
    
    # Analyze the codebase
    try:
        project_dir = Path(args.project_dir).resolve()
        if not project_dir.exists() or not project_dir.is_dir():
            console.print(f"[red]Error: Project directory '{args.project_dir}' does not exist or is not a directory.[/red]")
            return 1
            
        console.print(f"[green]Analyzing codebase at {project_dir}...[/green]")
        codebase_context = await agent.analyze_codebase(project_dir)
        
        # Print some information about the analysis
        console.print("\n[bold green]Analysis complete![/bold green]")
        console.print(f"Project type: [cyan]{codebase_context.project_type or 'Unknown'}[/cyan]")
        console.print(f"Primary language: [cyan]{codebase_context.primary_language or 'Unknown'}[/cyan]")
        
        if codebase_context.frameworks:
            console.print("Frameworks: [cyan]" + ", ".join(codebase_context.frameworks) + "[/cyan]")
            
        console.print(f"Files analyzed: [cyan]{len(codebase_context.files)}[/cyan]")
        
        # Save analysis report to file
        output_file = settings.output_dir / "codebase_analysis.md"
        with open(output_file, "w") as f:
            f.write("# Codebase Analysis Report\n\n")
            f.write(f"**Project Type:** {codebase_context.project_type or 'Unknown'}\n\n")
            f.write(f"**Primary Language:** {codebase_context.primary_language or 'Unknown'}\n\n")
            
            if codebase_context.frameworks:
                f.write("**Frameworks:**\n\n")
                for framework in codebase_context.frameworks:
                    f.write(f"- {framework}\n")
                f.write("\n")
                
            f.write(f"**Files Analyzed:** {len(codebase_context.files)}\n\n")
            
            if codebase_context.dependencies:
                f.write("**Key Dependencies:**\n\n")
                for dep in codebase_context.dependencies[:10]:  # Show top 10 dependencies
                    f.write(f"- {dep.source} → {dep.target} ({dep.dependency_type})\n")
                    
        console.print(f"Analysis report saved to: [green]{output_file}[/green]")
        
        return 0
        
    except Exception as e:
        console.print(f"[red]Error analyzing codebase: {str(e)}[/red]")
        logger.exception("Error in analyze command")
        return 1


async def handle_feature_command(args: argparse.Namespace) -> int:
    """Handle 'feature' command."""
    settings = get_settings(args)
    
    # Initialize Gemini sync manager if enabled
    gemini_sync_manager = None
    if settings.use_gemini_files:
        from src.agent.gemini_files import GeminiFileManager
        console.print("[green]Initializing Gemini Files API integration...[/green]")
        project_dir = Path(args.project_dir).resolve()
        gemini_sync_manager = GeminiFileManager(
            api_key=settings.gemini_api_key,
            project_dir=project_dir,
            max_file_size_mb=settings.max_file_size_mb
        )
    
    # Create the agent
    agent = GeminiUpdateAgent(
        settings=settings,
        model_name=settings.gemini_model,
        gemini_sync_manager=gemini_sync_manager
    )
    
    # Generate feature spec and implementation plan
    try:
        project_dir = Path(args.project_dir).resolve()
        if not project_dir.exists() or not project_dir.is_dir():
            console.print(f"[red]Error: Project directory '{args.project_dir}' does not exist or is not a directory.[/red]")
            return 1
            
        console.print(f"[green]Analyzing project and generating feature implementation plan...[/green]")
        console.print(f"Feature: [cyan]{args.feature_description}[/cyan]\n")
        
        # Perform the feature update
        feature_spec, implementation_plan = await agent.perform_feature_update(
            feature_description=args.feature_description,
            project_dir=project_dir,
            output_dir=settings.output_dir
        )
        
        # Report success
        console.print("\n[bold green]Feature planning complete![/bold green]")
        console.print(f"Feature Name: [cyan]{feature_spec.name}[/cyan]")
        console.print(f"Type: [cyan]{feature_spec.feature_type.value}[/cyan]")
        console.print(f"Priority: [cyan]{feature_spec.priority.value}[/cyan]")
        console.print(f"Complexity: [cyan]{implementation_plan.estimated_complexity}[/cyan]")
        console.print(f"Files to modify: [cyan]{len(implementation_plan.scope.affected_files)}[/cyan]")
        console.print(f"New files to create: [cyan]{len(implementation_plan.scope.new_files)}[/cyan]")
        
        return 0
        
    except Exception as e:
        console.print(f"[red]Error generating feature plan: {str(e)}[/red]")
        logger.exception("Error in feature command")
        return 1


def handle_version_command(_args: argparse.Namespace) -> int:
    """Handle 'version' command."""
    console.print("Gemini Update v0.1.0")
    console.print("A tool to analyze codebases and plan feature implementations using Google Gemini AI")
    return 0


async def main_async() -> int:
    """Asynchronous entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "analyze":
        return await handle_analyze_command(args)
    elif args.command == "feature":
        return await handle_feature_command(args)
    elif args.command == "version":
        return handle_version_command(args)
    else:
        parser.print_help()
        return 1


def main() -> int:
    """Synchronous entry point for the CLI."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
