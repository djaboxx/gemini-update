"""
Demo script to show the capabilities of the Gemini Update Agent.

This script demonstrates how to use the GeminiUpdateAgent
to analyze a codebase and plan a feature implementation.

Usage:
    python demo.py

Requirements:
    - Set the GEMINI_API_KEY environment variable
    - Python 3.11+
    - Required packages listed in requirements.txt
"""

import asyncio
import os
from pathlib import Path
import sys

# Add the repository root to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from src.agent.agent import GeminiUpdateAgent
from src.models import Settings
from examples.example_feature_spec import create_example_feature_spec
from examples.example_implementation_plan import create_example_implementation_plan

from rich.console import Console
from rich.markdown import Markdown


# Initialize console
console = Console()


async def demo_feature_spec_generation():
    """Demonstrate how to generate a feature spec."""
    console.print("[bold blue]Demonstration: Generate Feature Specification[/bold blue]\n")
    
    # Create an example feature spec
    feature_spec = create_example_feature_spec()
    
    # Convert to markdown
    markdown = feature_spec.to_markdown()
    
    # Display the markdown
    console.print(Markdown(markdown))
    console.print("\n[green]Feature specification generated successfully![/green]\n")


async def demo_implementation_plan_generation():
    """Demonstrate how to generate an implementation plan."""
    console.print("[bold blue]Demonstration: Generate Implementation Plan[/bold blue]\n")
    
    # Create an example implementation plan
    implementation_plan = create_example_implementation_plan()
    
    # Convert to markdown
    markdown = implementation_plan.to_markdown()
    
    # Display the markdown
    console.print(Markdown(markdown))
    console.print("\n[green]Implementation plan generated successfully![/green]\n")


async def demo_live_agent(project_dir=None):
    """
    Demonstrate a live agent interaction (if API key is available).
    
    Args:
        project_dir: Optional path to a project directory to analyze
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        console.print("[yellow]Note: GEMINI_API_KEY environment variable not set.[/yellow]")
        console.print("[yellow]Skipping live agent demonstration.[/yellow]\n")
        return
        
    console.print("[bold blue]Demonstration: Live Agent Interaction[/bold blue]\n")
    
    # Use the current directory if no project_dir is specified
    if project_dir is None:
        project_dir = os.getcwd()
        
    # Create a settings object    
    settings = Settings(
        gemini_api_key=api_key,
        output_dir=Path("./demo_output"),
    )
    
    # Create the output directory if it doesn't exist
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the agent
    agent = GeminiUpdateAgent(settings=settings)
    
    # Feature description
    feature_description = "Add a simple logging system with different log levels (info, warning, error) and file output"
    
    console.print(f"[green]Analyzing project and planning implementation for:[/green]")
    console.print(f"[cyan]{feature_description}[/cyan]\n")
    
    try:
        # Perform the feature update
        feature_spec, implementation_plan = await agent.perform_feature_update(
            feature_description=feature_description,
            project_dir=project_dir,
            output_dir=settings.output_dir
        )
        
        console.print("\n[bold green]Live agent demonstration completed successfully![/bold green]")
        console.print(f"Output files saved to [cyan]{settings.output_dir}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error during live agent demonstration: {str(e)}[/red]")


async def main():
    """Run the demonstration."""
    console.print("[bold green]===== Gemini Update Agent Demonstration =====[/bold green]\n")
    
    # Demo feature specification generation
    await demo_feature_spec_generation()
    
    # Add a separator
    console.print("\n" + "=" * 50 + "\n")
    
    # Demo implementation plan generation
    await demo_implementation_plan_generation()
    
    # Add a separator
    console.print("\n" + "=" * 50 + "\n")
    
    # Demo live agent (if API key is available)
    await demo_live_agent()
    
    console.print("\n[bold green]===== Demonstration Complete =====[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
