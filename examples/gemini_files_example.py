"""
Example demonstrating the Gemini Files API integration with feature analysis.
"""

import asyncio
import os
import sys
from pathlib import Path

from src.agent.agent import GeminiUpdateAgent
from src.agent.gemini_files import GeminiFileManager
from src.models import Settings


async def main():
    """Run a feature analysis with Gemini Files API integration."""
    # Check command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python examples/gemini_files_example.py <project_directory> <feature_description>")
        sys.exit(1)
    
    project_dir = Path(sys.argv[1]).resolve()
    if not project_dir.exists() or not project_dir.is_dir():
        print(f"Error: Project directory '{project_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    feature_description = sys.argv[2]
    
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it before running this example.")
        sys.exit(1)
    
    # Create settings
    settings = Settings(
        gemini_api_key=api_key,
        output_dir=Path.cwd(),
        use_gemini_files=True,
        max_file_size_mb=4.0,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    )
    
    print("Initializing Gemini Files API integration...")
    
    # Create Gemini file manager
    gemini_file_manager = GeminiFileManager(
        api_key=api_key,
        project_dir=project_dir,
        max_file_size_mb=4.0
    )
    
    print("Creating agent with Gemini Files support...")
    
    # Create the agent with Gemini Files support
    agent = GeminiUpdateAgent(
        settings=settings,
        gemini_sync_manager=gemini_file_manager
    )
    
    print(f"Analyzing codebase at {project_dir}...")
    
    # Analyze the codebase
    codebase_context = await agent.analyze_codebase(project_dir)
    
    print("Analysis complete!")
    
    # Use identify_affected_files to find relevant files
    print(f"Identifying files for feature: {feature_description}")
    
    # Create a run context for the feature tools
    run_ctx = await agent.create_run_context(codebase_context)
    affected_files = await agent.agent.execute(
        "identify_affected_files",
        ctx=run_ctx,
        feature_description=feature_description
    )
    
    print("\nAffected files identified:")
    for file in affected_files:
        print(f"- {file}")
    
    # Generate feature spec and implementation plan
    print("\nGenerating feature implementation plan...")
    feature_spec, implementation_plan = await agent.perform_feature_update(
        feature_description=feature_description,
        project_dir=project_dir,
        output_dir=settings.output_dir
    )
    
    print("\nFeature planning complete!")
    print(f"Feature Name: {feature_spec.name}")
    print(f"Type: {feature_spec.feature_type.value}")
    print(f"Priority: {feature_spec.priority.value}")
    print(f"Complexity: {implementation_plan.estimated_complexity}")
    print(f"Files to modify: {len(implementation_plan.scope.affected_files)}")
    print(f"New files to create: {len(implementation_plan.scope.new_files)}")
    
    # Save the implementation plan to a file
    output_file = settings.output_dir / "implementation_plan.md"
    with open(output_file, "w") as f:
        f.write(implementation_plan.to_markdown())
    
    print(f"\nImplementation plan saved to: {output_file}")
    
    # Clean up Gemini Files API resources
    print("\nCleaning up Gemini Files API resources...")
    deleted_count = gemini_file_manager.clear_all_files()
    print(f"Deleted {deleted_count} files from Gemini Files API.")


if __name__ == "__main__":
    asyncio.run(main())
