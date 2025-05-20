"""
Integration test for documentation tools with live Gemini API.

Run this script directly to test the documentation tools with the actual Gemini API.
Make sure you have set the GEMINI_API_KEY environment variable.

Example:
    $ export GEMINI_API_KEY=your_api_key
    $ python tests/test_documentation_tools_integration.py
"""

import asyncio
import os
import sys
from pathlib import Path
from pprint import pprint

from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.analysis import CodebaseContext, Settings
from src.tools.documentation_tools import register_documentation_tools


async def main():
    """Run a simple integration test with the actual Gemini API."""
    print("Documentation Tools Integration Test")
    print("====================================")
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Set it with: export GEMINI_API_KEY=your_api_key")
        sys.exit(1)
    
    print("Creating test environment...")
    
    # Create a settings object
    settings = Settings(
        gemini_api_key=api_key,
        gemini_model="gemini-1.5-pro",
        use_gemini_files=True,
    )
    
    # Create a codebase context
    codebase_ctx = CodebaseContext(
        project_dir=str(Path(__file__).parent.parent),
        primary_language="python",
        frameworks=["fastapi", "pytest"]
    )
    
    # Attach settings to the context
    codebase_ctx.settings = settings
    
    # Create an agent
    agent = Agent(str, CodebaseContext)
    
    # Register documentation tools
    register_documentation_tools(agent)
    
    # Create a run context
    run_ctx = RunContext(
        agent, 
        codebase_ctx, 
        args={"current_date": "2025-05-19"}
    )
    
    # Get the tools
    search_library_documentation = next(
        tool for tool in agent.tools if tool.__name__ == "search_library_documentation"
    )
    
    search_implementation_patterns = next(
        tool for tool in agent.tools if tool.__name__ == "search_implementation_patterns"
    )
    
    # Test library documentation search
    print("\nTesting search_library_documentation...")
    print("Searching for 'FastAPI routing' information...\n")
    
    try:
        doc_result = await search_library_documentation(
            run_ctx,
            library_name="fastapi",
            query="routing and path parameters"
        )
        
        print("Documentation Results:")
        print("---------------------")
        print(f"Library: {doc_result['library']}")
        print(f"Query: {doc_result['query']}")
        print("\nDocumentation:")
        print("--------------")
        print(doc_result['documentation'][:500] + "...\n")
        
        print("Sources:")
        print("--------")
        for i, source in enumerate(doc_result['sources'], 1):
            print(f"{i}. {source['title']}: {source['url'] or 'No URL'}")
        
    except Exception as e:
        print(f"Error testing library documentation search: {str(e)}")
    
    # Test implementation patterns search
    print("\nTesting search_implementation_patterns...")
    print("Searching for implementation patterns for user authentication...\n")
    
    try:
        pattern_result = await search_implementation_patterns(
            run_ctx,
            feature_description="implement user authentication with OAuth2",
            languages=["python"],
            frameworks=["fastapi"]
        )
        
        print("Implementation Patterns Results:")
        print("-------------------------------")
        print(f"Feature: {pattern_result['feature_description']}")
        print(f"Languages: {', '.join(pattern_result['languages'])}")
        print(f"Frameworks: {', '.join(pattern_result['frameworks'])}")
        print("\nPatterns:")
        print("---------")
        print(pattern_result['implementation_patterns'][:500] + "...\n")
        
        print("Sources:")
        print("--------")
        for i, source in enumerate(pattern_result['sources'], 1):
            print(f"{i}. {source['title']}: {source['url'] or 'No URL'}")
            
    except Exception as e:
        print(f"Error testing implementation patterns search: {str(e)}")
    
    print("\nIntegration test completed.")


if __name__ == "__main__":
    asyncio.run(main())
