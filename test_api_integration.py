#!/usr/bin/env python3
"""
Integration test to verify that the agent can properly use the feature tools with real API.
"""

import asyncio
import logging
import os
from pathlib import Path
import sys
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")

from src.agent.agent import GeminiUpdateAgent
from src.models.feature import FeatureSpec
from src.models.analysis import ImplementationPlan, Settings

async def test_real_agent_with_api():
    """Test that the agent properly handles only Pydantic model objects in tool calls with real API."""
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Skipping real API test.")
        return
        
    logger.info("Starting real API integration test")
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Find the current project directory
        project_dir = Path(__file__).parent
        
        # Create settings using environment variables
        try:
            settings = Settings.from_env()
            settings.output_dir = output_dir
            settings.max_files_to_analyze = 10  # Limit for faster test
            logger.info(f"Using output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create settings: {e}")
            raise
            
        # Create the agent with real API
        try:
            agent = GeminiUpdateAgent(settings=settings)
            logger.info(f"Using model: {agent.model_name}")
            
            # Use a simple feature description for the test
            feature_description = "Add a simple function to log errors with stack traces"
            
            # Run the agent with real API
            logger.info(f"Starting agent with feature description: {feature_description}")
            feature_spec, impl_plan = await agent.perform_feature_update(
                feature_description=feature_description,
                project_dir=str(project_dir),
                output_dir=str(output_dir)
            )
            
            # Verify the results
            assert isinstance(feature_spec, FeatureSpec), "feature_spec is not a FeatureSpec instance"
            assert isinstance(impl_plan, ImplementationPlan), "impl_plan is not an ImplementationPlan instance"
            
            # Check if files were created
            spec_files = list(output_dir.glob("*_spec.md"))
            plan_files = list(output_dir.glob("*_plan.md"))
            
            assert len(spec_files) > 0, "No spec file was created"
            assert len(plan_files) > 0, "No plan file was created"
            
            logger.info(f"Spec file: {spec_files[0]}")
            logger.info(f"Plan file: {plan_files[0]}")
            
            # Check file contents
            with open(spec_files[0], 'r') as f:
                spec_content = f.read()
                assert "Feature Specification" in spec_content, "Invalid spec file content"
                logger.info(f"Feature spec content starts with: {spec_content[:200]}...")
            
            with open(plan_files[0], 'r') as f:
                plan_content = f.read()
                assert "Implementation Plan" in plan_content, "Invalid plan file content"
                logger.info(f"Implementation plan content starts with: {plan_content[:200]}...")
            
            logger.info("Real API integration test passed successfully!")
            
        except Exception as e:
            logger.error(f"Error during real API test: {str(e)}", exc_info=True)
            raise

async def main():
    """Run all tests."""
    await test_real_agent_with_api()

if __name__ == "__main__":
    asyncio.run(main())
