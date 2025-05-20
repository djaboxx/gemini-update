#!/usr/bin/env python3
"""
Test script to verify that the model objects are handled correctly and string parsing is disabled.
"""

import asyncio
import os
import json
import pytest
from pathlib import Path
import tempfile
import logging
from unittest.mock import patch, MagicMock

try:
    from dotenv import load_dotenv
except ImportError:
    print("WARNING: python-dotenv not installed. Install with: pip install python-dotenv")

from src.models.feature import FeatureSpec, Requirement, RequirementType, FeatureType, Priority
from src.models.analysis import ImplementationPlan, CodeChange, ChangeType, FeatureScope, Settings
from src.agent.agent import GeminiUpdateAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_feature_spec_handling():
    """
    Test that FeatureSpec correctly handles serialization and deserialization.
    """
    print("Testing FeatureSpec handling...")
    
    # Create a test feature spec
    feature_spec = FeatureSpec(
        name="Test Feature",
        description="A test feature to verify function signature changes",
        feature_type=FeatureType.UI,
        priority=Priority.MEDIUM,
        requirements=[
            Requirement(
                id="REQ-001",
                description="A test requirement",
                type=RequirementType.FUNCTIONAL,
                acceptance_criteria=["Test passes"]
            )
        ],
        user_personas=["Tester"],
        success_metrics=["Test completes successfully"],
        technical_notes="Testing the fix for passing Pydantic models to tools"
    )
    
    # Serialize to JSON string
    json_str = feature_spec.model_dump_json()
    print(f"Serialized FeatureSpec: {json_str[:100]}...")
    
    # Deserialize back to object
    deserialized_spec = FeatureSpec.model_validate_json(json_str)
    
    # Verify fields match
    assert deserialized_spec.name == feature_spec.name, "Name field doesn't match after serialization"
    assert deserialized_spec.description == feature_spec.description, "Description field doesn't match"
    assert len(deserialized_spec.requirements) == len(feature_spec.requirements), "Requirements count doesn't match"
    
    print("Feature spec test passed!")

def test_implementation_plan_handling():
    """
    Test that ImplementationPlan correctly handles serialization and deserialization.
    """
    print("Testing ImplementationPlan handling...")
    
    # Create a test implementation plan
    impl_plan = ImplementationPlan(
        feature_name="Test Feature",
        description="Implementation plan for test feature",
        scope=FeatureScope(
            affected_files=["test_file.py"],
            new_files=[],
            dependencies_needed=[],
            config_changes=[]
        ),
        changes=[
            CodeChange(
                file_path="test_file.py",
                change_type=ChangeType.MODIFY,
                description="Test change",
                code_snippet="print('Hello world')"
            )
        ],
        estimated_complexity="Low",
        dependencies=[]
    )
    
    # Serialize to JSON string
    json_str = impl_plan.model_dump_json()
    print(f"Serialized ImplementationPlan: {json_str[:100]}...")
    
    # Deserialize back to object
    deserialized_plan = ImplementationPlan.model_validate_json(json_str)
    
    # Verify fields match
    assert deserialized_plan.feature_name == impl_plan.feature_name, "Feature name field doesn't match after serialization"
    assert deserialized_plan.description == impl_plan.description, "Description field doesn't match"
    assert len(deserialized_plan.changes) == len(impl_plan.changes), "Changes count doesn't match"
    
    print("Implementation plan test passed!")

async def test_real_agent_with_string_injection():
    """Test that the real agent properly rejects string responses for FeatureSpec and ImplementationPlan."""
    print("Testing real agent with string injection...")
    
    try:
        # Load environment variables - API key should be in the environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Create settings for the real agent
        settings = Settings.from_env()
        settings.output_dir = Path(tempfile.mkdtemp())
        
        # Let the agent select the best model instead of using the one from settings
        # The _select_best_model method will choose an appropriate model
        settings.gemini_model = None
        print("Note: Using agent's model selection instead of environment setting")
        
        print(f"Using output directory: {settings.output_dir}")
        print(f"Using model: {settings.gemini_model}")
        
        # Create the real GeminiUpdateAgent with actual settings
        agent = GeminiUpdateAgent(settings=settings)
        
        # Create a context for testing
        context = MagicMock()
        context.language = "Python"
        context.files = []
        context.project_path = str(Path.cwd())
        
        # Override the agent's run method to return strings instead of objects
        original_run = agent.agent.run
        
        # First, test with FeatureSpec
        async def mock_feature_spec_string(*args, **kwargs):
            # Return a valid JSON string instead of a FeatureSpec object
            return """
            {
                "name": "Test Feature",
                "description": "A test feature",
                "feature_type": "ui",
                "priority": "medium",
                "requirements": [
                    {
                        "id": "REQ-001",
                        "type": "functional",
                        "description": "Test requirement",
                        "acceptance_criteria": ["Test passes"]
                    }
                ]
            }
            """
        
        # Replace the run method temporarily
        agent.agent.run = mock_feature_spec_string
        
        # Test that generate_feature_spec raises TypeError
        print("Testing FeatureSpec with string injection...")
        try:
            await agent.generate_feature_spec("Test feature", context)
            assert False, "Should have raised TypeError"
        except TypeError as e:
            print(f"Successfully rejected string for FeatureSpec: {e}")
            assert "Expected FeatureSpec object" in str(e)
        
        # Now test with ImplementationPlan
        async def mock_implementation_plan_string(*args, **kwargs):
            # Return a valid JSON string instead of an ImplementationPlan object
            return """
            {
                "feature_name": "Test Feature",
                "description": "Implementation plan",
                "scope": {
                    "affected_files": ["test.py"],
                    "new_files": [],
                    "dependencies_needed": [],
                    "config_changes": []
                },
                "changes": [
                    {
                        "file_path": "test.py",
                        "change_type": "modify",
                        "description": "Test change",
                        "code_snippet": "print('hello')"
                    }
                ],
                "estimated_complexity": "Low",
                "dependencies": []
            }
            """
        
        # Replace the run method for implementation plan test
        agent.agent.run = mock_implementation_plan_string
        
        # Test that plan_implementation raises TypeError
        print("Testing ImplementationPlan with string injection...")
        feature_spec = FeatureSpec(
            name="Test Feature",
            description="A test feature",
            feature_type=FeatureType.UI,
            priority=Priority.MEDIUM,
            requirements=[
                Requirement(
                    id="REQ-001",
                    description="Test requirement",
                    type=RequirementType.FUNCTIONAL,
                    acceptance_criteria=["Test passes"]
                )
            ],
            user_personas=["Tester"],
            success_metrics=["Success"]
        )
        
        try:
            await agent.plan_implementation(feature_spec, context)
            assert False, "Should have raised TypeError"
        except TypeError as e:
            print(f"Successfully rejected string for ImplementationPlan: {e}")
            assert "Expected ImplementationPlan object" in str(e)
        
        # Restore the original run method
        agent.agent.run = original_run
        
        print("Real agent test passed!")
    
    except Exception as e:
        print(f"Error during real agent test: {e}")
        raise

def main():
    """Run all tests."""
    # Setup environment for tests
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY environment variable not set. API tests will be skipped.")
        print("Set the GEMINI_API_KEY environment variable to run all tests.")
        run_api_tests = False
    else:
        run_api_tests = True
    
    # Run model serialization tests
    test_feature_spec_handling()
    print("-" * 50)
    test_implementation_plan_handling()
    
    # Run API tests if API key is available
    if run_api_tests:
        print("-" * 50)
        print("Running API tests...")
        # Run the async test using asyncio
        asyncio.run(test_real_agent_with_string_injection())
    else:
        print("-" * 50)
        print("Skipping API tests due to missing API key.")

if __name__ == "__main__":
    main()
