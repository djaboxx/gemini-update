"""
Tests for documentation search tools using Gemini's grounded search capabilities.
"""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest
import google.generativeai as genai

from pydantic_ai import Agent, RunContext
from src.models.analysis import CodebaseContext, Settings
from src.tools.documentation_tools import (
    register_documentation_tools,
    LibraryDocumentationResponse,
    ImplementationPatternsResponse,
    SourceReference
)


class TestDocumentationTools(unittest.TestCase):
    """Test suite for documentation search tools."""

    def setUp(self):
        """Set up test environment with a mock agent and context."""
        # Create a mock agent
        self.agent = Agent(str, CodebaseContext)
        
        # Register the documentation tools
        register_documentation_tools(self.agent)
        
        # Create mock settings with a test API key
        self.settings = Settings(
            gemini_api_key="test_api_key_12345",
            gemini_model="gemini-1.5-pro",
            use_gemini_files=True
        )
        
        # Create a mock CodebaseContext
        self.codebase_ctx = CodebaseContext(
            project_dir=os.path.dirname(__file__),
            primary_language="python",
            frameworks=["fastapi", "pytest"]
        )
        
        # Attach the settings to the context
        self.codebase_ctx.settings = self.settings

    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_search_library_documentation(self, mock_gen_model):
        """Test search_library_documentation tool with mocked Gemini API."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.text = "FastAPI is a modern, fast web framework for building APIs with Python."
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].grounding_metadata.segments = [MagicMock()]
        
        # Mock source object
        mock_source = MagicMock()
        mock_source.title = "FastAPI Documentation"
        mock_source.uri = "https://fastapi.tiangolo.com/"
        
        # Add mock source to segments
        mock_response.candidates[0].grounding_metadata.segments[0].sources = [mock_source]
        
        # Set up the mock model to return our mock response
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model_instance
        
        # Create a mock run context
        run_ctx = RunContext(
            self.agent, 
            self.codebase_ctx, 
            args={"current_date": "2025-05-19"}
        )
        
        # Get the search_library_documentation tool
        search_library_documentation = next(
            tool for tool in self.agent.tools if tool.__name__ == "search_library_documentation"
        )
        
        # Call the tool
        result = await search_library_documentation(
            run_ctx,
            library_name="fastapi",
            query="how to create a simple API"
        )
        
        # Verify the result
        self.assertIsInstance(result, dict)
        self.assertEqual(result["library"], "fastapi")
        self.assertEqual(result["query"], "how to create a simple API")
        self.assertEqual(result["documentation"], mock_response.text)
        
        # Verify sources were extracted
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["title"], "FastAPI Documentation")
        self.assertEqual(result["sources"][0]["url"], "https://fastapi.tiangolo.com/")
        
        # Verify the generate_content call
        mock_model_instance.generate_content.assert_called_once()
        call_args = mock_model_instance.generate_content.call_args[0][0]
        self.assertIn("fastapi", call_args)
        self.assertIn("how to create a simple API", call_args)

    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    async def test_search_implementation_patterns(self, mock_gen_model):
        """Test search_implementation_patterns tool with mocked Gemini API."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.text = "## Repository Pattern\nThe Repository pattern isolates data access logic..."
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].grounding_metadata.segments = [MagicMock()]
        
        # Mock source objects
        mock_source1 = MagicMock()
        mock_source1.title = "Design Patterns in Python"
        mock_source1.uri = "https://refactoring.guru/design-patterns/python"
        
        mock_source2 = MagicMock()
        mock_source2.title = "FastAPI Best Practices"
        mock_source2.uri = "https://fastapi.tiangolo.com/tutorial/"
        
        # Add mock sources to segments
        mock_response.candidates[0].grounding_metadata.segments[0].sources = [mock_source1, mock_source2]
        
        # Set up the mock model to return our mock response
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model_instance
        
        # Create a mock run context
        run_ctx = RunContext(
            self.agent, 
            self.codebase_ctx, 
            args={"current_date": "2025-05-19"}
        )
        
        # Get the search_implementation_patterns tool
        search_implementation_patterns = next(
            tool for tool in self.agent.tools if tool.__name__ == "search_implementation_patterns"
        )
        
        # Call the tool
        result = await search_implementation_patterns(
            run_ctx,
            feature_description="implement user authentication",
            languages=["python"],
            frameworks=["fastapi"]
        )
        
        # Verify the result
        self.assertIsInstance(result, dict)
        self.assertEqual(result["feature_description"], "implement user authentication")
        self.assertEqual(result["languages"], ["python"])
        self.assertEqual(result["frameworks"], ["fastapi"])
        self.assertEqual(result["implementation_patterns"], mock_response.text)
        
        # Verify sources were extracted
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["title"], "Design Patterns in Python")
        self.assertEqual(result["sources"][0]["url"], "https://refactoring.guru/design-patterns/python")
        
        # Verify the generate_content call
        mock_model_instance.generate_content.assert_called_once()
        call_args = mock_model_instance.generate_content.call_args[0][0]
        self.assertIn("implement user authentication", call_args)
        self.assertIn("python", call_args)
        self.assertIn("fastapi", call_args)

    def test_library_documentation_response_model(self):
        """Test the LibraryDocumentationResponse Pydantic model."""
        source = SourceReference(title="Documentation", url="https://example.com")
        response = LibraryDocumentationResponse(
            library="fastapi",
            query="routing",
            documentation="FastAPI uses Starlette for routing...",
            sources=[source]
        )
        
        # Test the model
        self.assertEqual(response.library, "fastapi")
        self.assertEqual(response.query, "routing")
        self.assertEqual(response.documentation, "FastAPI uses Starlette for routing...")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].title, "Documentation")
        self.assertEqual(response.sources[0].url, "https://example.com")
        
        # Test serialization
        json_data = response.model_dump_json()
        parsed = json.loads(json_data)
        self.assertEqual(parsed["library"], "fastapi")
        self.assertEqual(parsed["sources"][0]["title"], "Documentation")

    def test_implementation_patterns_response_model(self):
        """Test the ImplementationPatternsResponse Pydantic model."""
        source = SourceReference(title="Design Patterns", url="https://example.com/patterns")
        response = ImplementationPatternsResponse(
            feature_description="user authentication",
            languages=["python"],
            frameworks=["fastapi"],
            implementation_patterns="Use JWT for authentication...",
            sources=[source]
        )
        
        # Test the model
        self.assertEqual(response.feature_description, "user authentication")
        self.assertEqual(response.languages, ["python"])
        self.assertEqual(response.frameworks, ["fastapi"])
        self.assertEqual(response.implementation_patterns, "Use JWT for authentication...")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].title, "Design Patterns")
        
        # Test serialization
        json_data = response.model_dump_json()
        parsed = json.loads(json_data)
        self.assertEqual(parsed["feature_description"], "user authentication")
        self.assertEqual(parsed["sources"][0]["url"], "https://example.com/patterns")


if __name__ == "__main__":
    unittest.main()
