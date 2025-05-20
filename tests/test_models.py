"""
Basic tests for the Gemini Update Agent models.
"""

import unittest
from datetime import datetime

from src.models import (
    # Analysis models
    CodebaseFile,
    CodeDependency, 
    FeatureScope,
    ChangeType,
    CodeChange,
    ImplementationPlan,
    CodebaseContext,
    
    # Feature models
    FeatureType,
    Priority,
    RequirementType,
    Requirement,
    FeatureSpec
)

class TestAnalysisModels(unittest.TestCase):
    """Tests for the analysis models."""
    
    def test_codefile_creation(self):
        """Test creating a CodebaseFile."""
        file = CodebaseFile(
            path="src/main.py",
            file_type="Python"
        )
        self.assertEqual(file.path, "src/main.py")
        self.assertEqual(file.file_type, "Python")
        
    def test_code_dependency_creation(self):
        """Test creating a CodeDependency."""
        dependency = CodeDependency(
            source="src/main.py",
            target="src/utils.py",
            dependency_type="import"
        )
        self.assertEqual(dependency.source, "src/main.py")
        self.assertEqual(dependency.target, "src/utils.py")
        self.assertEqual(dependency.dependency_type, "import")
        
    def test_feature_scope_creation(self):
        """Test creating a FeatureScope."""
        scope = FeatureScope(
            affected_files=["src/main.py", "src/utils.py"],
            new_files=["src/new_feature.py"],
            dependencies_needed=["numpy", "pandas"],
            config_changes=["Add numpy to requirements.txt"]
        )
        self.assertEqual(len(scope.affected_files), 2)
        self.assertEqual(len(scope.new_files), 1)
        self.assertEqual(len(scope.dependencies_needed), 2)
        self.assertEqual(len(scope.config_changes), 1)
        
    def test_code_change_creation(self):
        """Test creating a CodeChange."""
        change = CodeChange(
            file_path="src/main.py",
            change_type=ChangeType.MODIFY,
            description="Add new function",
            code_snippet="def new_function():\n    pass",
            line_range="10-20"
        )
        self.assertEqual(change.file_path, "src/main.py")
        self.assertEqual(change.change_type, ChangeType.MODIFY)
        self.assertEqual(change.description, "Add new function")
        
    def test_implementation_plan_creation(self):
        """Test creating an ImplementationPlan."""
        scope = FeatureScope(
            affected_files=["src/main.py"],
            new_files=["src/new_feature.py"]
        )
        
        change = CodeChange(
            file_path="src/main.py",
            change_type=ChangeType.MODIFY,
            description="Add new function",
            code_snippet="def new_function():\n    pass",
            line_range="10-20"
        )
        
        plan = ImplementationPlan(
            feature_name="New Feature",
            description="Add a new feature",
            scope=scope,
            changes=[change],
            estimated_complexity="Medium",
            dependencies=["Complete task X first"],
            generated_at=datetime.now()
        )
        
        self.assertEqual(plan.feature_name, "New Feature")
        self.assertEqual(plan.estimated_complexity, "Medium")
        self.assertEqual(len(plan.changes), 1)
        self.assertEqual(len(plan.dependencies), 1)
        
    def test_markdown_generation(self):
        """Test generating markdown from an ImplementationPlan."""
        scope = FeatureScope(
            affected_files=["src/main.py"],
            new_files=["src/new_feature.py"]
        )
        
        change = CodeChange(
            file_path="src/main.py",
            change_type=ChangeType.MODIFY,
            description="Add new function",
            code_snippet="def new_function():\n    pass",
            line_range="10-20"
        )
        
        plan = ImplementationPlan(
            feature_name="New Feature",
            description="Add a new feature",
            scope=scope,
            changes=[change],
            estimated_complexity="Medium",
            dependencies=["Complete task X first"],
            generated_at=datetime.now()
        )
        
        markdown = plan.to_markdown()
        self.assertIsInstance(markdown, str)
        self.assertIn("# Implementation Plan for New Feature", markdown)
        self.assertIn("**Estimated Complexity**: Medium", markdown)
        
class TestFeatureModels(unittest.TestCase):
    """Tests for the feature models."""
    
    def test_requirement_creation(self):
        """Test creating a Requirement."""
        requirement = Requirement(
            id="REQ-1",
            type=RequirementType.FUNCTIONAL,
            description="The system shall do X",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            dependencies=["REQ-0"]
        )
        self.assertEqual(requirement.id, "REQ-1")
        self.assertEqual(requirement.type, RequirementType.FUNCTIONAL)
        self.assertEqual(len(requirement.acceptance_criteria), 2)
        
    def test_feature_spec_creation(self):
        """Test creating a FeatureSpec."""
        requirement = Requirement(
            id="REQ-1",
            type=RequirementType.FUNCTIONAL,
            description="The system shall do X",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            dependencies=[]
        )
        
        feature_spec = FeatureSpec(
            name="New Feature",
            description="Add a new feature",
            feature_type=FeatureType.CORE,
            priority=Priority.HIGH,
            requirements=[requirement],
            user_personas=["Admin", "User"],
            success_metrics=["Metric 1", "Metric 2"],
            technical_notes="Some technical notes"
        )
        
        self.assertEqual(feature_spec.name, "New Feature")
        self.assertEqual(feature_spec.feature_type, FeatureType.CORE)
        self.assertEqual(feature_spec.priority, Priority.HIGH)
        self.assertEqual(len(feature_spec.requirements), 1)
        
    def test_feature_spec_markdown(self):
        """Test generating markdown from a FeatureSpec."""
        requirement = Requirement(
            id="REQ-1",
            type=RequirementType.FUNCTIONAL,
            description="The system shall do X",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            dependencies=[]
        )
        
        feature_spec = FeatureSpec(
            name="New Feature",
            description="Add a new feature",
            feature_type=FeatureType.CORE,
            priority=Priority.HIGH,
            requirements=[requirement],
            user_personas=["Admin", "User"],
            success_metrics=["Metric 1", "Metric 2"],
            technical_notes="Some technical notes"
        )
        
        markdown = feature_spec.to_markdown()
        self.assertIsInstance(markdown, str)
        self.assertIn("# Feature Specification: New Feature", markdown)
        self.assertIn("**Type:** core", markdown)
        self.assertIn("**Priority:** high", markdown)


if __name__ == "__main__":
    unittest.main()
