"""
Example implementation of a feature specification for demonstration purposes.
"""

from src.models import (
    FeatureType,
    Priority,
    RequirementType,
    Requirement,
    FeatureSpec
)

def create_example_feature_spec():
    """Create an example feature specification for demonstration."""
    
    # Create requirements
    requirements = [
        Requirement(
            id="REQ-1",
            type=RequirementType.FUNCTIONAL,
            description="Implement automatic detection of system dark mode preference",
            acceptance_criteria=[
                "Application detects system dark mode settings on initial load",
                "Application responds to system dark mode setting changes in real-time",
                "Default theme matches the system preference"
            ],
            dependencies=[]
        ),
        Requirement(
            id="REQ-2",
            type=RequirementType.FUNCTIONAL,
            description="Add manual toggle to switch between light and dark modes",
            acceptance_criteria=[
                "Toggle is accessible from the main navigation area",
                "Toggle state persists between sessions via local storage",
                "Manual selection overrides system preference"
            ],
            dependencies=["REQ-1"]
        ),
        Requirement(
            id="REQ-3",
            type=RequirementType.TECHNICAL,
            description="Create a consistent dark theme across all UI components",
            acceptance_criteria=[
                "All components have appropriate dark mode styles",
                "Maintain appropriate contrast ratios for accessibility",
                "Support smooth transition animations between modes"
            ],
            dependencies=[]
        )
    ]
    
    # Create feature specification
    feature_spec = FeatureSpec(
        name="Dark Mode Support",
        description="Add dark mode support to the UI components with automatic detection of system preferences and a manual toggle.",
        feature_type=FeatureType.UI,
        priority=Priority.MEDIUM,
        requirements=requirements,
        user_personas=[
            "Regular users who prefer dark mode for eye comfort",
            "Users with visual sensitivity who require dark mode",
            "Users who switch between light and dark environments"
        ],
        success_metrics=[
            "Increased session duration for users with dark mode enabled",
            "Reduced number of reported eye strain issues",
            "Positive feedback on accessibility features"
        ],
        technical_notes="Ensure compatibility with all major browsers and responsive behavior on various screen sizes."
    )
    
    return feature_spec
