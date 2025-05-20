"""
Example feature prompt template.

This template helps generate comprehensive feature specifications
for the Gemini Update Agent.
"""

FEATURE_PROMPT_TEMPLATE = """
# Feature Request Analysis

## Feature Description
{feature_description}

## Project Context
The project is a {project_type} developed using {primary_language}. 
{additional_context}

## Constraints
- {constraints}

## Expected Output
Please analyze this feature request and provide:

1. A detailed feature specification with:
   - Feature name and description
   - Feature type and priority
   - Detailed requirements with acceptance criteria
   - User personas affected
   - Success metrics
   - Technical notes or considerations

2. An implementation plan including:
   - Files that need to be modified
   - New files that need to be created
   - Specific code changes with descriptions and snippets
   - Dependencies between changes
   - Estimated complexity

Format your response as structured JSON objects matching the FeatureSpec and ImplementationPlan models.
"""

def generate_feature_prompt(
    feature_description: str,
    project_type: str,
    primary_language: str,
    additional_context: str = "",
    constraints: str = "Must maintain backward compatibility"
) -> str:
    """
    Generate a feature prompt using the template.
    
    Args:
        feature_description: Description of the feature
        project_type: Type of project (web app, CLI tool, etc.)
        primary_language: Primary programming language
        additional_context: Additional context about the project
        constraints: Constraints for the feature implementation
        
    Returns:
        Formatted feature prompt string
    """
    return FEATURE_PROMPT_TEMPLATE.format(
        feature_description=feature_description,
        project_type=project_type,
        primary_language=primary_language,
        additional_context=additional_context,
        constraints=constraints
    )
