"""
Prompts for feature specification generator agent.
"""

from jinja2 import Template
from typing import Optional

# System prompt template for FeatureSpecGeneratorAgent
FEATURE_SPEC_SYSTEM_PROMPT_TEMPLATE = """
You are a specialized feature specification agent. Your role is to:
1. Analyze feature descriptions and codebase context
2. Generate detailed, actionable feature specifications
3. Define clear requirements and acceptance criteria
4. Identify affected user personas
5. Specify success metrics

Your outputs must be structured FeatureSpec objects with:
- Feature name and clear description
- Feature type and priority level
- Detailed requirements list
- Acceptance criteria for each requirement
- Relevant user personas
- Measurable success metrics
- Technical notes considering the codebase context
"""

# Prompt template for feature specification generation
FEATURE_SPEC_PROMPT_TEMPLATE = """
Based on the following feature description and the analyzed codebase,
generate a detailed feature specification with requirements:

Feature: {{ feature_description }}

Include:
- Feature name and description
- Feature type and priority
- Detailed requirements with acceptance criteria
- User personas affected
- Success metrics
- Technical notes or considerations

Respond with a valid FeatureSpec object.

{% if custom_instructions %}
Additional instructions:
{{ custom_instructions }}
{% endif %}
"""

def get_feature_spec_system_prompt(custom_instructions: Optional[str] = None) -> str:
    """Get system prompt for feature spec generator."""
    template = Template(FEATURE_SPEC_SYSTEM_PROMPT_TEMPLATE)
    return template.render(
        custom_instructions=custom_instructions if custom_instructions else ""
    )

def get_feature_spec_prompt(feature_description: str, custom_instructions: Optional[str] = None) -> str:
    """Get prompt for feature specification generation."""
    template = Template(FEATURE_SPEC_PROMPT_TEMPLATE)
    return template.render(
        feature_description=feature_description,
        custom_instructions=custom_instructions if custom_instructions else ""
    )
