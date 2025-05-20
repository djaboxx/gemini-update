"""
Prompts for implementation planner agent.
"""

from jinja2 import Template
from typing import Optional
from src.models import FeatureSpec

# System prompt template for ImplementationPlannerAgent
IMPLEMENTATION_PLANNER_SYSTEM_PROMPT_TEMPLATE = """
You are a specialized implementation planning agent. Your role is to:
1. Analyze feature specifications in the context of the codebase
2. Plan detailed implementation steps
3. Identify files requiring changes
4. Generate specific code changes
5. Define dependencies between changes

IMPORTANT: Your implementation plan MUST use the actual file paths and structure from the analyzed codebase. Do not assume web frameworks or frontend/backend structure unless they actually exist in the codebase.

Your outputs must be structured ImplementationPlan objects with:
- Clear scope of changes (files, dependencies)
- Detailed code changes with snippets
- Dependencies between changes
- Complexity estimates
- Implementation steps in the correct order

{% if custom_instructions %}
Additional instructions:
{{ custom_instructions }}
{% endif %}
"""

# Prompt template for implementation planning
IMPLEMENTATION_PLANNER_PROMPT_TEMPLATE = """
Based on the following feature specification and the analyzed codebase,
create a detailed implementation plan:

{{ feature_spec_markdown }}

CRITICAL: Ensure ALL file paths match the actual project structure. Do not invent fictional paths like 'frontend/' or 'backend/' unless they exist in the codebase. Use the correct project layout from the codebase analysis.

Include:
- Files that need to be modified
- New files that need to be created
- Specific code changes with descriptions and snippets
- Dependencies between changes
- Estimated complexity

Respond with a valid ImplementationPlan object.
{% if custom_instructions %}
Additional instructions:
{{ custom_instructions }}
{% endif %}
"""

def get_implementation_planner_system_prompt(custom_instructions: Optional[str] = None) -> str:
    """Get system prompt for implementation planner."""
    template = Template(IMPLEMENTATION_PLANNER_SYSTEM_PROMPT_TEMPLATE)
    return template.render(
        custom_instructions=custom_instructions if custom_instructions else ""
    )

def get_implementation_planner_prompt(feature_spec: FeatureSpec, custom_instructions: Optional[str] = None) -> str:
    """Get prompt for implementation planning."""
    template = Template(IMPLEMENTATION_PLANNER_PROMPT_TEMPLATE)
    return template.render(
        feature_spec_markdown=feature_spec.to_markdown(),
        custom_instructions=custom_instructions if custom_instructions else ""
    )
