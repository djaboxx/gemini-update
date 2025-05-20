"""
Pydantic-AI Tools for feature specification and implementation planning.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from functools import wraps
import json

from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai import ModelRetry

from src.models import (
    CodebaseContext,
    FeatureSpec,
    Requirement,
    RequirementType,
    FeatureType,
    Priority,
    ImplementationPlan,
    CodeChange,
    ChangeType,
    FeatureScope
)


logger = logging.getLogger("gemini_update")


def register_feature_tools(agent: Agent[CodebaseContext, str], max_retries: int = 1) -> None:
    """Register feature-related tools with the agent."""
    
    def retry_on_error(func):
        """Decorator to add retry logic to tools."""
        @wraps(func)  # Preserve the original function's metadata
        async def wrapped_tool(*args, **kwargs):
            retries = 0
            last_error = None
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    retries += 1
                    if retries <= max_retries:
                        logger.warning(f"Tool {func.__name__} failed, attempt {retries}/{max_retries}: {str(e)}")
                        continue
                    raise ModelRetry(f"Error in {func.__name__}: {str(e)}")
            raise last_error
        return wrapped_tool

    @retry_on_error
    @agent.tool
    async def create_feature_spec(
        ctx: RunContext[CodebaseContext],
        name: str,
        description: str,
        feature_type: str,
        priority: str,
        requirements: List[Dict[str, Any]]
    ) -> str:
        """
        Create a structured feature specification.

        Args:
            ctx: The run context
            name: Feature name
            description: Feature description
            feature_type: Type of feature (core, ui, integration, security, performance, other)
            priority: Priority level (low, medium, high, critical)
            requirements: List of requirements for this feature

        Returns:
            JSON string representing the FeatureSpec
        """
        try:
            # Validate feature type
            try:
                feature_type_enum = FeatureType(feature_type.lower())
            except ValueError:
                raise ModelRetry(f"Invalid feature type: {feature_type}. Must be one of: {', '.join([t.value for t in FeatureType])}")
                
            # Validate priority
            try:
                priority_enum = Priority(priority.lower())
            except ValueError:
                raise ModelRetry(f"Invalid priority: {priority}. Must be one of: {', '.join([p.value for p in Priority])}")
                
            # Validate and create requirements
            validated_requirements = []
            for i, req in enumerate(requirements):
                # Generate an ID if not provided
                req_id = req.get("id", f"REQ-{i+1}")
                
                # Validate requirement type
                req_type = req.get("type", "functional")
                try:
                    req_type_enum = RequirementType(req_type.lower())
                except ValueError:
                    raise ModelRetry(f"Invalid requirement type: {req_type}. Must be one of: {', '.join([t.value for t in RequirementType])}")
                    
                # Create Requirement object
                validated_requirements.append(
                    Requirement(
                        id=req_id,
                        type=req_type_enum,
                        description=req.get("description", ""),
                        acceptance_criteria=req.get("acceptance_criteria", []),
                        dependencies=req.get("dependencies", [])
                    )
                )
                
            # Create FeatureSpec
            feature_spec = FeatureSpec(
                name=name,
                description=description,
                feature_type=feature_type_enum,
                priority=priority_enum,
                requirements=validated_requirements,
                user_personas=ctx.args.get("user_personas", []),
                success_metrics=ctx.args.get("success_metrics", []),
                technical_notes=ctx.args.get("technical_notes")
            )
            
            # Return as JSON
            return feature_spec.model_dump_json(indent=2)
            
        except Exception as e:
            raise ModelRetry(f"Error creating feature specification: {str(e)}")
            
    @retry_on_error
    @agent.tool
    async def create_implementation_plan(
        ctx: RunContext[CodebaseContext],
        feature_name: str,
        description: str,
        affected_files: List[str],
        new_files: List[str],
        changes: List[Dict[str, Any]],
        estimated_complexity: str
    ) -> str:
        """
        Create a structured implementation plan for a feature.

        Args:
            ctx: The run context
            feature_name: Name of the feature
            description: Description of what the feature does
            affected_files: List of existing files that need to be modified
            new_files: List of new files that need to be created
            changes: List of specific code changes
            estimated_complexity: Estimated complexity (Low, Medium, High)

        Returns:
            JSON string representing the ImplementationPlan
        """
        try:
            # Create feature scope
            scope = FeatureScope(
                affected_files=affected_files,
                new_files=new_files,
                dependencies_needed=ctx.args.get("dependencies_needed", []),
                config_changes=ctx.args.get("config_changes", [])
            )
            
            # Validate and create code changes
            validated_changes = []
            for change_data in changes:
                # Validate change type
                change_type_str = change_data.get("change_type", "modify")
                try:
                    change_type = ChangeType(change_type_str.lower())
                except ValueError:
                    raise ModelRetry(f"Invalid change type: {change_type_str}. Must be one of: {', '.join([t.value for t in ChangeType])}")
                    
                # Create CodeChange object
                validated_changes.append(
                    CodeChange(
                        file_path=change_data.get("file_path", ""),
                        change_type=change_type,
                        description=change_data.get("description", ""),
                        code_snippet=change_data.get("code_snippet"),
                        line_range=change_data.get("line_range")
                    )
                )
                
            # Create ImplementationPlan
            implementation_plan = ImplementationPlan(
                feature_name=feature_name,
                description=description,
                scope=scope,
                changes=validated_changes,
                estimated_complexity=estimated_complexity,
                dependencies=ctx.args.get("dependencies", [])
            )
            
            # Return as JSON
            return implementation_plan.model_dump_json(indent=2)
            
        except Exception as e:
            raise ModelRetry(f"Error creating implementation plan: {str(e)}")
            
    @retry_on_error
    @agent.tool
    async def identify_affected_files(
        ctx: RunContext[CodebaseContext],
        feature_description: str,
        file_patterns: Optional[List[str]] = None
    ) -> List[str]:
        """
        Identify files that might be affected by a new feature.

        Args:
            ctx: The run context containing project filesystem access
            feature_description: Description of the feature to implement
            file_patterns: Optional list of glob patterns to restrict the search

        Returns:
            List of file paths that might need modification
        """
        try:
            project_dir = ctx.deps.project_dir
            result = set()
            file_access = ctx.deps.file_access
            
            # Extract keywords from feature description
            keywords = []
            # Extract important nouns, verbs, and technical terms from the feature description
            words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]+\b', feature_description.lower())
            # Filter out common stopwords
            stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                        'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                        'some', 'such', 'no', 'nor', 'too', 'very', 'can', 'will', 'just', 'should',
                        'now', 'to', 'of', 'for', 'in', 'on', 'by', 'about', 'with', 'that', 'this'}
            keywords = [word for word in words if word not in stopwords and len(word) > 2]
            
            # 1. First check for main configuration and entry point files
            config_files = []
            for config_file in [
                "package.json", "pyproject.toml", "setup.py", "requirements.txt",
                "app.py", "main.py", "index.js", "server.js", "config.py", "settings.py",
                "urls.py", "routes.js", "app.js", "main.js", "webpack.config.js",
                "tsconfig.json", "tox.ini", ".env.example", "docker-compose.yml",
                "Dockerfile", "Makefile"
            ]:
                if file_access and file_access.file_exists(config_file):
                    config_files.append(config_file)
                elif (project_dir / config_file).exists():
                    config_files.append(config_file)
            
            # 2. Use the codebase tools to search for code related to the feature keywords
            for keyword in keywords[:5]:  # Limit to top 5 keywords to avoid too many searches
                try:
                    # Search for the keyword in code files
                    search_results = await ctx.execute(
                        "search_code",
                        query=keyword,
                        file_patterns=file_patterns,
                        case_sensitive=False
                    )
                    
                    # Add matched files to results
                    for match in search_results:
                        result.add(match["file"])
                except Exception as search_error:
                    logger.warning(f"Error searching for keyword '{keyword}': {str(search_error)}")
            
            # 3. Analyze project structure to find relevant directories and files
            try:
                # Get project structure
                project_info = await ctx.execute("get_project_info")
                
                # Identify main application directories
                common_dirs = ["src", "app", "lib", "modules", "components", "controllers", "models", "views"]
                for common_dir in common_dirs:
                    is_dir = False
                    
                    # Check if directory exists using file access layer if available
                    if file_access and file_access.is_directory(common_dir):
                        is_dir = True
                        dir_items = file_access.list_directory(common_dir)
                    elif (project_dir / common_dir).is_dir():
                        is_dir = True
                        dir_items = os.listdir(project_dir / common_dir)
                    
                    if is_dir:
                        # Add specific subdirectories based on feature keywords
                        for keyword in keywords:
                            # Check for directories that might match the feature
                            for item in dir_items:
                                item_path = f"{common_dir}/{item}"
                                
                                # Check if item is directory or file
                                is_item_dir = False
                                if file_access:
                                    is_item_dir = file_access.is_directory(item_path)
                                else:
                                    is_item_dir = (project_dir / common_dir / item).is_dir()
                                
                                if is_item_dir and (keyword in item.lower()):
                                    result.add(item_path)
                                elif not is_item_dir and (keyword in item.lower()):
                                    result.add(item_path)
            except Exception as struct_error:
                logger.warning(f"Error analyzing project structure: {str(struct_error)}")
            
            # 4. Analyze imports to find related files for the matched files
            analyzed_files = set()
            files_to_analyze = list(result)
            dependency_depth = 0
            max_dependency_depth = 2  # Limit dependency traversal depth
            
            while files_to_analyze and dependency_depth < max_dependency_depth:
                next_files = []
                
                for file_path in files_to_analyze:
                    if file_path in analyzed_files or not file_path.endswith(('.py', '.js', '.ts')):
                        continue
                    
                    analyzed_files.add(file_path)
                    
                    try:
                        # Only analyze imports for Python files for now
                        if file_path.endswith('.py'):
                            imports = await ctx.execute("analyze_imports", file_path=file_path)
                            
                            # Process direct imports
                            for imp in imports.get("imports", []):
                                imp_name = imp["name"]
                                # Look for corresponding file
                                possible_files = [
                                    f"{imp_name.replace('.', '/')}.py",
                                    f"{'/'.join(imp_name.split('.'))}.py"
                                ]
                                for pf in possible_files:
                                    # Check if file exists using file access layer if available
                                    file_exists = False
                                    if file_access:
                                        file_exists = file_access.file_exists(pf)
                                    else:
                                        file_exists = (project_dir / pf).exists()
                                        
                                    if file_exists:
                                        result.add(pf)
                                        next_files.append(pf)
                            
                            # Process from imports
                            for imp in imports.get("from_imports", []):
                                module = imp["module"]
                                if module:
                                    # Add the module file and handle package imports
                                    module_file = f"{module.replace('.', '/')}.py"
                                    module_init = f"{module.replace('.', '/')}/__init__.py"
                                    
                                    # Check module file
                                    module_file_exists = False
                                    if file_access:
                                        module_file_exists = file_access.file_exists(module_file)
                                    else:
                                        module_file_exists = (project_dir / module_file).exists()
                                    
                                    # Check init file
                                    module_init_exists = False
                                    if file_access:
                                        module_init_exists = file_access.file_exists(module_init)
                                    else:
                                        module_init_exists = (project_dir / module_init).exists()
                                    
                                    if module_file_exists:
                                        result.add(module_file)
                                        next_files.append(module_file)
                                    elif module_init_exists:
                                        result.add(module_init)
                                        next_files.append(module_init)
                    except Exception as imp_error:
                        logger.warning(f"Error analyzing imports for {file_path}: {str(imp_error)}")
                
                files_to_analyze = next_files
                dependency_depth += 1
            
            # 5. Add configuration files that might need updating for any new feature
            for config_file in config_files:
                result.add(config_file)
            
            # Convert to list and ensure paths are normalized
            result_list = sorted(list(result))
            
            # Handle the case where no files were found - return some sensible defaults
            if not result_list:
                # Look for common entry point files
                for entry_file in ["app.py", "main.py", "index.js", "server.js"]:
                    file_exists = False
                    if file_access:
                        file_exists = file_access.file_exists(entry_file)
                    else:
                        file_exists = (project_dir / entry_file).exists()
                        
                    if file_exists:
                        result_list.append(entry_file)
                
                # Include src directory if it exists
                src_is_dir = False
                if file_access:
                    src_is_dir = file_access.is_directory("src")
                else:
                    src_is_dir = (project_dir / "src").is_dir()
                    
                if src_is_dir:
                    result_list.append("src")
            
            return result_list
            
        except Exception as e:
            logger.error(f"Error identifying affected files: {str(e)}", exc_info=True)
            raise ModelRetry(f"Error identifying affected files: {str(e)}")
            
    @retry_on_error
    @agent.tool
    async def save_feature_spec(
        ctx: RunContext[CodebaseContext],
        feature_spec_json: str,
        output_path: str
    ) -> str:
        """
        Save a feature specification to a markdown file.

        Args:
            ctx: The run context containing project filesystem access
            feature_spec_json: JSON string representing a FeatureSpec
            output_path: Path to save the markdown file

        Returns:
            Path to the saved file
        """
        try:
            # Parse the feature spec JSON
            try:
                feature_spec = FeatureSpec.model_validate_json(feature_spec_json)
            except Exception as e:
                raise ModelRetry(f"Error parsing feature spec JSON: {str(e)}")
                
            # Convert to markdown
            markdown = feature_spec.to_markdown()
            
            # Save to file
            output_file_path = ctx.deps.validate_file_path(output_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(markdown)
                
            return str(output_file_path)
            
        except Exception as e:
            raise ModelRetry(f"Error saving feature spec: {str(e)}")
            
    @retry_on_error
    @agent.tool
    async def save_implementation_plan(
        ctx: RunContext[CodebaseContext],
        implementation_plan_json: str,
        output_path: str
    ) -> str:
        """
        Save an implementation plan to a markdown file.

        Args:
            ctx: The run context containing project filesystem access
            implementation_plan_json: JSON string representing an ImplementationPlan
            output_path: Path to save the markdown file

        Returns:
            Path to the saved file
        """
        try:
            # Parse the implementation plan JSON
            try:
                implementation_plan = ImplementationPlan.model_validate_json(implementation_plan_json)
            except Exception as e:
                raise ModelRetry(f"Error parsing implementation plan JSON: {str(e)}")
                
            # Convert to markdown
            markdown = implementation_plan.to_markdown()
            
            # Save to file
            output_file_path = ctx.deps.validate_file_path(output_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(markdown)
                
            return str(output_file_path)
            
        except Exception as e:
            raise ModelRetry(f"Error saving implementation plan: {str(e)}")
            
    @retry_on_error
    @agent.tool
    async def extract_feature_concepts(
        ctx: RunContext[CodebaseContext],
        feature_description: str
    ) -> Dict[str, Any]:
        """
        Extract key concepts and technical components from a feature description.

        Args:
            ctx: The run context containing project filesystem access
            feature_description: Description of the feature to implement

        Returns:
            Dictionary with extracted feature concepts, patterns, and categories
        """
        try:
            # Initialize result structure
            result = {
                "main_concepts": [],
                "technical_components": [],
                "ui_elements": [],
                "data_elements": [],
                "api_endpoints": [],
                "patterns": []
            }
            
            # Extract key technical components
            # These are common technical terms that might indicate specific implementation areas
            tech_components = re.findall(r'\b(api|database|authentication|authorization|endpoint|component|service|'
                                        r'controller|model|view|handler|middleware|hook|event|listener|router|'
                                        r'reducer|store|context|provider|validator|parser|formatter|renderer|'
                                        r'serializer|deserializer|transformer|converter|adapter|factory|builder|'
                                        r'manager|coordinator|director|proxy|facade|decorator|observer|'
                                        r'strategy|template|command|interpreter|visitor|mediator|memento|'
                                        r'iterator|composite|bridge|flyweight|singleton|prototype)\b',
                                        feature_description.lower())
            result["technical_components"] = list(set(tech_components))
            
            # Extract UI elements
            ui_elements = re.findall(r'\b(button|form|input|select|dropdown|menu|modal|dialog|popup|tooltip|'
                                    r'alert|notification|tab|panel|card|grid|table|list|sidebar|navbar|footer|'
                                    r'header|layout|container|section|divider|badge|label|image|icon|'
                                    r'avatar|carousel|slider|progress|spinner|loader|toggle|switch|'
                                    r'checkbox|radio|textbox|textarea|datepicker|timepicker|'
                                    r'autocomplete|combobox|multiselect|tree|treeview|stepper|'
                                    r'accordion|collapse|expansion|drawer|window|frame|canvas)\b',
                                    feature_description.lower())
            result["ui_elements"] = list(set(ui_elements))
            
            # Extract data elements
            data_elements = re.findall(r'\b(user|profile|account|session|token|credential|permission|role|'
                                    r'config|setting|preference|option|data|record|entity|object|'
                                    r'document|file|image|media|audio|video|text|content|message|'
                                    r'notification|event|log|error|exception|warning|info|debug|'
                                    r'trace|status|state|flag|indicator|counter|timer|date|time|'
                                    r'timestamp|duration|interval|period|range|limit|threshold|'
                                    r'constraint|rule|policy|contract|agreement|license|term|'
                                    r'condition|requirement|specification)\b',
                                    feature_description.lower())
            result["data_elements"] = list(set(data_elements))
            
            # Extract possible API endpoints
            api_endpoints = re.findall(r'\b(get|post|put|patch|delete|options|head|connect|trace)\s+[/\w-]+',
                                    feature_description.lower())
            result["api_endpoints"] = list(set(api_endpoints))
            
            # Extract main concepts (significant nouns and technical terms)
            # Remove stop words and common words
            stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                        'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                        'some', 'such', 'no', 'nor', 'too', 'very', 'can', 'will', 'just', 'should',
                        'now', 'to', 'of', 'for', 'in', 'on', 'by', 'about', 'with', 'feature', 'implement'}
            
            # Extract nouns and compound terms
            words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*(?:\s+[a-zA-Z][a-zA-Z0-9_]*){0,2}\b', feature_description.lower())
            filtered_concepts = []
            for word in words:
                if word not in stopwords and len(word) > 2:
                    filtered_concepts.append(word)
            
            # Keep only the most significant concepts
            result["main_concepts"] = filtered_concepts[:10] if len(filtered_concepts) > 10 else filtered_concepts
            
            # Detect common implementation patterns - use search instead of hard-coded patterns
            patterns = []
            try:
                # Try using the search_implementation_patterns tool if we have grounded search capability
                if ctx.deps.settings.use_gemini_files:
                    logger.info("Using grounded search to identify implementation patterns...")
                    # Get the primary language and frameworks from the codebase context
                    languages = []
                    if ctx.deps.primary_language:
                        languages.append(ctx.deps.primary_language.lower())
                    
                    frameworks = []
                    if ctx.deps.frameworks:
                        frameworks.extend([f.lower() for f in ctx.deps.frameworks])
                    
                    # Execute the search_implementation_patterns tool
                    search_result = await ctx.execute(
                        "search_implementation_patterns",
                        feature_description=feature_description,
                        languages=languages if languages else None,
                        frameworks=frameworks if frameworks else None
                    )
                    
                    # Extract patterns from the search result - look for pattern names in bullet points, section titles, etc.
                    if search_result and "implementation_patterns" in search_result:
                        # Try to extract pattern names from headings and bullet points
                        text = search_result["implementation_patterns"].lower()
                        lines = text.split("\n")
                        for line in lines:
                            line = line.strip()
                            if line.startswith("#") or line.startswith("-") or line.startswith("*"):
                                # Extract potential pattern name (first 3-5 words)
                                words = line.split()
                                pattern_name_words = [w for w in words[:5] if w not in ["#", "-", "*", "1.", "2.", "3.", "4.", "5."]]
                                if pattern_name_words:
                                    pattern_name = " ".join(pattern_name_words)
                                    patterns.append(pattern_name)
                        
                        # If we couldn't extract patterns, use key phrases as patterns
                        if not patterns:
                            key_phrases = ["design pattern", "architecture", "best practice", "approach"]
                            for phrase in key_phrases:
                                if phrase in text:
                                    idx = text.index(phrase)
                                    context = text[max(0, idx-30):min(len(text), idx+30)]
                                    patterns.append(context)
                    
                        # Clean up and limit patterns
                        patterns = [p[:50] for p in patterns]
                        patterns = patterns[:10]  # Limit to 10 patterns
                
            except Exception as e:
                logger.warning(f"Error using grounded search for patterns: {str(e)}. Falling back to basic analysis.")
                
            # Fall back to basic pattern detection if needed
            if not patterns:
                logger.info("Using basic pattern detection...")
                # Use simple word matching for basic pattern detection
                pattern_words = {
                    "CRUD operations": ["create", "read", "update", "delete", "list", "crud"],
                    "Authentication": ["login", "logout", "sign", "register", "auth", "password"],
                    "Data validation": ["validate", "sanitize", "verify", "check"],
                    "State management": ["state", "store", "reducer", "context", "provider"],
                    "Async operations": ["async", "await", "promise", "callback", "future"],
                    "API integration": ["api", "endpoint", "service", "client", "request", "response"]
                }
                
                for pattern_name, keywords in pattern_words.items():
                    if any(kw in feature_description.lower() for kw in keywords):
                        patterns.append(pattern_name)