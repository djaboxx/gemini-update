# Implementation Plan for GitHub Integration Feature

*Generated at: 2025-05-19 10:30:00*

## Description

Add GitHub integration to the gemini-update agent to allow creating issues directly from implementation plans. This will streamline the workflow by enabling users to convert their implementation plans into trackable GitHub issues with a single command.

**Estimated Complexity**: Medium

## Scope

### Files to Modify
- `src/agent/agent.py`
- `gemini-update.py`
- `src/tools/feature_tools.py`
- `src/models/analysis.py`

### New Files to Create
- `src/tools/github_tools.py`
- `src/models/github.py`

## Implementation Steps

### Step 1: Create GitHub integration models

**File:** `src/models/github.py`

**Description:**
Create Pydantic models for GitHub API integration.

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class GitHubAuth(BaseModel):
    """GitHub authentication configuration."""
    token: str = Field(..., description="GitHub personal access token")
    
class GitHubRepository(BaseModel):
    """GitHub repository information."""
    owner: str = Field(..., description="Repository owner (username or organization)")
    name: str = Field(..., description="Repository name")
    
class GitHubIssueOptions(BaseModel):
    """Options for creating GitHub issues."""
    create_task_issues: bool = Field(False, description="Whether to create separate task issues")
    labels: List[str] = Field(default_factory=lambda: ["enhancement", "gemini-update"], 
                            description="Labels to apply to the issue")
```

### Step 2: Implement GitHub integration tools

**File:** `src/tools/github_tools.py`

**Description:**
Create functions for interacting with the GitHub API to create issues from implementation plans.

```python
import logging
import requests
from typing import Optional, Dict, Any, List

from src.models.github import GitHubAuth, GitHubRepository, GitHubIssueOptions
from src.models.analysis import ImplementationPlan

logger = logging.getLogger("gemini_update")

def create_github_issue(
    auth: GitHubAuth,
    repo: GitHubRepository, 
    title: str, 
    body: str, 
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a GitHub issue.
    
    Args:
        auth: GitHub authentication information
        repo: Repository information
        title: Issue title
        body: Issue body in markdown format
        labels: Optional list of labels to apply to the issue
    
    Returns:
        Response data from GitHub API
    """
    url = f"https://api.github.com/repos/{repo.owner}/{repo.name}/issues"
    
    headers = {
        "Authorization": f"token {auth.token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "title": title,
        "body": body
    }
    
    if labels:
        data["labels"] = labels
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating GitHub issue: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        raise
```

### Step 3: Add a tool for creating GitHub issues from implementation plans

**File:** `src/tools/feature_tools.py`

**Description:**
Add a tool that uses the GitHub API to create issues from implementation plans.

```python
@retry_on_error
@agent.tool
async def create_github_issues_from_plan(
    ctx: RunContext[CodebaseContext],
    implementation_plan: ImplementationPlan,
    repo_owner: str,
    repo_name: str,
    github_token: str,
    create_task_issues: bool = False,
    labels: Optional[List[str]] = None
) -> str:
    """
    Create GitHub issues from an implementation plan.
    
    Args:
        ctx: The run context
        implementation_plan: The implementation plan to convert to GitHub issues
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        github_token: GitHub personal access token
        create_task_issues: Whether to create separate issues for each task
        labels: Labels to apply to the issues
        
    Returns:
        URL of the created main issue
    """
    try:
        from src.tools.github_tools import create_github_issue
        from src.models.github import GitHubAuth, GitHubRepository, GitHubIssueOptions
        
        # Create GitHub models
        auth = GitHubAuth(token=github_token)
        repo = GitHubRepository(owner=repo_owner, name=repo_name)
        options = GitHubIssueOptions(
            create_task_issues=create_task_issues,
            labels=labels or ["enhancement", "gemini-update"]
        )
        
        # Prepare the main issue title and body
        main_issue_title = f"Implement: {implementation_plan.feature_name}"
        
        # Create the main issue body using implementation_plan.to_markdown()
        main_issue_body = implementation_plan.to_markdown()
        main_issue_body += "\n---\n"
        main_issue_body += "_This issue was automatically generated by the gemini-update agent._\n"
        
        # Create the main GitHub issue
        main_issue = create_github_issue(
            auth,
            repo,
            main_issue_title,
            main_issue_body,
            options.labels
        )
        
        # Create task issues if requested
        if options.create_task_issues and implementation_plan.changes:
            for i, change in enumerate(implementation_plan.changes, 1):
                task_title = f"Task {i}: {change.description}"
                
                # Create task issue body
                task_body = f"# {change.description}\n\n"
                
                if change.file_path:
                    task_body += f"**File**: `{change.file_path}`\n\n"
                
                if change.change_type:
                    task_body += f"**Change Type**: {change.change_type.value}\n\n"
                
                if change.code_snippet:
                    task_body += "**Code**:\n\n```\n"
                    task_body += f"{change.code_snippet}\n"
                    task_body += "```\n\n"
                
                if change.line_range:
                    task_body += f"**Location**: Lines {change.line_range}\n\n"
                
                # Add reference to main issue
                task_body += f"\n\nPart of #{main_issue['number']} ({implementation_plan.feature_name})\n\n"
                task_body += "\n---\n"
                task_body += "_This task issue was automatically generated by the gemini-update agent._\n"
                
                # Create the task issue
                create_github_issue(
                    auth,
                    repo,
                    task_title,
                    task_body,
                    options.labels + ["task"]
                )
        
        return main_issue["html_url"]
        
    except Exception as e:
        raise ModelRetry(f"Error creating GitHub issues: {str(e)}")
```

### Step 4: Update the implementation plan model to support GitHub integration

**File:** `src/models/analysis.py`

**Description:**
Add a method to the ImplementationPlan model to create GitHub issues directly.

```python
def create_github_issues(
    self,
    repo_owner: str,
    repo_name: str,
    github_token: str,
    create_task_issues: bool = False,
    labels: Optional[List[str]] = None
) -> str:
    """
    Create GitHub issues from this implementation plan.
    
    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        github_token: GitHub personal access token
        create_task_issues: Whether to create separate issues for each task
        labels: Labels to apply to the issues
        
    Returns:
        URL of the created main issue
    """
    from src.tools.github_tools import create_github_issue
    from src.models.github import GitHubAuth, GitHubRepository, GitHubIssueOptions
    
    # Create GitHub models
    auth = GitHubAuth(token=github_token)
    repo = GitHubRepository(owner=repo_owner, name=repo_name)
    options = GitHubIssueOptions(
        create_task_issues=create_task_issues,
        labels=labels or ["enhancement", "gemini-update"]
    )
    
    # Prepare the main issue title and body
    main_issue_title = f"Implement: {self.feature_name}"
    
    # Create the main issue body
    main_issue_body = self.to_markdown()
    main_issue_body += "\n---\n"
    main_issue_body += "_This issue was automatically generated by the gemini-update agent._\n"
    
    # Create the main GitHub issue
    main_issue = create_github_issue(
        auth,
        repo,
        main_issue_title,
        main_issue_body,
        options.labels
    )
    
    # Create task issues if requested
    if options.create_task_issues and self.changes:
        for i, change in enumerate(self.changes, 1):
            # Create task issues for each change
            # [Implementation details omitted for brevity]
            pass
    
    return main_issue["html_url"]
```

### Step 5: Add GitHub integration command to the CLI

**File:** `gemini-update.py`

**Description:**
Add a new command to the CLI to allow creating GitHub issues from an existing implementation plan.

```python
@cli.command()
@click.option(
    "--implementation-plan-path",
    required=True,
    help="Path to an existing implementation plan markdown file",
)
@click.option(
    "--repo-owner",
    required=True,
    help="GitHub repository owner (username or organization)",
)
@click.option(
    "--repo-name",
    required=True,
    help="GitHub repository name",
)
@click.option(
    "--github-token",
    envvar="GITHUB_TOKEN",
    help="GitHub personal access token (defaults to GITHUB_TOKEN environment variable)",
)
@click.option(
    "--create-task-issues",
    is_flag=True,
    help="Create separate issues for each task in the implementation plan",
)
@click.option(
    "--labels",
    multiple=True,
    default=["enhancement", "gemini-update"],
    help="Labels to apply to the GitHub issues (can be specified multiple times)",
)
def github(
    implementation_plan_path: str,
    repo_owner: str,
    repo_name: str,
    github_token: str,
    create_task_issues: bool,
    labels: List[str],
) -> None:
    """Create GitHub issues from an implementation plan."""
    from examples.github_integration import main as github_integration_main
    
    # Set environment variables for the GitHub integration script
    os.environ["GITHUB_TOKEN"] = github_token
    
    # Build arguments for the GitHub integration script
    sys.argv = [
        "github_integration.py",
        f"--implementation-plan-path={implementation_plan_path}",
        f"--repo-owner={repo_owner}",
        f"--repo-name={repo_name}",
    ]
    
    if github_token:
        sys.argv.append(f"--github-token={github_token}")
        
    if create_task_issues:
        sys.argv.append("--create-task-issues")
        
    for label in labels:
        sys.argv.append(f"--labels={label}")
    
    # Run the GitHub integration script
    github_integration_main()
```

### Step 6: Update agent to support GitHub integration

**File:** `src/agent/agent.py`

**Description:**
Update the GeminiUpdateAgent class to support GitHub integration as part of the feature update workflow.

```python
async def create_github_issues(
    self,
    implementation_plan: ImplementationPlan,
    repo_owner: str,
    repo_name: str,
    github_token: str,
    create_task_issues: bool = False,
    labels: Optional[List[str]] = None
) -> str:
    """
    Create GitHub issues from an implementation plan.
    
    Args:
        implementation_plan: The implementation plan to convert to GitHub issues
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        github_token: GitHub personal access token
        create_task_issues: Whether to create separate issues for each task
        labels: Labels to apply to the issues
        
    Returns:
        URL of the created main issue
    """
    logger.info(f"Creating GitHub issues for implementation plan: {implementation_plan.feature_name}")
    
    return await self.planner.agent.run_tool(
        "create_github_issues_from_plan", 
        {
            "implementation_plan": implementation_plan,
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "github_token": github_token,
            "create_task_issues": create_task_issues,
            "labels": labels
        },
        deps=None
    )
```

## Dependencies Between Changes

- Step 1 must be completed before Step 2 as GitHub tools depend on the models
- Steps 1 and 2 must be completed before Step 3
- Steps 1, 2, and 3 must be completed before Step 4
- Steps 1-4 must be completed before Steps 5 and 6
