#!/usr/bin/env python
"""
Example demonstrating a GitHub integration feature for the gemini-update agent.
This script creates GitHub issues from implementation plans.
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Optional
import markdown
import argparse
import logging
import json

# Add the parent directory to sys.path for local development/testing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ImplementationPlan
from src.agent.agents import ImplementationPlannerAgent
from src.agent.common import CommonGeminiTools
from src.models import Settings, CodebaseContext, FeatureSpec
from rich.console import Console
from dotenv import load_dotenv
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('github_integration')

# Constants
console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create GitHub issues from implementation plans")
    
    parser.add_argument(
        "--implementation-plan-path",
        required=True,
        help="Path to an existing implementation plan markdown file"
    )
    
    parser.add_argument(
        "--repo-owner",
        required=True,
        help="GitHub repository owner (username or organization)"
    )
    
    parser.add_argument(
        "--repo-name",
        required=True,
        help="GitHub repository name"
    )
    
    parser.add_argument(
        "--github-token",
        help="GitHub personal access token (defaults to GITHUB_TOKEN environment variable)"
    )
    
    parser.add_argument(
        "--create-task-issues",
        action="store_true",
        help="Create separate issues for each task in the implementation plan"
    )
    
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["enhancement", "gemini-update"],
        help="Labels to apply to the GitHub issues"
    )
    
    return parser.parse_args()

def read_implementation_plan(file_path: str) -> dict:
    """
    Read and parse an implementation plan from a markdown file.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with parsed implementation plan data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the markdown to extract the implementation plan structure
        lines = content.split('\n')
        
        # Extract title/feature name
        title = lines[0].replace('# Implementation Plan for ', '').strip()
        
        # Extract description
        description_section = content.split('## Description')[1].split('##')[0].strip()
        
        # Extract complexity
        complexity = "Unknown"
        for line in lines:
            if "Estimated Complexity" in line:
                complexity = line.replace('**Estimated Complexity**:', '').strip()
                break
        
        # Extract scope - affected files and new files
        scope = {
            "affected_files": [],
            "new_files": []
        }
        
        if "### Files to Modify" in content:
            files_section = content.split('### Files to Modify')[1]
            if '###' in files_section:
                files_section = files_section.split('###')[0]
            
            for line in files_section.strip().split('\n'):
                if line.startswith('- `'):
                    scope['affected_files'].append(line.replace('- `', '').replace('`', '').strip())
        
        if "### New Files to Create" in content:
            files_section = content.split('### New Files to Create')[1]
            if '###' in files_section:
                files_section = files_section.split('###')[0]
            
            for line in files_section.strip().split('\n'):
                if line.startswith('- `'):
                    scope['new_files'].append(line.replace('- `', '').replace('`', '').strip())
        
        # Extract changes
        changes = []
        if "## Implementation Steps" in content:
            changes_section = content.split('## Implementation Steps')[1]
            
            if '## Dependencies Between Changes' in changes_section:
                changes_section = changes_section.split('## Dependencies Between Changes')[0]
                
            # Split into steps by looking for "### Step" markers
            steps_content = changes_section.split('### Step')
            for step_content in steps_content[1:]:  # Skip the first empty split
                lines = step_content.strip().split('\n')
                step_title = lines[0].strip()
                
                # Extract file path if available
                file_path = None
                description = None
                
                for i, line in enumerate(lines):
                    if "**File:**" in line:
                        file_path = line.replace('**File:**', '').replace('`', '').strip()
                    elif "**Description:**" in line:
                        # Get everything until the next marker or the end
                        description_lines = []
                        j = i + 1
                        while j < len(lines) and not lines[j].strip().startswith('**'):
                            description_lines.append(lines[j])
                            j += 1
                        description = '\n'.join(description_lines).strip()
                
                # Extract code snippet if available
                code_snippet = None
                if '```' in step_content:
                    code_parts = step_content.split('```')
                    if len(code_parts) > 2:  # Ensure there are actually code blocks
                        # Get the first code block (ignore the language specifier line)
                        code_lines = code_parts[1].strip().split('\n')
                        if len(code_lines) > 1:
                            code_snippet = '\n'.join(code_lines[1:])
                        else:
                            code_snippet = code_lines[0]
                
                changes.append({
                    "title": step_title,
                    "file_path": file_path,
                    "description": description,
                    "code_snippet": code_snippet
                })
        
        # Extract dependencies
        dependencies = []
        if "## Dependencies Between Changes" in content:
            deps_section = content.split('## Dependencies Between Changes')[1].strip()
            for line in deps_section.split('\n'):
                if line.startswith('- '):
                    dependencies.append(line.replace('- ', '').strip())
        
        # Create the implementation plan data structure
        plan = {
            "feature_name": title,
            "description": description_section,
            "estimated_complexity": complexity,
            "scope": scope,
            "changes": changes,
            "dependencies": dependencies
        }
        
        return plan
    
    except Exception as e:
        logger.error(f"Error parsing implementation plan: {str(e)}")
        raise

def create_github_issue(
    repo_owner: str, 
    repo_name: str, 
    token: str, 
    title: str, 
    body: str, 
    labels: Optional[list] = None
) -> dict:
    """
    Create a GitHub issue.
    
    Args:
        repo_owner: Repository owner (username or organization)
        repo_name: Repository name
        token: GitHub personal access token
        title: Issue title
        body: Issue body in markdown format
        labels: Optional list of labels to apply to the issue
    
    Returns:
        Response data from GitHub API
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    
    headers = {
        "Authorization": f"token {token}",
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

def main():
    """Main function to create GitHub issues from an implementation plan."""
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Get GitHub token from args or environment
    github_token = args.github_token or os.getenv('GITHUB_TOKEN')
    if not github_token:
        console.print("[bold red]Error: GitHub token not provided. Use --github-token or set GITHUB_TOKEN environment variable.[/bold red]")
        sys.exit(1)
    
    try:
        # Read and parse the implementation plan
        console.print(f"[bold blue]Reading implementation plan from {args.implementation_plan_path}...[/bold blue]")
        plan = read_implementation_plan(args.implementation_plan_path)
        
        # Create the main issue with the implementation plan
        main_issue_title = f"Implement: {plan['feature_name']}"
        
        # Prepare the main issue body
        main_issue_body = f"# {plan['feature_name']}\n\n"
        main_issue_body += f"{plan['description']}\n\n"
        main_issue_body += f"**Complexity**: {plan['estimated_complexity']}\n\n"
        
        # Add scope information
        main_issue_body += "## Scope\n\n"
        
        if plan['scope']['affected_files']:
            main_issue_body += "### Files to Modify\n\n"
            for file in plan['scope']['affected_files']:
                main_issue_body += f"- `{file}`\n"
            main_issue_body += "\n"
        
        if plan['scope']['new_files']:
            main_issue_body += "### New Files to Create\n\n"
            for file in plan['scope']['new_files']:
                main_issue_body += f"- `{file}`\n"
            main_issue_body += "\n"
        
        # Add implementation steps overview
        main_issue_body += "## Implementation Steps\n\n"
        for i, change in enumerate(plan['changes'], 1):
            main_issue_body += f"{i}. {change['title']}"
            if change['file_path']:
                main_issue_body += f" (`{change['file_path']}`)"
            main_issue_body += "\n"
        main_issue_body += "\n"
        
        # Add dependencies
        if plan['dependencies']:
            main_issue_body += "## Dependencies\n\n"
            for dep in plan['dependencies']:
                main_issue_body += f"- {dep}\n"
            main_issue_body += "\n"
        
        # Add a note about the source
        main_issue_body += "\n---\n"
        main_issue_body += "_This issue was automatically generated by the gemini-update agent._\n"
        
        # Create the main GitHub issue
        console.print("[bold blue]Creating main GitHub issue...[/bold blue]")
        main_issue = create_github_issue(
            args.repo_owner,
            args.repo_name,
            github_token,
            main_issue_title,
            main_issue_body,
            args.labels
        )
        
        console.print(f"[bold green]Main issue created successfully: {main_issue['html_url']}[/bold green]")
        
        # Create task issues if requested
        if args.create_task_issues and plan['changes']:
            console.print("[bold blue]Creating task issues...[/bold blue]")
            for i, change in enumerate(plan['changes'], 1):
                task_title = f"Task {i}: {change['title']}"
                
                # Create task issue body
                task_body = f"# {change['title']}\n\n"
                
                if change['file_path']:
                    task_body += f"**File**: `{change['file_path']}`\n\n"
                
                if change['description']:
                    task_body += f"**Description**:\n\n{change['description']}\n\n"
                
                if change['code_snippet']:
                    task_body += "**Code**:\n\n```\n"
                    task_body += f"{change['code_snippet']}\n"
                    task_body += "```\n\n"
                
                # Add reference to main issue
                task_body += f"\n\nPart of #{main_issue['number']} ({plan['feature_name']})\n\n"
                task_body += "\n---\n"
                task_body += "_This task issue was automatically generated by the gemini-update agent._\n"
                
                # Create the task issue
                task_issue = create_github_issue(
                    args.repo_owner,
                    args.repo_name,
                    github_token,
                    task_title,
                    task_body,
                    args.labels + ["task"]
                )
                
                console.print(f"[green]Task issue created: {task_issue['html_url']}[/green]")
        
        console.print("\n[bold green]✓ GitHub issues created successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
