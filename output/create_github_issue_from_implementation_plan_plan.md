# Implementation Plan for Create GitHub Issue from Implementation Plan

*Generated at: 2025-05-20 10:59:14*

## Description

Add functionality to generate and create a GitHub issue directly from a generated implementation plan, pre-populating the issue with details from the plan.

**Estimated Complexity**: Medium

## Scope

### Files to Modify

- `backend/api_routes.py`
- `frontend/plan_viewer.html`

### New Files to Create

- `backend/github_service.py`

### Dependencies to Add

- `requests`

## Implementation Steps

### Step 1: Add backend/github_service.py

**Description:** Create a new Python module for GitHub API interactions.

**Code:**

```
# backend/github_service.py

import requests
import os # Example for token retrieval

class GitHubService:
    def __init__(self, auth_token=None):
        # In a real app, handle token securely (e.g., OAuth flow, config management)
        self.auth_token = auth_token or os.environ.get("GITHUB_TOKEN")
        if not self.auth_token:
            # Handle authentication failure appropriately
            raise ValueError("GitHub authentication token not provided.")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.auth_token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def create_issue(self, owner: str, repo: str, title: str, body: str):
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        data = {
            "title": title,
            "body": body
        }
        try:
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json() # Returns the created issue object
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error creating GitHub issue: {e.response.status_code} - {e.response.text}")
            # Implement specific error handling (e.g., 401, 404, 403 rate limit)
            if e.response.status_code == 403 and 'rate limit exceeded' in e.response.text.lower():
                 # Handle rate limit - potentially raise a specific exception or wait/retry
                 pass # Placeholder
            raise # Re-raise the exception after logging/handling
        except requests.exceptions.RequestException as e:
            print(f"Error creating GitHub issue: {e}")
            raise # Re-raise other request exceptions

# Helper function to format plan into markdown
def format_plan_for_github_issue(plan_data: dict) -> str:
    # Assuming plan_data is the dictionary representation of the ImplementationPlan
    body = f"# Feature: {plan_data.get('feature_name', 'N/A')}\n\n"
    body += f"**Description:**\n{plan_data.get('description', 'N/A')}\n\n"
    body += f"**Estimated Complexity:** {plan_data.get('estimated_complexity', 'N/A')}\n\n"

    scope = plan_data.get('scope', {})
    if scope:
        body += "**Scope:**\n"
        if scope.get('affected_files'):
            body += "- **Affected Files:**\n"
            for f in scope['affected_files']:
                body += f"  - `{f}`\n"
        if scope.get('new_files'):
            body += "- **New Files:**\n"
            for f in scope['new_files']:
                body += f"  - `{f}`\n"
        if scope.get('dependencies_needed'):
             body += "- **Dependencies Needed:**\n"
             for d in scope['dependencies_needed']:
                 body += f"  - `{d}`\n"
        if scope.get('config_changes'):
             body += "- **Configuration Changes:**\n"
             for c in scope['config_changes']:
                 body += f"  - {c}\n"
        body += "\n"

    changes = plan_data.get('changes', [])
    if changes:
        body += "**Implementation Changes:**\n\n"
        for i, change in enumerate(changes):
            body += f"### Change {i+1}: {change.get('description', 'No description')}\n"
            body += f"- **File:** `{change.get('file_path', 'N/A')}`\n"
            body += f"- **Type:** `{change.get('change_type', 'N/A')}`\n"
            if change.get('line_range'):
                 body += f"- **Lines:** `{change.get('line_range', 'N/A')}`\n"
            if change.get('code_snippet'):
                body += "\n```\n"
                body += change['code_snippet']
                body += "\n```\n"
            body += "\n"

    dependencies = plan_data.get('dependencies', [])
    if dependencies:
        body += "**Dependencies between Changes:**\n"
        for dep in dependencies:
            body += f"- {dep}\n"
        body += "\n"

    # Add a note about the source
    body += "---\n*Generated from an implementation plan.*\n"

    return body

def generate_issue_title(feature_name: str) -> str:
    return f"Implement Feature: {feature_name}"

```

### Step 2: Modify backend/api_routes.py

**Description:** Add a new API endpoint to receive the implementation plan and repository details, then call the GitHub service.

**Code:**

```
# backend/api_routes.py

# Assuming a framework like FastAPI
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .github_service import GitHubService, format_plan_for_github_issue, generate_issue_title

router = APIRouter()

class CreateGitHubIssueRequest(BaseModel):
    owner: str
    repo: str
    plan_data: dict # Assuming the plan data is sent as a dictionary

@router.post("/create_github_issue")
async def create_github_issue(request: CreateGitHubIssueRequest):
    try:
        # Initialize GitHubService (handle auth_token securely in a real app)
        github_service = GitHubService() # Token should be managed securely

        title = generate_issue_title(request.plan_data.get('feature_name', 'New Feature'))
        body = format_plan_for_github_issue(request.plan_data)

        issue = github_service.create_issue(request.owner, request.repo, title, body)

        return {
            "message": "GitHub issue created successfully!",
            "issue_url": issue.get("html_url")
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Authentication Error: {e}")
    except Exception as e:
        # Catch other potential errors from the service (API errors, etc.)
        print(f"Error in /create_github_issue endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create GitHub issue: {e}")

# ... other routes ...

```

### Step 3: Modify frontend/plan_viewer.html

**Description:** Add UI elements (button, repo input) and JavaScript logic to trigger the GitHub issue creation endpoint.

**Code:**

```
<!-- frontend/plan_viewer.html -->

<!-- ... existing plan display ... -->

<div id="github-issue-section">
    <h3>Create GitHub Issue</h3>
    <label for="github-repo">Repository (owner/repo):</label>
    <input type="text" id="github-repo" placeholder="e.g., google/gemini-api-cookbook">
    <button id="create-github-issue-btn">Create GitHub Issue</button>
    <div id="github-issue-status"></div>
</div>

<script>
    // Assuming planData is available globally or passed to this script
    // let planData = { ... }; // The generated implementation plan object

    document.getElementById('create-github-issue-btn').addEventListener('click', async () => {
        const repoInput = document.getElementById('github-repo');
        const statusDiv = document.getElementById('github-issue-status');
        const repo = repoInput.value.trim();

        if (!repo) {
            statusDiv.innerText = 'Please enter a repository (owner/repo).';
            statusDiv.style.color = 'red';
            return;
        }

        const repoParts = repo.split('/');
        if (repoParts.length !== 2 || !repoParts[0] || !repoParts[1]) {
             statusDiv.innerText = 'Invalid repository format. Use owner/repo.';
             statusDiv.style.color = 'red';
             return;
        }

        statusDiv.innerText = 'Creating GitHub issue...';
        statusDiv.style.color = 'black';

        try {
            // Assuming planData is the variable holding the current plan
            const response = await fetch('/api/create_github_issue', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    owner: repoParts[0],
                    repo: repoParts[1],
                    plan_data: planData // Send the plan data
                })
            });

            const result = await response.json();

            if (response.ok) {
                statusDiv.innerHTML = `Success! Issue created: <a href="${result.issue_url}" target="_blank">${result.issue_url}</a>`;
                statusDiv.style.color = 'green';
            } else {
                statusDiv.innerText = `Error: ${result.detail || 'Unknown error'}`;
                statusDiv.style.color = 'red';
            }
        } catch (error) {
            statusDiv.innerText = `Request failed: ${error}`;
            statusDiv.style.color = 'red';
            console.error('Error creating GitHub issue:', error);
        }
    });
</script>

<!-- ... rest of the HTML ... -->

```

## Dependencies Between Changes

- backend/github_service.py must be created before backend/api_routes.py is modified.
- backend/api_routes.py must be modified before frontend/plan_viewer.html is modified.

