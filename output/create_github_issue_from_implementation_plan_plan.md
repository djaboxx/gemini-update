# Implementation Plan for Create GitHub Issue from Implementation Plan

*Generated at: 2025-05-20 00:38:47*

## Description

Allows users to generate a GitHub issue directly from a generated implementation plan, populating the issue with details from the plan.

**Estimated Complexity**: Medium

## Scope

### Files to Modify

- `frontend/src/components/ImplementationPlanView.js`
- `backend/src/config.py`
- `backend/src/main.py`

### New Files to Create

- `frontend/src/components/GitHubIssueModal.js`
- `frontend/src/services/githubService.js`
- `backend/src/routes/github.py`
- `backend/src/services/github_service.py`

### Dependencies to Add

- `requests (backend)`
- `FastAPI (backend, if not already used)`
- `Pydantic (backend, if not already used)`

### Configuration Changes

- Add GITHUB_TOKEN to backend configuration/environment variables.

## Implementation Steps

### Step 1: Modify frontend/src/components/ImplementationPlanView.js

**Description:** Add 'Create GitHub Issue' button to the implementation plan view.

**Code:**

```
// Assuming a React-like structure
<div className="implementation-plan-view">
  {/* ... existing plan details ... */}
  <button onClick={handleCreateGitHubIssue}>Create GitHub Issue</button>
  {/* ... rest of the component ... */}
</div>
```

### Step 2: Add frontend/src/components/GitHubIssueModal.js

**Description:** Create a new modal component for collecting GitHub issue details.

**Code:**

```
// Basic structure for a modal form
import React, { useState } from 'react';

function GitHubIssueModal({ isOpen, onClose, onSubmit, planDetails }) {
  const [repository, setRepository] = useState('');
  const [titlePrefix, setTitlePrefix] = useState('');
  const [assignees, setAssignees] = useState(''); // Comma-separated usernames
  const [labels, setLabels] = useState(''); // Comma-separated labels

  const handleSubmit = () => {
    onSubmit({
      repository,
      titlePrefix,
      assignees: assignees.split(',').map(s => s.trim()).filter(s => s),
      labels: labels.split(',').map(s => s.trim()).filter(s => s),
      planDetails,
    });
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>Create GitHub Issue</h2>
        <label>Repository (e.g., owner/repo):</label>
        <input type="text" value={repository} onChange={e => setRepository(e.target.value)} />

        <label>Title Prefix (Optional):</label>
        <input type="text" value={titlePrefix} onChange={e => setTitlePrefix(e.target.value)} />

        <label>Assignees (comma-separated usernames, Optional):</label>
        <input type="text" value={assignees} onChange={e => setAssignees(e.target.value)} />

        <label>Labels (comma-separated, Optional):</label>
        <input type="text" value={labels} onChange={e => setLabels(e.target.value)} />

        <button onClick={handleSubmit}>Create Issue</button>
        <button onClick={onClose}>Cancel</button>
      </div>
    </div>
  );
}

export default GitHubIssueModal;
```

### Step 3: Modify frontend/src/components/ImplementationPlanView.js

**Description:** Integrate the GitHub issue modal into the implementation plan view component and handle button click.

**Code:**

```
// Assuming a React-like structure
import React, { useState } from 'react';
import GitHubIssueModal from './GitHubIssueModal';
import { createGitHubIssue } from '../services/githubService'; // Assuming a service file

function ImplementationPlanView({ plan }) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleCreateGitHubIssue = () => {
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
  };

  const handleModalSubmit = async (issueDetails) => {
    try {
      // Call backend service to create issue
      const result = await createGitHubIssue(issueDetails);
      console.log('Issue created:', result);
      // Display success message (req-8)
      alert(`Issue created successfully: ${result.html_url}`);
    } catch (error) {
      console.error('Error creating issue:', error);
      // Display error message (req-9)
      alert(`Failed to create issue: ${error.message}`);
    } finally {
      setIsModalOpen(false);
    }
  };

  return (
    <div className="implementation-plan-view">
      {/* ... existing plan details ... */}
      <button onClick={handleCreateGitHubIssue}>Create GitHub Issue</button>
      {/* ... rest of the component ... */}

      <GitHubIssueModal
        isOpen={isModalOpen}
        onClose={handleModalClose}
        onSubmit={handleModalSubmit}
        planDetails={plan} // Pass the current plan details
      />
    </div>
  );
}

export default ImplementationPlanView;
```

### Step 4: Add frontend/src/services/githubService.js

**Description:** Create a frontend service to call the backend API for GitHub issue creation.

**Code:**

```
// Basic service file
import api from './api'; // Assuming an existing API client setup

export const createGitHubIssue = async (issueDetails) => {
  try {
    const response = await api.post('/api/github/create-issue', issueDetails);
    return response.data;
  } catch (error) {
    console.error('API call error:', error);
    throw error; // Re-throw to be caught by the component
  }
};
```

### Step 5: Add backend/src/routes/github.py

**Description:** Create a new backend endpoint to handle GitHub issue creation requests.

**Code:**

```
# Example using FastAPI
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ..services.github_service import GitHubService # Assuming a service file
from ..dependencies import get_github_service # Assuming dependency injection

router = APIRouter()

class CreateIssueRequest(BaseModel):
    repository: str
    titlePrefix: Optional[str] = None
    assignees: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    planDetails: dict # Structure matches the ImplementationPlan object

@router.post('/create-issue')
async def create_github_issue(request: CreateIssueRequest, github_service: GitHubService = Depends(get_github_service)):
    try:
        issue_url = await github_service.create_issue(
            repository=request.repository,
            title_prefix=request.titlePrefix,
            assignees=request.assignees,
            labels=request.labels,
            plan_details=request.planDetails
        )
        return {'html_url': issue_url}
    except Exception as e:
        # Basic error handling, needs refinement based on specific API errors (req-15)
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 6: Add backend/src/services/github_service.py

**Description:** Create a backend service class for interacting with the GitHub API.

**Code:**

```
# Basic structure for a GitHub service
import os # Example for getting token from env var
import requests # Using requests library

class GitHubService:
    def __init__(self, github_token: str):
        self.base_url = 'https://api.github.com'
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def format_issue_body(self, plan_details: dict) -> str:
        # Implement mapping from plan_details to Markdown body (req-6, req-14)
        body = f"""# Implementation Plan: {plan_details.get('feature_name', 'N/A')}\n\n"
        body += f"## Description\n{plan_details.get('description', 'N/A')}\n\n"
        body += f"## Scope\n"
        scope = plan_details.get('scope', {})
        if scope.get('affected_files'):
            body += f"- **Affected Files:** {', '.join(scope['affected_files'])}\n"
        if scope.get('new_files'):
            body += f"- **New Files:** {', '.join(scope['new_files'])}\n"
        if scope.get('dependencies_needed'):
             body += f"- **Dependencies Needed:** {', '.join(scope['dependencies_needed'])}\n"
        if scope.get('config_changes'):
             body += f"- **Config Changes:** {', '.join(scope['config_changes'])}\n"
        body += f"\n"

        body += f"## Changes\n"
        changes = plan_details.get('changes', [])
        if changes:
            for change in changes:
                body += f"- **{change.get('change_type', 'Change')}** {change.get('file_path', 'N/A')}: {change.get('description', 'N/A')}\n"
                # Optionally include code snippet, handle large snippets (req-6)
                # if change.get('code_snippet'):
                #     body += f"\n```\n{change['code_snippet'][:500]}...\n```\n"
        else:
            body += "No specific code changes detailed.\n"
        body += f"\n"

        body += f"## Estimated Complexity\n{plan_details.get('estimated_complexity', 'N/A')}\n\n"

        # Add dependencies if available
        dependencies = plan_details.get('dependencies')
        if dependencies:
             body += f"## Dependencies\n"
             for dep in dependencies:
                 body += f"- {dep}\n"
             body += f"\n"

        return body

    async def create_issue(
        self, repository: str, title_prefix: Optional[str], assignees: Optional[List[str]], labels: Optional[List[str]], plan_details: dict
    ) -> str:
        # Implement GitHub API call to create issue (req-7, req-12)
        url = f'{self.base_url}/repos/{repository}/issues'

        title = plan_details.get('feature_name', 'New Issue from Plan')
        if title_prefix:
            title = f'{title_prefix} {title}'

        body = self.format_issue_body(plan_details)

        payload = {
            'title': title,
            'body': body,
            'assignees': assignees or [],
            'labels': labels or []
        }

        response = requests.post(url, headers=self.headers, json=payload)

        # Handle API response and errors (req-9, req-10, req-15)
        if response.status_code == 201:
            return response.json()['html_url']
        elif response.status_code == 401:
             raise Exception('GitHub Authentication failed. Check your token.')
        elif response.status_code == 404:
             raise Exception(f'Repository not found: {repository}')
        elif response.status_code == 403 and 'rate limit exceeded' in response.text:
             raise Exception('GitHub API rate limit exceeded. Please wait before trying again.') # req-10
        else:
            raise Exception(f'Failed to create GitHub issue: {response.status_code} - {response.text}')

# Example dependency provider (needs actual implementation)
def get_github_service():
    # Securely retrieve GitHub token (req-11)
    github_token = os.environ.get('GITHUB_TOKEN') # Example: get from environment variable
    if not github_token:
        raise Exception('GitHub token not configured.')
    return GitHubService(github_token=github_token)
```

### Step 7: Modify backend/src/config.py

**Description:** Add configuration for GitHub token (e.g., environment variable, config file).

**Code:**

```
# Example config file
import os

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN') # Securely retrieve token
```

### Step 8: Modify backend/src/main.py

**Description:** Update main backend app to include the new GitHub router.

**Code:**

```
# Example FastAPI app
from fastapi import FastAPI
from .routes import github # Import the new router

app = FastAPI()

# Include the new router
app.include_router(github.router, prefix='/api/github', tags=['github'])

# ... other routes ...
```

## Dependencies Between Changes

- Implement backend GitHubService (backend/src/services/github_service.py) before creating the backend endpoint (backend/src/routes/github.py).
- Implement the backend endpoint (backend/src/routes/github.py) before implementing the frontend service (frontend/src/services/githubService.js).
- Implement the frontend modal component (frontend/src/components/GitHubIssueModal.js) before integrating it into the ImplementationPlanView (frontend/src/components/ImplementationPlanView.js).
- Securely configure the GITHUB_TOKEN before running the backend service.

