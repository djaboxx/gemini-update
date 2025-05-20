#!/bin/bash
set -e

echo "Setting up Gemini Update dev environment..."

# Install any additional dev dependencies not in requirements.txt
pip install --user black isort pylint pytest-cov

# Create output directory if it doesn't exist
mkdir -p /workspaces/gemini-update/output

# Setup pre-commit hooks if .git directory exists and pre-commit is installed
if [ -d "/workspaces/gemini-update/.git" ] && pip list | grep -q pre-commit; then
    echo "Setting up pre-commit hooks..."
    cd /workspaces/gemini-update
    pre-commit install
fi

echo "Setup complete!"
