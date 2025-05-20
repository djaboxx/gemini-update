# Gemini Update Agent

Gemini Update Agent is a Python tool that combines codebase analysis and feature planning capabilities. It leverages Google's Gemini AI models to understand your project and help implement new features by analyzing your code and generating detailed implementation plans.

## Features

* **Codebase Analysis**: Analyzes your codebase to understand its structure, dependencies, and architecture
* **Feature Specification**: Generates detailed feature specifications from natural language descriptions
* **Implementation Planning**: Creates step-by-step implementation plans for new features
* **Markdown Output**: All outputs are saved as well-formatted markdown files
* **Gemini Files API**: Support for analyzing codebases using Gemini's Files API capabilities

## Prerequisites

* Python 3.11+
* Google Gemini API Key (see [Google AI Studio](https://ai.google.dev/))

## Installation

```bash
# Clone the repository
git clone https://github.com/your-organization/gemini-update.git
cd gemini-update

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Set Environment Variables

You can set environment variables directly:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or copy the `.env.example` file to `.env` and edit it with your values:

```bash
cp .env.example .env
# Now edit .env with your API key and other settings
```

### Analyze a Codebase

```bash
./gemini-update.py analyze --project-dir /path/to/your/project
```

### Generate a Feature Specification and Implementation Plan

```bash
./gemini-update.py feature --project-dir /path/to/your/project --feature-description "Add dark mode support to the UI components"
```

### Using Gemini Files API

To leverage the Gemini Files API for improved codebase analysis:

```bash
# For codebase analysis
./gemini-update.py analyze --project-dir /path/to/your/project --use-gemini-files

# For feature planning
./gemini-update.py feature --project-dir /path/to/your/project --feature-description "Your feature description" --use-gemini-files
```

Using the Gemini Files API enables:

* More reliable code context for the AI model
* Support for codebases hosted on remote systems
* Consistent file handling across different environments

#### File Size Limits

By default, files up to 4MB can be uploaded to the Gemini Files API. You can adjust this limit:

```bash
./gemini-update.py feature --project-dir /path/to/your/project --feature-description "..." --use-gemini-files --max-file-size 6.0
```

### Command Options

```
usage: gemini-update.py [-h] {analyze,feature,version} ...

Gemini Update - Analyze codebases and plan feature implementations

positional arguments:
  {analyze,feature,version}
                        Command to run
    analyze             Analyze a codebase
    feature             Generate a feature specification and implementation plan
    version             Show version information

options:
  -h, --help            show this help message and exit
```

## Docker Usage

You can also run Gemini Update Agent in a Docker container:

### Building the Docker Image

```bash
docker build -t gemini-update .
```

### Using the Docker Image

```bash
# Analyze a codebase
docker run --rm \
  -v /path/to/your/project:/workspace:ro \
  -v /path/to/output:/output \
  -e GEMINI_API_KEY="your_api_key_here" \
  gemini-update analyze --project-dir /workspace

# Generate a feature plan
docker run --rm \
  -v /path/to/your/project:/workspace:ro \
  -v /path/to/output:/output \
  -e GEMINI_API_KEY="your_api_key_here" \
  gemini-update feature \
  --project-dir /workspace \
  --feature-description "Add support for OAuth authentication"

# Using Gemini Files API
docker run --rm \
  -v /path/to/your/project:/workspace:ro \
  -v /path/to/output:/output \
  -e GEMINI_API_KEY="your_api_key_here" \
  -e GEMINI_UPDATE_USE_FILES="true" \
  gemini-update analyze --project-dir /workspace
```

### Using Docker Compose

The repository includes a Docker Compose file for easier container management:

```bash
# Set environment variables
export GEMINI_API_KEY="your_api_key_here"
export PROJECT_DIR="/path/to/your/project" 
export OUTPUT_DIR="/path/to/output"

# Run the container with Docker Compose
docker-compose up
```

### Using the Helper Script

A helper script is provided for convenience:

```bash
# Make the script executable
chmod +x gemini-update.sh

# Run the script
export GEMINI_API_KEY="your_api_key_here"
./gemini-update.sh analyze --project-dir /path/to/your/project

# Use Gemini Files API
export GEMINI_API_KEY="your_api_key_here"
./gemini-update.sh analyze --project-dir /path/to/your/project --use-gemini-files
```

## Environment Variables

The following environment variables are supported:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | (Required) |
| `GEMINI_MODEL` | Gemini model to use | `gemini-1.5-pro` |
| `GEMINI_UPDATE_OUTPUT_DIR` | Directory to save output files | Current directory |
| `GEMINI_UPDATE_USE_FILES` | Enable Gemini Files API | `false` |
| `GEMINI_UPDATE_MAX_FILE_SIZE` | Maximum file size in MB for Gemini Files | `4.0` |
| `GEMINI_UPDATE_MAX_FILES` | Maximum number of files to analyze | `100` |

## How It Works

1. **Codebase Analysis**: The agent scans your project directory, identifies file types, analyzes imports and dependencies, and builds a structural model of your codebase.

2. **Feature Specification**: Using natural language processing, the agent converts your feature description into a detailed specification with requirements, acceptance criteria, and technical notes.

3. **Implementation Planning**: Based on the codebase analysis and feature specification, the agent creates a detailed plan showing which files to modify or create, with specific code suggestions.

4. **Output Generation**: All results are saved as markdown files in the specified output directory.

## License

MIT

## Credits

This project builds on concepts from:
- [Gemini Stack Trace](https://github.com/happypathway/gemini-stacktrace)
- [Terraform Prompt Template](https://github.com/example/terraform-prompt-template)
- [Gemini Workspace](https://github.com/example/GeminiWorkspace)
