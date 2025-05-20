#!/bin/bash
# Script to run gemini-update in a Docker container

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if the API key is provided
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set."
    echo "Please set it using: export GEMINI_API_KEY=your_api_key"
    exit 1
fi

# Default values
PROJECT_DIR="$(pwd)"
OUTPUT_DIR="$(pwd)/gemini-update-output"
GEMINI_MODEL=${GEMINI_MODEL:-"gemini-1.5-pro"}
USE_GEMINI_FILES=${USE_GEMINI_FILES:-"false"}
MAX_FILE_SIZE=${MAX_FILE_SIZE:-"4.0"}

# Parse command line arguments
COMMAND="analyze"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            GEMINI_MODEL="$2"
            shift 2
            ;;
        --use-gemini-files)
            USE_GEMINI_FILES="true"
            shift
            ;;
        --max-file-size)
            MAX_FILE_SIZE="$2"
            shift 2
            ;;
        analyze|feature|version)
            COMMAND="$1"
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the container
docker run --rm \
    -v "$PROJECT_DIR:/workspace:ro" \
    -v "$OUTPUT_DIR:/output" \
    -e GEMINI_API_KEY="$GEMINI_API_KEY" \
    -e GEMINI_MODEL="$GEMINI_MODEL" \
    -e GEMINI_UPDATE_OUTPUT_DIR="/output" \
    -e GEMINI_UPDATE_USE_FILES="$USE_GEMINI_FILES" \
    -e GEMINI_UPDATE_MAX_FILE_SIZE="$MAX_FILE_SIZE" \
    gemini-update:latest \
    "$COMMAND" --project-dir /workspace "${POSITIONAL_ARGS[@]}"
