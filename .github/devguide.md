# Gemini Update Agent

This repository contains code for the Gemini Update Agent, which combines codebase analysis and feature planning.

## Structure

- `src/agent/agent.py`: Main agent implementation
- `src/models/analysis.py`: Models for codebase analysis
- `src/models/feature.py`: Models for feature specifications
- `src/tools/codebase_tools.py`: Tools for interacting with the codebase
- `src/tools/feature_tools.py`: Tools for feature planning

## Development Instructions

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the demo:
   ```bash
   python demo.py
   ```

4. Run tests:
   ```bash
   python -m unittest discover -s tests
   ```

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key

## Debugging

Use the provided VS Code launch configurations in the `.vscode` folder for easy debugging.
