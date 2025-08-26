# CUDA Python Project

A collaborative CUDA Python project for GPU-accelerated computing.

## Setup for Windows + VS Code

1. Clone the repository
2. Open in VS Code
3. Create virtual environment: `python -m venv cuda_env`
4. Activate environment: `cuda_env\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Install recommended VS Code extensions (you'll get a popup)

## Prerequisites

- Windows 10/11
- NVIDIA GPU with CUDA capability
- CUDA Toolkit 11.x or 12.x
- Python 3.8+
- VS Code
- Git for Windows

## Usage

### VS Code Integration

- **F5**: Run current file with debugging
- **Ctrl+F5**: Run without debugging  
- **Ctrl+Shift+P** → `Python: Select Interpreter` → Choose `./cuda_env/Scripts/python.exe`
- **Ctrl+Shift+` : Open integrated terminal
- **Ctrl+Shift+P** → `Tasks: Run Task` → Select `Run Main App` or `Run Tests`

### Command Line

Run the main application:
```cmd
cuda_env\Scripts\activate
python src\main.py
```

Run tests:
```cmd
pytest tests\ -v
```

## Project Structure

- `src/`: Source code
  - `kernels/`: CUDA kernel implementations
  - `utils/`: Utility functions
- `tests/`: Test files
- `docs/`: Documentation
- `examples/`: Example scripts
- `data/`: Data files
- `.vscode/`: VS Code configuration

## Team Workflow

- Main branch: Production code
- Develop branch: Integration branch  
- Feature branches: Individual development
- Use pull requests for all merges

## Debugging in VS Code

1. Set breakpoints by clicking left of line numbers
2. Press F5 to start debugging
3. Use Debug Console for CUDA-specific debugging
4. View variables in the Variables pane
