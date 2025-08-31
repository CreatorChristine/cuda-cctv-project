#setup_cupy_project.py - Installation script for CuPy-focused setup
import subprocess
import sys

def install_cupy_project():
    """Install CuPy-focused CUDA project"""
    
    packages = [
        'cupy-cuda11x',  # or cupy-cuda12x for CUDA 12
        'numpy',
        'scipy', 
        'matplotlib',
        'jupyter',
        'pytest',
        'pandas',
        'scikit-image',
        'Pillow'
    ]
    
    print("Installing CuPy-focused CUDA Python environment...")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nTesting installation...")
    try:
        import cupy as cp
        print(f"✓ CuPy {cp.__version__} working")
        print(f"✓ GPU available: {cp.cuda.is_available()}")
    except ImportError:
        print("✗ CuPy installation failed")

if __name__ == "__main__":
    install_cupy_project()