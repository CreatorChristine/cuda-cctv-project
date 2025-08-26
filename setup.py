from setuptools import setup, find_packages

setup(
    name="cuda-python-project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "cupy-cuda11x>=11.0.0",
        "numpy>=1.21.0",
        "numba>=0.56.0",
    ],
    python_requires=">=3.8",
)
