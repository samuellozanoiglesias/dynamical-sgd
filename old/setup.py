#!/usr/bin/env python3
"""
Setup script for dynamical-sgd package.

This package implements neural network learning dynamics analysis
using periodic batch composition and non-equilibrium physics concepts.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        "jax>=0.4.10",
        "jaxlib>=0.4.10",
        "optax>=0.1.7",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "imageio>=2.9.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "absl-py>=1.0.0"
    ]

setup(
    name="dynamical-sgd",
    version="1.0.0",
    author="Nicolas Ratier Werbin",
    author_email="nicolasratierwerbin@gmail.com",
    description="Neural network learning dynamics analysis using periodic batch composition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NicolasRW/dynamical-sgd",
    
    packages=find_packages(),
    package_dir={"": "."},
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    
    python_requires=">=3.8",
    install_requires=requirements,
    
    extras_require={
        "gpu": [
            "jax[cuda12_pip] @ https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "analysis": [
            "scikit-learn>=1.0.0",
            "networkx>=2.6",
            "plotly>=5.0.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "dynamical-sgd=run_experiment:main",
            "dynamical-sgd-systematic=systematic_train:main",
        ],
    },
    
    keywords=[
        "machine learning",
        "neural networks", 
        "dynamical systems",
        "statistical physics",
        "non-equilibrium",
        "SGD",
        "continual learning",
        "JAX"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/NicolasRW/dynamical-sgd/issues",
        "Source": "https://github.com/NicolasRW/dynamical-sgd",
        "Documentation": "https://github.com/NicolasRW/dynamical-sgd#readme",
    },
    
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.yml"],
        "docs": ["*.md"],
    },
    
    zip_safe=False,
)