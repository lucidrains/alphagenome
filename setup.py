"""
Setup script for bactagenome
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bactagenome",
    version="0.1.0",
    author="bactagenome Team",
    author_email="bactagenome@example.com",
    description="Bacterial genome modeling with AlphaGenome architecture",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/bactagenome",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bactagenome-train=scripts.train:main",
            "bactagenome-evaluate=scripts.evaluate:main",
            "bactagenome-preprocess=scripts.preprocess_data:main",
            "bactagenome-download=scripts.download_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bactagenome": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
    },
)