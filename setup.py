"""
Setup script for the VolSSCORE package.

This script defines the package metadata and dependencies required for
installing and distributing the VolSSCORE package.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="volscore",
    version="0.1.0",
    description="A Python-based screening platform implementing the Barclays short-straddle strategy for overpriced volatility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VolSSCORE Team",
    author_email="volscore@example.com",
    url="https://github.com/volscore/volscore",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "yfinance>=0.1.70",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tabulate>=0.8.9",
        "plotly>=5.3.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
        "colorama>=0.4.4",
        "joblib>=1.0.0",
        "requests>=2.26.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "mypy>=0.812",
        ]
    },
    entry_points={
        "console_scripts": [
            "volscore=volscore.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    keywords="finance, options, volatility, trading, stocks, screening",
)