"""
Setup configuration for polocalc-data-loader package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="pils",
    version="0.1.0",
    description="PILS - POLOCALC Inertial & Drone Loading System. Integrated flight data loading and decoding with STOUT database interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="POLOCALC Team",
    author_email="info@polocalc.org",
    url="https://github.com/POLOCALC/data-loader",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "opencv-python>=4.6.0",
        "matplotlib>=3.5.0",
        "pyubx2>=1.2.0",
    ],
    extras_require={
        "stout": ["stout>=2.0.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="drone data-loading sensors stout database",
)
