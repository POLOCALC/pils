from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pils",
    version="0.1.0",
    author="POLOCALC",
    author_email="",  # Add your email if desired
    description="Tools to load and visualize data from drone missions, including drone logs, payload sensor data, and Litchi flight logs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/POLOCALC/data_loader",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "pyubx2",
        "astropy",
        "opencv-python",  # Required for camera.py - use PyPI version to avoid conda conflicts
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "report": [
            "jinja2",
            "weasyprint",
            "markdown",
        ],
    },
    entry_points={
        "console_scripts": [
            "polocalc-quick-look=utils.quick_look:main",
            "polocalc-process-data=utils.process_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
    },
)
