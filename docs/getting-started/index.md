# Getting Started

This section will guide you through installing PILS and running your first flight analysis.

## Prerequisites

Before installing PILS, ensure you have:

- **Python 3.10+** (required for modern type hints)
- **Conda** (recommended for environment management)
- **Git** (for cloning the repository)

## Installation Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __[Installation](installation.md)__

    ---

    Complete installation guide with conda environment setup

-   :material-play:{ .lg .middle } __[Quick Start](quickstart.md)__

    ---

    5-minute guide to your first flight analysis

-   :material-airplane:{ .lg .middle } __[First Flight](first-flight.md)__

    ---

    Detailed walkthrough of loading and analyzing a flight

</div>

## Recommended Setup

```bash
# Clone the repository
git clone https://github.com/polocalc/pils.git
cd pils

# Create conda environment
conda create -n dm python=3.10 -y
conda activate dm

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import pils; print('PILS installed successfully')"
```

## What's Next?

After installation, proceed to the [Quick Start](quickstart.md) guide to run your first analysis.
