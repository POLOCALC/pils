# Development

Contributing to and extending PILS.

## Sections

<div class="grid cards" markdown>

-   :material-map: **[Architecture](architecture.md)**
    
    System design and component overview
    
    ---
    
    Class hierarchy • Data flow • Design patterns

-   :material-test-tube: **[Testing](testing.md)**
    
    Test-driven development practices
    
    ---
    
    pytest • Coverage • TDD workflow

-   :material-account-plus: **[Contributing](contributing.md)**
    
    How to contribute to PILS
    
    ---
    
    Git workflow • Pull requests • Code review

-   :material-plus-circle: **[Adding Sensors](adding-sensors.md)**
    
    Extend PILS with new sensor types
    
    ---
    
    Sensor interface • Registration • Examples

-   :material-format-paint: **[Code Style](code-style.md)**
    
    Coding standards and conventions
    
    ---
    
    Python style • Type hints • Documentation

-   :material-format-paint: **[Code Style](versioning.md)**
    
    Versioning rules for PILS
    
    ---
    
    Versioninig rules explained

</div>

## Quick Links

- [GitHub Repository](https://github.com/polocalc/pils)
- [Issue Tracker](https://github.com/polocalc/pils/issues)
- [Changelog](../changelog.md)

## Development Setup

```bash
# Clone repository
git clone https://github.com/polocalc/pils.git
cd pils

# Create conda environment
conda create -n dm python=3.10
conda activate dm

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black pils/ tests/
isort pils/ tests/
```

---

## See Also

- [Getting Started](../getting-started/index.md) - User installation
- [API Reference](../api/index.md) - API documentation
