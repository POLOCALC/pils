# Contributing

How to contribute to PILS.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Conda (recommended)

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/pils.git
cd pils

# Create conda environment
conda create -n dm python=3.10
conda activate dm

# Install in development mode
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

---

## Git Workflow

### Branch Naming

```
feature/add-new-sensor
bugfix/fix-gps-parsing
docs/update-readme
refactor/cleanup-loader
```

### Commit Messages

Follow conventional commits:

```
feat: add barometer sensor support
fix: correct GPS timestamp parsing
docs: update installation guide
test: add IMU synchronization tests
refactor: simplify loader interface
```

### Pull Request Process

1. **Create branch** from `main`
2. **Write tests** first (TDD)
3. **Implement feature**
4. **Run tests** and linting
5. **Push** and create PR
6. **Address review** comments
7. **Merge** when approved

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push to fork
git push origin feature/new-feature

# Create PR on GitHub
```

---

## Code Standards

### Must Follow

- [x] Type hints on all functions
- [x] Docstrings for public functions
- [x] Tests before implementation (TDD)
- [x] Polars for all DataFrames
- [x] Pathlib for file operations
- [x] F-strings for formatting
- [x] Logging, not print statements

### Formatting

```bash
# Format code
black pils/ tests/

# Sort imports
isort pils/ tests/

# Lint
flake8 pils/ tests/

# Type check
mypy pils/

# Run all
black pils/ && isort pils/ && flake8 pils/ && mypy pils/
```

---

## Test-Driven Development

### TDD Cycle

1. **RED**: Write failing test
2. **GREEN**: Write minimal code to pass
3. **REFACTOR**: Improve code quality

### Example

```python
# 1. Write test first
def test_barometer_load():
    """Test barometer data loading."""
    baro = Barometer(file=Path("baro.csv"))
    assert baro.data.shape[0] > 0

# 2. Run test - it fails (RED)
# pytest tests/test_barometer.py -v

# 3. Write implementation
class Barometer:
    def __init__(self, file: Path):
        self.file = file
        self._data = None
    
    @property
    def data(self) -> pl.DataFrame:
        if self._data is None:
            self._data = pl.read_csv(self.file)
        return self._data

# 4. Run test - it passes (GREEN)
# 5. Refactor if needed
```

---

## Adding New Features

### New Sensor

See [Adding Sensors](adding-sensors.md) guide.

### New Loader

1. Create `pils/loader/new_loader.py`
2. Implement `load_single_flight()` method
3. Add tests in `tests/test_new_loader.py`
4. Update documentation

### New Analysis Module

1. Create `pils/analyze/new_analysis.py`
2. Follow existing patterns (PPK, RTK)
3. Add comprehensive tests
4. Document API and usage

---

## Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted (black, isort)
- [ ] Linting passes (flake8)
- [ ] Type hints added
- [ ] All tests pass

## Related Issues
Fixes #123
```

---

## Code Review

### What We Look For

- [ ] Tests cover new functionality
- [ ] Code follows style guide
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] No breaking changes (or documented)
- [ ] Performance considerations

### Review Etiquette

**For Reviewers:**

- Be constructive and specific
- Explain the "why" behind suggestions
- Approve when ready, don't nitpick

**For Authors:**

- Respond to all comments
- Ask clarifying questions
- Don't take feedback personally

---

## Issue Reporting

### Bug Reports

Include:

1. **Environment**: Python version, OS, PILS version
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What should happen
4. **Actual behavior**: What happens
5. **Error messages**: Full traceback

### Feature Requests

Include:

1. **Use case**: Why is this needed?
2. **Proposed solution**: How might it work?
3. **Alternatives**: Other approaches considered

---

## Documentation

### When to Update Docs

- New features → Add to relevant section
- API changes → Update API reference
- Bug fixes → Update if behavior changed
- Examples → Add practical examples

### Doc Standards

- Use MkDocs Material syntax
- Include code examples
- Cross-reference related pages
- Keep examples tested and working

---

## Release Process

### Versioning

PILS uses [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

1.0.0 → 1.0.1  # Patch: Bug fixes
1.0.1 → 1.1.0  # Minor: New features (backward compatible)
1.1.0 → 2.0.0  # Major: Breaking changes
```

### Release Checklist

- [ ] All tests pass
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Documentation current
- [ ] Tag created

---

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/polocalc/pils/discussions)
- **Bugs**: Open an [Issue](https://github.com/polocalc/pils/issues)
- **Security**: Email security@polocalc.com

---

## See Also

- [Testing](testing.md) - Testing practices
- [Code Style](code-style.md) - Style guide
- [Adding Sensors](adding-sensors.md) - Extend PILS
