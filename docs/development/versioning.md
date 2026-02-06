# Auto-Versioning with CalVer

This project uses **Calendar Versioning (CalVer)** with automatic version bumping via GitHub Actions when PRs are merged to main.

## Version Format

**YYYY.MM.MICRO** (e.g., `2026.02.1`, `2026.02.2`, `2026.03.1`)

- **YYYY**: Year (4 digits)
- **MM**: Month (1-2 digits, no leading zero)
- **MICRO**: Incrementing number within the month

## How It Works

1. Developers create feature branches and make commits using conventional commit messages
2. When PR is merged to `main`, GitHub Actions automatically:
   - Analyzes commits since last version
   - Bumps version if `feat:` or `fix:` commits found
   - Updates `pyproject.toml` and `pils/__init__.py`
   - Creates git tag
   - Updates CHANGELOG.md
   - Pushes version commit and tag back to main

No local setup required - versioning happens automatically in CI/CD.

## Setup

No setup needed for developers. Just use conventional commit messages.

## Usage

### Commit with Conventional Commit Messages

Use these prefixes for your commits:

| Prefix | Description | Version Bump |
|--------|-------------|--------------|
| `feat:` | New feature | MINOR (month or micro) |
| `fix:` | Bug fix | PATCH (micro) |
| `docs:` | Documentation only | None |
| `style:` | Code style (formatting) | None |
| `refactor:` | Code refactoring | PATCH (micro) |
| `perf:` | Performance improvement | PATCH (micro) |
| `test:` | Adding tests | None |
| `chore:` | Maintenance tasks | None |

### Examples

```bash
# Feature (bumps version)
git commit -m "feat: add PPK analysis cleanup on failure"

# Bug fix (bumps version)
git commit -m "fix: correct GPS timestamp parsing"

# Documentation (no version bump)
git commit -m "docs: update README with new examples"

# Refactoring (bumps version)
git commit -m "refactor: simplify flight data loading"
```

### Push and Create PR

```bash
git push origin feature-branch
```

Create a PR and merge to main. GitHub Actions automatically handles versioning.

## Manual Version Bump

If maintainers need to manually bump the version (requires commitizen installed locally):

```bash
# Install commitizen
pip install commitizen

# Auto-detect bump type from commits
cz bump

# Manually specify bump type
cz bump --increment PATCH
cz bump --increment MINOR
cz bump --increment MAJOR

# Dry-run to see what would happen
cz bump --dry-run
```

## Check Current Version

```bash
# From Python
python -c "import pils; print(pils.__version__)"

# From pyproject.toml
grep '^version = ' pyproject.toml
```

## Skip Auto-Versioning

To merge to main without triggering version bump, include `[skip-version]` in the merge commit:

```bash
git commit -m "chore: update CI config [skip-version]"
```

## Changelog

The CHANGELOG.md is automatically updated with version bumps. It includes:
- Version number and date
- Grouped changes by type (feat, fix, etc.)
- Commit messages

## Troubleshooting

### "No version bump needed"
This means no `feat:` or `fix:` commits were found since the last version. Only documentation or style changes don't trigger version bumps.

### Version not bumping after PR merge
Check the GitHub Actions logs in the Actions tab. Common issues:
- No `feat:` or `fix:` commits in the PR
- Commit message contains `[skip-version]`
- GitHub Actions workflow failed

### Version out of sync
If `pyproject.toml` and `pils/__init__.py` have different versions, manually fix and push:
```bash
# Edit both files to match
git commit -m "fix: sync version numbers"
git push
```

## Version History

View all versions:
```bash
git tag -l "v*"
```

View changelog:
```bash
cat CHANGELOG.md
```
