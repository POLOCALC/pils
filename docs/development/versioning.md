# Auto-Versioning with CalVer

This project uses **Calendar Versioning (CalVer)** with automatic version bumping via GitHub Actions when PRs are merged to main.

## Version Format

**YYYY.MM.PATCH** (e.g., `2026.2.0`, `2026.2.1`, `2026.3.0`)

- **YYYY**: Year (4 digits)
- **MM**: Month (no leading zero)
- **PATCH**: Incrementing number within the month

## How It Works

1. Developers create feature branches and make commits
2. When PR is merged to `main`, GitHub Actions automatically:
   - Checks if month has changed (bumps to YYYY.MM.0)
   - Otherwise increments patch version (YYYY.MM.PATCH+1)
   - Updates `pyproject.toml` and `pils/__init__.py`
   - Creates git tag
   - Pushes version commit and tag back to main

No local setup required - versioning happens automatically in CI/CD.

## Setup

No setup needed for developers. Versioning is calendar-based and automatic.

## Usage

### Regular Commits

Since this project uses CalVer (calendar versioning), you don't need to follow any specific commit message format. Version bumps are calendar-based, not commit-based.

```bash
# Any commit message works
git commit -m "Add PPK analysis cleanup on failure"
git commit -m "Fix GPS timestamp parsing"
git commit -m "Update README with examples"
```

### Push and Create PR

```bash
git push origin feature-branch
```

Create a PR and merge to main. GitHub Actions automatically handles versioning based on the current date.

## Manual Version Bump

If maintainers need to manually bump the version (requires bumpver installed locally):

```bash
# Install bumpver
pip install bumpver

# Bump patch version (2026.2.0 -> 2026.2.1)
bumpver update --patch

# Bump to new month (2026.2.1 -> 2026.3.0)
bumpver update --minor

# Bump to new year (2026.2.1 -> 2027.0.0)
bumpver update --major

# Dry-run to see what would happen
bumpver update --patch --dry
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
