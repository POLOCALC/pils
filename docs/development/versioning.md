# Auto-Versioning with CalVer

This project uses **Calendar Versioning (CalVer)**. Version bumping is handled automatically by GitHub Actions when Pull Requests are merged to `main`.

## Version Format

**YYYY.MM.INC0** (e.g., `2026.02.0`, `2026.02.1`, `2026.03.0`)

- **YYYY**: Year (4 digits)
- **MM**: Month (Zero-padded, e.g., 01, 02... 12)
- **INC0**: Incrementing number (Resets to 0 automatically when the month changes)

## How It Works

1. Developers create feature branches and make commits.
2. When a PR is merged to `main`, GitHub Actions automatically:
   - **Calculates the new version:**
     - If the date has changed since the last release (new month/year), it updates the date and resets the counter to `.0`.
     - If the date is the same, it increments the counter (e.g., `.0` -> `.1`).
   - **Updates Files:** `pyproject.toml` and `pils/__init__.py`.
   - **Git Operations:** Creates a commit and a git tag.
   - **Push:** Pushes the new version commit and tag back to main.

## Setup

No setup is needed for developers. Versioning is calendar-based and automated.

## Usage

### Regular Commits

Since this project uses CalVer, version bumps are time-based. You do not need to format your commit messages specifically to trigger a version bump (unlike Semantic Versioning).

```bash
# Any commit message works
git commit -m "Add PPK analysis cleanup on failure"
git commit -m "Fix GPS timestamp parsing"

```

### Push and Create PR

```bash
git push origin feature-branch
```

Create a PR and merge to `main`. The GitHub Action will detect the merge and run the version bump.

---

## Manual Versioning (Local Release)

**Note:** You generally do not need to do this. The CI/CD pipeline handles releases.
However, if GitHub Actions is broken or you need to force a release locally, follow these steps.

### Prerequisites

1. Ensure your local `main` branch is up to date.
2. Ensure you have no uncommitted changes (`git status` must be clean).
3. Install the tool:

```bash
pip install bumpver
```



### 1. Standard Manual Release (Recommended)

This mimics the CI process. It checks today's date against the current version and increments automatically.

```bash
# 1. Switch to main and pull latest
git checkout main
git pull origin main

# 2. Run the update (Updates files, commits, and tags)
bumpver update

# 3. Push the commit and the tag to GitHub
git push --follow-tags
```

### 2. Force a Specific Version

Use this if you need to override the date logic (e.g., fixing a mistake or backdating).

```bash
# Force version to 2026.05.01 regardless of today's date
bumpver update --set-version "2026.05.01"

# Push changes
git push --follow-tags
```

### 3. Dry Run (Test)

Check what the version *would* be without changing any files.

```bash
bumpver update --dry
```

---

## Check Current Version

```bash
# From Python
python -c "import pils; print(pils.__version__)"

# From pyproject.toml
grep '^version = ' pyproject.toml
```

## Skip Auto-Versioning

To merge to `main` without triggering a version bump (e.g., for CI config changes or readme updates), include `[skip-version]` in your **Merge Commit** message or the PR title.

```bash
# Example commit message
git commit -m "chore: update CI config [skip-version]"
```

## Changelog

*Note: `bumpver` updates version numbers but does not generate text changelogs.*

The `CHANGELOG.md` is updated via a separate GitHub Action (e.g., Release Drafter or Git Cliff) which groups commits from the PR.

## Troubleshooting

### Version not bumping after PR merge

Check the GitHub Actions logs. Common causes:

1. **Protected Branch:** The Action failed to push because it lacks permission to bypass branch protection on `main`.
2. **Skip Flag:** The commit message contained `[skip-version]`.
3. **Dirty Directory:** The runner had uncommitted changes (rare in CI).

### Version out of sync

If `pyproject.toml` and `pils/__init__.py` have different versions, manually fix them to match and push:

```bash
# 1. Edit both files to have the same version string
# 2. Commit the fix
git commit -m "chore: sync version numbers"
git push
```