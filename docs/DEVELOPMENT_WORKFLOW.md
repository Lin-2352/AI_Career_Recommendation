# Development Workflow

## Branching

`main` is reserved for stable release commits. Future changes should start from the current working branch and use professional branch names:

- `feature/profile-insights`
- `feature/model-evaluation-report`
- `fix/artifact-loading`
- `docs/deployment-guide`

When a feature branch needs another follow-up change before merging, create the next branch from that feature branch:

```powershell
git checkout feature/profile-insights
git checkout -b fix/profile-insights-validation
```

## Commit Standard

Use short, outcome-focused commit messages:

- `Release v1.0.0 career recommendation app`
- `Add model evaluation report`
- `Fix Streamlit artifact loading`

## Repository Hygiene

Before every commit:

```powershell
git status --short
git diff --stat
```

Do not commit virtual environments, IDE folders, local secrets, Kaggle credentials, cache folders, logs, or large temporary outputs.

Raw files containing names, emails, or other direct identifiers must be sanitized before they are used in committed training data.

## Code Style

Production code should be readable through naming and structure. Comments should be reserved for non-obvious checkpoints or operational warnings.

## Verification

Use these commands before pushing:

```powershell
python scripts/02_train_model.py
python -m compileall app.py backend frontend scripts tests
python -m unittest discover -s tests -v
streamlit run app.py
```
