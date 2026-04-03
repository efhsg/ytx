---
name: new-branch
description: Create a feature or fix branch with consistent naming
area: workflow
provides:
  - branch_creation
depends_on: []
---

# New Branch

Create a properly named branch for a new feature or fix.

## Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feat/short-description` | `feat/json-output` |
| Fix | `fix/short-description` | `fix/missing-transcript-handling` |
| Refactor | `refactor/short-description` | `refactor/extract-optimization` |
| Docs | `docs/short-description` | `docs/update-readme` |

## Algorithm

1. **Ask the user** what the branch is for (if not provided via $ARGUMENTS)
2. **Check working tree** — `git status` must be clean (no uncommitted changes)
3. **Determine type** — feat/fix/refactor/docs based on description
4. **Create branch** — `git checkout -b {type}/{description}` from main
5. **Confirm** — show branch name and status

## Rules

- Branch from `main` unless user specifies otherwise
- Use lowercase, hyphens for spaces
- Keep description under 5 words
- If working tree is dirty, ask user to commit or stash first

## Definition of Done

- [ ] Working tree was clean before branching
- [ ] Branch created from correct base
- [ ] Branch name follows naming convention
- [ ] User confirmed the branch name
