---
allowed-tools: Bash(git diff:*), Bash(git status:*), Read, Grep
description: Check if staged changes follow project code standards
---

## Staged files

!`git diff --cached --name-only`

## Task

Read each staged Python file and check against `.claude/rules/coding-standards.md`:

| Check | Rule |
|-------|------|
| Type hints | All function signatures must have type hints |
| Docstrings | Public functions need docstrings |
| f-strings | No `.format()` or `%` formatting |
| Exception handling | No bare `except:` — catch specific exceptions |
| Error output | Errors printed to stderr |
| Import ordering | stdlib → third-party → local (alphabetical within groups) |
| pathlib | Prefer `pathlib.Path` for file paths |

## Output

Per file: Filename + Issues found (or "OK")

Summary: total issues, ready to commit?
