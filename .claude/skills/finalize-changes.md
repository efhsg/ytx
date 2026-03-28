---
name: finalize-changes
description: Pre-commit validation and commit message suggestion.
area: meta
provides:
  - pre_commit_validation
  - commit_message_suggestion
depends_on:
  - .claude/rules/coding-standards.md
  - .claude/rules/commits.md
---

# FinalizeChanges

Pre-commit validation workflow. Run before committing to catch issues early and generate a standards-compliant commit message.

## Persona

- Senior Python engineer with experience in CLI tools and API integrations.
- Write concise, type-hinted code following PEP 8 conventions.
- Focus on simplicity and avoiding over-engineering.

## When to use

- Before committing changes
- After implementing a feature to validate before review
- When user runs `/finalize-changes`

## Inputs

- `scope`: Optional file path or directory to narrow validation (default: all changed files)

## Outputs

- Validation report with pass/fail per check
- List of issues found, categorized by severity
- Suggested commit message following project conventions

## Validation Phases

### Phase 1: Syntax & Lint

Run automated checks to catch errors early.

#### 1. Syntax check

```bash
python -m py_compile ytx.py
```

#### 2. Lint (if available)

```bash
# Try ruff first, fall back to flake8
ruff check ytx.py 2>/dev/null || flake8 ytx.py 2>/dev/null || echo "No linter installed"
```

### Phase 2: Coding Standards

Per `.claude/rules/coding-standards.md`:

- Type hints on all function signatures
- Docstrings for public functions
- f-strings for formatting (no `.format()` or `%`)
- Specific exception handling (no bare `except:`)
- Errors printed to stderr
- Import ordering: stdlib → third-party → local

### Phase 3: Commit Message

Per `.claude/rules/commits.md`:

- Analyze staged changes with `git diff --staged`
- Determine appropriate `TYPE` (feat, fix, refactor, docs, chore)
- Determine optional `scope` (cli, output, api)
- Propose message: `TYPE(scope): description`
- Imperative mood, lowercase, no period, under 72 characters
- No `Co-Authored-By` trailer

## Algorithm

1. Run `git status --porcelain` to identify changed files
2. Run `git diff --staged` to see staged changes (fall back to `git diff` if nothing staged)
3. Run syntax check on each changed Python file
4. Run linter on each changed Python file
5. Read each changed file to check against coding standards
6. Load project rules from `.claude/rules/`
7. Evaluate each file against Phase 2 checklist
8. Categorize findings by severity (Critical/High/Medium/Low)
9. If no Critical/High issues, generate commit message suggestion
10. Compile final report

## Severity levels

| Level | Criteria | Action |
|-------|----------|--------|
| **Critical** | Syntax errors, security vulnerabilities, breaking changes | Must fix before commit |
| **High** | Bugs, incorrect logic, bare except clauses | Should fix before commit |
| **Medium** | Missing type hints, code style violations | Recommended to fix |
| **Low** | Minor improvements, naming suggestions | Consider fixing |

## Output format

```
## Finalize Summary

**Files checked:** N files
**Status:** READY TO COMMIT | NEEDS CHANGES

## Validation

✓ Syntax check passed
✓ No linter issues (ruff)
✓ Coding standards: OK

## Issues (if any)

### Critical
- (none or list with file:line references)

### High
- `path/to/file:123` — description of issue

### Medium
- `path/to/file:45` — description of issue

### Low
- `path/to/file:67` — suggestion

## Suggested commit

  TYPE(scope): description
```

**Note:** If Critical/High issues exist, status is NEEDS CHANGES and no commit message is suggested until issues are resolved.

## Definition of Done

- All changed Python files pass syntax check
- Linter runs without Critical/High issues
- Changed code reviewed against coding standards
- Findings reference specific file and line numbers
- Each finding has clear, actionable description
- Commit message follows `.claude/rules/commits.md` conventions
- Output follows the required format
