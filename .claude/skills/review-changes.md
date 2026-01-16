---
name: review-changes
description: Review code changes for correctness, style, and project compliance.
area: meta
provides:
  - code_review
  - compliance_check
depends_on:
  - .claude/rules/coding-standards.md
---

# ReviewChanges

Perform a structured code review of staged or unstaged changes against project rules and best practices.

## Persona

- Senior Python engineer with experience in CLI tools and API integrations.
- Write concise, type-hinted code following PEP 8 conventions.
- Focus on simplicity and avoiding over-engineering.

## When to use

- User asks to review changes, code, or a PR
- Before finalizing changes for commit
- After implementing a feature to catch issues early

## Inputs

- `scope`: Optional file path, directory, or description to narrow review
- `staged`: If true, review only staged changes (default: review all changes)

## Outputs

- Summary with file count and overall status
- Findings categorized by severity
- Actionable recommendations

## Review Phases

### Phase 1: Defect Detection

Run this phase first. Fix all Critical/High/Medium issues before proceeding to Phase 2.

#### 1. Correctness

- Logic is correct and handles edge cases
- No obvious bugs or regressions
- Error handling is appropriate (no silent failures, no bare `except:`)

#### 2. Style & Standards

Per `.claude/rules/coding-standards.md`:

- Type hints on all function signatures
- Docstrings for public functions
- f-strings for formatting (no `.format()` or `%`)
- Specific exception handling (no bare `except:`)
- Errors printed to stderr
- Import ordering: stdlib → third-party → local

#### 3. CLI Conventions

- Uses argparse correctly
- Exit codes: 0 success, 1 error
- Output to stdout, errors to stderr

#### 4. Security

- No command injection vulnerabilities
- No hardcoded credentials or API keys
- Input validation where needed

### Phase 2: Design Refinement

Run only after Phase 1 has no Critical/High/Medium findings. Apply judiciously — stop when goal is met.

#### DRY (Don't Repeat Yourself)

- Is there duplicated logic that should be extracted?
- Are there repeated patterns that warrant a shared function?

#### YAGNI (You Aren't Gonna Need It)

- Is there speculative code that isn't currently used?
- Is there over-engineering for hypothetical future requirements?

#### Code Conventions

- Consistent naming patterns
- Consistent parameter ordering in similar functions
- Prefer early returns over deep nesting

## Algorithm

### Phase 1 Algorithm

1. Run `git status --porcelain` to identify changed files
2. Run `git diff` (or `git diff --staged`) to see specific changes
3. Read each changed file to understand full context
4. Load project rules from `.claude/rules/`
5. Evaluate each file against Phase 1 checklist
6. Categorize findings by severity (Critical/High/Medium/Low)
7. Report findings — stop here if Critical/High/Medium issues exist

### Phase 2 Algorithm

Only run after Phase 1 issues are resolved:

1. Re-read changed files with design lens
2. Evaluate against DRY/YAGNI principles
3. Check code conventions for consistency
4. Report refinement suggestions as Low severity
5. Compile final report with all recommendations

## Severity levels

| Level | Criteria | Action |
|-------|----------|--------|
| **Critical** | Security vulnerabilities, data corruption risks, breaking changes | Must fix before merge |
| **High** | Bugs, incorrect logic, missing error handling, bare except clauses | Should fix before merge |
| **Medium** | Missing type hints, code style violations, suboptimal patterns | Recommended to fix |
| **Low** | Minor improvements, documentation gaps, naming suggestions | Consider fixing |

## Output format

```
## Review Summary

**Files reviewed:** N files
**Phase:** 1 (Defect Detection) | 2 (Design Refinement)
**Status:** PASS | PASS WITH COMMENTS | NEEDS CHANGES

## Phase 1 Findings

### Critical
- (none or list with file:line references)

### High
- `path/to/file:123` — description of issue

### Medium
- `path/to/file:45` — description of issue

### Low
- `path/to/file:67` — suggestion

## Phase 2 Findings (if applicable)

### Refinement
- `path/to/file:89` — DRY/YAGNI suggestion

## Recommendations

1. Specific action to take
2. ...
```

**Note:** If Phase 1 has Critical/High/Medium findings, do not proceed to Phase 2. Report Phase 1 findings and recommend fixes first.

## Definition of Done

- All changed files reviewed against Phase 1 checklist
- Findings reference specific file and line numbers
- Each finding has clear, actionable description
- Overall status reflects severity of findings
- Phase 2 only runs when Phase 1 has no Critical/High/Medium issues
- Output follows the required format
