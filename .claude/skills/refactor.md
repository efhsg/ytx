---
name: refactor
description: Structural improvements to ytx.py without behavior change
area: workflow
provides:
  - refactoring_patterns
depends_on: []
---

# Refactor

Apply structural improvements without changing external behavior.

## Non-Negotiables

- No behavior change — output must remain identical for all inputs
- Apply SOLID/DRY only when it eliminates real duplication; stop when goal is met
- Smallest diff that solves the problem
- Run tests before AND after refactoring
- Follow `rules/coding-standards.md` for style

## Algorithm

1. **Read target code** — understand current structure and intent
2. **Identify opportunities** — duplication, long functions, unclear names
3. **Run test baseline** — `python -m pytest tests/` must pass
4. **Apply changes** — one pattern at a time, smallest diff
5. **Lint** — `ruff check ytx.py` or `python -m py_compile ytx.py`
6. **Re-run tests** — must still pass with identical results
7. **Review diff** — ensure no behavior change leaked in

## Refactoring Patterns

| Pattern | When to apply |
|---------|--------------|
| Extract function | Function > 40 lines or does multiple things |
| Early return | Nested conditionals > 2 levels deep |
| Reduce duplication | 3+ identical code blocks |
| Rename for clarity | Name doesn't describe what it does |
| Simplify conditionals | Complex boolean expressions |

## Definition of Done

- [ ] Tests pass before and after (identical results)
- [ ] No behavior change in CLI output
- [ ] Lint passes
- [ ] Diff is minimal — no unrelated changes
- [ ] Type hints preserved or added per coding standards
