# /finalize-changes

Pre-commit validation workflow.

## Steps

1. **Check coding standards compliance**
   - Review changed code against `.claude/rules/coding-standards.md`
   - Flag any violations (missing type hints, bare except, etc.)

2. **Syntax check**
   ```bash
   python -m py_compile ytx.py
   ```

3. **Lint (if available)**
   ```bash
   # Try ruff first, fall back to flake8
   ruff check ytx.py 2>/dev/null || flake8 ytx.py 2>/dev/null || echo "No linter installed"
   ```

4. **Suggest commit message**
   - Analyze staged changes
   - Propose message following `.claude/rules/commits.md` format
   - Show: `TYPE(scope): description`

## Output

Report any issues found, then suggest the commit message.

## Example

```
✓ Syntax check passed
✓ No linter issues (ruff)
✓ Coding standards: OK

Suggested commit:
  feat(cli): add --push flag for PromptManager integration
```
