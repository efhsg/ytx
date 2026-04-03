---
name: commit-push
description: Commit staged changes and push to origin with conventional commit messages.
area: meta
provides:
  - git_commit
  - git_push
depends_on:
  - .claude/rules/commits.md
---

# CommitPush

Commit staged or unstaged changes and push to the remote origin.

## When to use

- User asks to commit and push changes
- After finalizing changes and review pass
- Quick ship of a single change

## Inputs

- `$ARGUMENTS`: Optional commit message. If provided, use it as-is (must still follow commit format).

## Outputs

- Confirmation with commit hash and push status
- Error details if commit or push fails

## Algorithm

### 1. Check for changes

```bash
git diff --staged --stat
git diff --stat
```

- If **staged changes exist** → proceed to step 2
- If **no staged but unstaged changes exist** → stage them with `git add -A`, then proceed
- If **no changes at all** → report "Nothing to commit" and stop

### 2. Determine commit message

Follow this order:

1. **If `$ARGUMENTS` is provided** → use it as the commit message
2. **If a commit message was previously suggested in this conversation** → use that message
3. **Otherwise** → generate a commit message:
   - Read `.claude/rules/commits.md` for format rules
   - Run `git diff --staged` to understand the changes
   - Format: `TYPE(scope): description`
   - Types: feat, fix, refactor, docs, chore
   - Scopes: cli, output, api (optional)

### 3. Commit

Use HEREDOC for proper formatting:

```bash
git commit -m "$(cat <<'EOF'
TYPE(scope): description
EOF
)"
```
### 4. Push

```bash
git push origin HEAD
```

**If push fails:**
- If rejected due to remote changes → run `git pull --rebase origin HEAD` then retry push
- If other error → report the error and stop

### 5. Report

Report success with commit hash.

## Severity levels

| Level | Criteria | Action |
|-------|----------|--------|
| **Critical** | Push to wrong branch, force push, commit secrets | Must prevent |
| **High** | Wrong commit format, missing scope where needed | Should fix before push |
| **Medium** | Suboptimal commit message wording | Recommend improvement |
| **Low** | Minor style preferences | Note for next time |

## Definition of Done

- Changes are committed with a message following `.claude/rules/commits.md`
- Commit is pushed to origin successfully
- User sees commit hash and confirmation
