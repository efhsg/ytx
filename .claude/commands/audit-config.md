---
allowed-tools: Read, Grep, Glob, Bash(ls:*), Bash(find:*)
description: Audit all Claude configuration files for completeness, correctness, and consistency
---

# Configuration Audit

Audit all Claude Code configuration files against the codebase.

## Files to Audit

### Main Configuration
@CLAUDE.md

### Project Configuration
@.claude/config/project.md

### Rules
@.claude/rules/coding-standards.md
@.claude/rules/commits.md
@.claude/rules/workflow.md
@.claude/rules/testing.md
@.claude/rules/skill-routing.md

### Skills Index
@.claude/skills/index.md

### Other AI Configurations
@AGENTS.md

## Commands List
!`ls -1 .claude/commands/*.md 2>/dev/null | xargs -I {} basename {}`

## Skills List
!`ls -1 .claude/skills/*.md 2>/dev/null | xargs -I {} basename {}`

## Codebase Context

### Functions in ytx.py
!`grep -n "^def " ytx.py`

### Test files
!`ls -1 tests/ 2>/dev/null`

---

## Audit Instructions

Perform a comprehensive audit checking:

### 1. Completeness
- Are all public functions in ytx.py reflected in onboarding.md?
- Are all commands documented in skills/index.md?
- Are all skills documented in skills/index.md?
- Are all CLI flags documented in config/project.md?
- Missing patterns or conventions that exist in code but not in rules?

### 2. Correctness
- Do documented paths match actual file locations?
- Are command examples correct and working?
- Do test commands work as documented?
- Are deprecated patterns still documented?

### 3. Duplicates & Conflicts
- Same rule defined in multiple places with different wording?
- Contradictory instructions between CLAUDE.md and rules files?
- Overlapping responsibilities between skills?
- Commands in CLAUDE.md that don't match skills/index.md?

### 4. AGENTS.md Alignment
- Must reference CLAUDE.md as single source of truth
- Quick reference commands must match CLAUDE.md
- No additional rules that contradict CLAUDE.md

## Output Format

```markdown
# Configuration Audit Report

## Summary
- Total issues found: X
- Critical: X | Warning: X | Info: X

## Completeness Issues
- [ ] Issue description — file:location

## Correctness Issues
- [ ] Issue description — file:location

## Duplicates & Conflicts
- [ ] Issue description — files involved

## AGENTS.md Issues
- [ ] Issue description

## Recommendations
1. Priority fix: ...
2. ...
```

After the report, ask the user which issues to fix.
