# Skills Index

Skills are reusable knowledge contracts with templates, patterns, and completion criteria. Commands reference these skills for consistent implementation.

## Project Configuration

For commands, paths, and environment: `.claude/config/project.md`

## How to Use

1. **Read rules first** — `.claude/rules/` applies to everything
2. **Scan this index** — find relevant skills for your task
3. **Load only needed skills** — minimize context
4. **Follow skill contracts** — inputs, outputs, DoD
5. **Create skills for gaps** — if behavior isn't covered, write a skill
6. **Update this index** — keep the registry current

## Skills by Category

### Workflow Skills

| Skill | Description | Contract |
|-------|-------------|----------|
| Onboarding | Quick start guide for new sessions | `skills/onboarding.md` |
| New Branch | Create feature/fix branch | `skills/new-branch.md` |
| Refactor | Structural improvements without behavior change | `skills/refactor.md` |
| Commit & Push | Commit staged changes and push to origin | `skills/commit-push.md` |
| Finalize Changes | Pre-commit validation and message suggestion | `skills/finalize-changes.md` |

### Validation Skills

| Skill | Description | Contract |
|-------|-------------|----------|
| Review Changes | Two-phase code review (defects then design) | `skills/review-changes.md` |
| Triage Review | Critically assess external reviews against codebase | `skills/triage-review.md` |
| Improve Prompt | Analyze and improve agent prompt files | `skills/improve-prompt.md` |

### Reference Skills

| Skill | Description | Contract |
|-------|-------------|----------|
| Custom Buttons | Choice buttons in prompt responses | `skills/custom-buttons.md` |

## Commands Reference

Commands are thin executable wrappers that invoke skills:

| Command | Invokes Skill | Purpose |
|---------|---------------|---------|
| `/onboarding` | `skills/onboarding.md` | Quick start for new sessions |
| `/new-branch` | `skills/new-branch.md` | Create feature/fix branch |
| `/refactor` | `skills/refactor.md` | Refactor code without behavior change |
| `/review-changes` | `skills/review-changes.md` | Review code changes |
| `/triage-review` | `skills/triage-review.md` | Triage external code review |
| `/check-standards` | (rules-based) | Validate code against standards |
| `/audit-config` | (workflow) | Audit config completeness and consistency |
| `/improve-prompt` | `skills/improve-prompt.md` | Improve agent prompt file |
| `/finalize-changes` | `skills/finalize-changes.md` | Lint, test, commit prep |
| `/commit-push` | `skills/commit-push.md` | Commit and push to origin |

## Agents

| Agent | Description | Contract |
|-------|-------------|----------|
| Code Reviewer | Reviews Python code for standards and correctness | `.claude/agents/code-reviewer.md` |

## Naming Conventions

- **new-*** — Creates new resources
- **check-*** — Validates against rules
- **audit-*** — Generates analysis reports
- **review-*** — Reviews code or reviews
- **finalize-*** — Completes workflows
