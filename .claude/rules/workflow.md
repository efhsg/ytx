# Development Workflow

## Skill-Driven Development

For every task:

1. Read project rules first — `.claude/rules/` applies to everything
2. Check `.claude/skills/index.md` — find relevant skills for your task
3. Load only needed skills — minimize context
4. Follow DoD — skill is done when Definition of Done passes
5. Create skills for gaps — if behavior isn't covered, write a skill
6. Update index — keep skill registry in `skills/index.md` current

## Before Coding

1. Read project rules in `.claude/rules/`
2. Check `.claude/skills/index.md` for relevant skills
3. Understand existing patterns in `ytx.py`
4. Plan changes before implementing

## Design Directory per Feature

For complex features, create a design directory:

```
.claude/design/[feature-name]/
├── spec.md         # Functional specification
├── plan.md         # Technical plan
└── insights.md     # Decisions, edge cases
```

### Implementation Memory

During longer implementations, track progress:

```
.claude/design/[feature-name]/impl/
├── context.md      # Goal, scope, key references
├── todos.md        # Steps checklist
└── insights.md     # Findings, deviations
```

### Working with todos.md — MANDATORY

When a `todos.md` exists for the current task, it is **the** authoritative checklist:

1. **Session start / context resume:** Read `todos.md` first. Resume from the first open task.
2. **Work in order:** Complete tasks sequentially as listed.
3. **Mark done immediately:** Update checkbox to `[x]` before starting next task.
4. **Never work without updating:** An unchecked completed task is a bug.
5. **Context compression recovery:** `todos.md` is your recovery point.

## Code Review

Use `/review-changes` for structured code review. See `skills/review-changes.md`.
