# Commit Conventions

## Format

```
TYPE(scope): description
```

## Types

| Type | Use for |
|------|---------|
| `feat` | New functionality |
| `fix` | Bug fixes |
| `refactor` | Code restructuring (no behavior change) |
| `docs` | Documentation only |
| `chore` | Maintenance (deps, config) |

## Scope

Optional. Use for clarity when helpful:
- `cli` - argument parsing, flags
- `output` - formatting, markdown/json generation
- `api` - external service integration

## Rules

- Use imperative mood: "add feature" not "added feature"
- Lowercase description
- No period at end
- Keep under 72 characters
- No `Co-Authored-By` trailer

## Examples

```
feat(cli): add --push flag for PromptManager integration
fix: handle missing transcript gracefully
refactor(output): extract markdown formatting to function
docs: update usage examples in README
chore: bump youtube-transcript-api to 0.6.3
```
