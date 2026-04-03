# ytx - YouTube Transcript Extractor

## Prime Directive

Before making any code changes:
1. Read `.claude/rules/coding-standards.md` for Python conventions
2. Follow `.claude/rules/commits.md` for commit messages
3. Follow `.claude/rules/workflow.md` for development process
4. Check `.claude/rules/skill-routing.md` to load relevant skills
5. Run `/finalize-changes` before committing

## Shared Rules

All rules in `.claude/rules/` are non-negotiable:

| Rule | Scope |
|------|-------|
| `coding-standards.md` | Python conventions, type hints, imports |
| `commits.md` | Commit message format and types |
| `workflow.md` | Skill-driven development process |
| `testing.md` | pytest conventions and mocking strategy |
| `skill-routing.md` | Auto-load skills by file pattern or topic |

## Intent

ytx extracts YouTube video transcripts via API and outputs clean markdown. No browser automation needed - just pass a URL, get formatted text in seconds.

## Quick Reference

See `README.md` for full usage, flags, output formats, and configuration.

```bash
python ytx.py "<url>" -o output.md
python ytx.py VIDEO_ID -O --push    # AI-optimized + push to PromptManager
```

## Dependencies

- `youtube-transcript-api` - transcript fetching (required)
- `yt-dlp` - metadata + description (optional, falls back to oEmbed)

## Testing

```bash
python -m pytest tests/
```
