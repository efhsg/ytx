# ytx - YouTube Transcript Extractor

> Canonical instructions live in `CLAUDE.md`. This file mirrors those for Codex/OpenAI agents.

## Prime Directive

Before making any code changes:
1. Follow the coding standards below
2. Follow the commit conventions below
3. Validate changes before committing (syntax check, lint, standards review)

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

## Coding Standards

- Type hints on all function signatures
- Docstrings for public functions (one-liner for simple functions)
- No bare `except:` clauses - always catch specific exceptions
- Prefer f-strings for formatting (no `.format()` or `%`)
- Prefer `pathlib.Path` for file paths when practical
- Import order: stdlib → third-party → local (alphabetical within groups)
- Exit codes: 0 success, 1 error
- Errors to stderr, output to stdout (or file with `-o`)

## Commit Conventions

Format: `TYPE(scope): description`

Types: `feat`, `fix`, `refactor`, `docs`, `chore`

Scopes (optional): `cli`, `output`, `api`

Rules:
- Imperative mood, lowercase, no period, under 72 characters
- No `Co-Authored-By` trailer
