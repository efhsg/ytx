---
name: onboarding
description: Quick start guide for new ytx development sessions
area: workflow
provides:
  - project_overview
depends_on: []
---

# Onboarding

Quick orientation for a new development session on ytx.

## Project Overview

**ytx** is a single-file Python CLI that extracts YouTube video transcripts and outputs clean markdown. No browser automation — uses `youtube-transcript-api` directly.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Entry point | `ytx.py` (~600 lines) |
| Transcript API | `youtube-transcript-api` |
| Metadata | `yt-dlp` (optional, falls back to oEmbed) |
| Tests | pytest (`tests/`) |
| Config | `.ytx.json` / env vars |

## Key Functions in ytx.py

| Function | Purpose |
|----------|---------|
| `extract_video_id()` | Parse YouTube URL to video ID |
| `fetch_video_metadata()` | Get title, channel, description (yt-dlp then oEmbed fallback) |
| `get_transcript()` | Fetch transcript via API (manual then auto, language fallback) |
| `clean_transcript()` | Remove fillers, normalize whitespace |
| `extract_sections()` | Split transcript into logical sections |
| `extract_tools()` | Detect tool/product names mentioned |
| `format_markdown()` | Standard markdown output |
| `format_optimized_markdown()` | AI-optimized output with TOC, tools, steps |
| `push_to_promptmanager()` | POST transcript to PromptManager API |
| `main()` | CLI entry point (argparse) |

## Available Commands

| Command | Purpose |
|---------|---------|
| `/onboarding` | This guide |
| `/new-branch` | Create feature/fix branch |
| `/refactor` | Structural improvements |
| `/review-changes` | Code review |
| `/finalize-changes` | Lint + test + commit message |
| `/commit-push` | Commit and push |

## Quick Commands

```bash
python ytx.py "<url>"                    # basic transcript
python ytx.py "<url>" -O --push          # optimized + push to PM
python -m pytest tests/                   # run tests
ruff check ytx.py                         # lint
```

## Next Steps

1. Read `CLAUDE.md` for prime directive
2. Check `rules/` for coding and commit standards
3. Start working on the task at hand
