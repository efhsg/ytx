# Project Configuration

Single source of truth for project-specific operations.

## Environment

| Setting | Value |
|---------|-------|
| Language | Python 3.12+ |
| Entry point | `ytx.py` (single-file CLI) |
| Test Framework | pytest |
| Package manager | pip + `requirements.txt` |

**Host-native**: ytx draait direct op het hostsysteem, geen container. Python venv wordt aanbevolen (`.venv/`).

**Git & GitHub**: Standaard git/gh setup. Geen speciale mounts of configuratie nodig.

## Commands

### Lint

```bash
# Syntax check
python -m py_compile ytx.py

# Lint (ruff preferred, flake8 als fallback)
ruff check ytx.py 2>/dev/null || flake8 ytx.py 2>/dev/null || echo "No linter installed"
```

### Tests

```bash
# Run alle tests
python -m pytest tests/

# Run enkel testbestand
python -m pytest tests/test_push_config.py

# Run enkele testmethode
python -m pytest tests/test_push_config.py::TestLoadConfig::test_env_var_overrides

# Verbose output
python -m pytest tests/ -v
```

### Run

```bash
# Basis gebruik
python ytx.py "<youtube_url>" -o output.md

# AI-optimized output
python ytx.py VIDEO_ID -O

# Push naar PromptManager
python ytx.py VIDEO_ID -O --push --project "AI Research"

# JSON output
python ytx.py VIDEO_ID --json
```

## File Structure

| Type | Location |
|------|----------|
| CLI entry point | `ytx.py` |
| Tests | `tests/` |
| Config (user) | `~/.ytx.json`, `.ytx.json` |
| Claude rules | `.claude/rules/` |
| Claude skills | `.claude/skills/` |
| Claude config | `.claude/config/` |
| Requirements | `requirements.txt` |

## Test Path Mapping

Single source file maps to test files by topic:

| Source | Test |
|--------|------|
| `ytx.py` (config/push logic) | `tests/test_push_config.py` |
| `ytx.py` (transcript extraction) | `tests/test_transcript.py` |
| `ytx.py` (video ID parsing) | `tests/test_extract_id.py` |
| `ytx.py` (output formatting) | `tests/test_formatting.py` |
| `ytx.py` (optimization pipeline) | `tests/test_optimize.py` |

**Conventie**: Test files groeperen per functioneel domein, niet per functie. Naamgeving: `tests/test_{domein}.py`.

## Key Domain Concepts

| Concept | Description |
|---------|-------------|
| Video ID | 11-character YouTube identifier, extracted from URL or passed directly |
| Transcript | List of timed text segments from YouTube's transcript API |
| Metadata | Video title, channel, description, tags, upload date |
| Optimization | Filler removal, section extraction, tool/step detection |
| Section | Logical segment of transcript, split on transitional phrases |
| PromptManager | External service for storing transcripts as notes (optional) |

## CLI Flags

| Flag | Description |
|------|-------------|
| `video` | YouTube URL or video ID (positional, required) |
| `-o FILE` | Save to file (default: stdout) |
| `-l LANG` | Preferred language code (default: `en`) |
| `--json` | Output as JSON instead of markdown |
| `-O, --optimize` | AI-optimized: clean filler, extract sections/tools/steps |
| `--push` | Push to PromptManager note (requires config) |
| `--project NAME` | PromptManager project name (default: `YouTube transcripts`) |

## Output Formats

| Mode | Content |
|------|---------|
| Standard (`md`) | Title, channel, URL, date, description, transcript |
| Optimized (`-O`) | + tools/products, key steps, table of contents, sections |
| JSON (`--json`) | Structured data with all fields |
| JSON + Optimized | + tools, steps, section titles |

## Configuration (PromptManager)

Alleen nodig bij `--push`. Prioriteit (hoogste wint):

1. Environment variables (`YTX_PROMPTMANAGER_URL`, `YTX_PROMPTMANAGER_TOKEN`)
2. `.ytx.json` (project root)
3. `~/.ytx.json` (home directory)

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `youtube-transcript-api` | Yes | Transcript fetching via YouTube API |
| `yt-dlp` | No | Rich metadata (description, tags, date). Falls back to oEmbed |
| `pytest` | Dev | Test runner |

## Metadata Fallback Chain

```
yt-dlp (rich: title, channel, description, tags, upload_date)
  |  (if yt-dlp unavailable or fails)
oEmbed API (basic: title, channel)
```
