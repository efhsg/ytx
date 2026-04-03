# ytx - YouTube Transcript Extractor

CLI tool that extracts YouTube video transcripts via API and outputs clean markdown. No browser automation needed — pass a URL, get formatted text in seconds.

## Prerequisites

| Dependency | Version | Required | Purpose |
|------------|---------|----------|---------|
| Python | 3.12+ | Yes | Runtime |
| pip | latest | Yes | Package installation |
| `youtube-transcript-api` | ≥1.0.0 | Yes | Transcript fetching |
| `yt-dlp` | ≥2024.0.0 | No | Video metadata (description, tags, upload date). Falls back to oEmbed API if missing |

## Installation

```bash
# 1. Clone the repository
git clone <repo-url> && cd ytx

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python ytx.py "<youtube_url>" -o output.md
python ytx.py VIDEO_ID -l nl        # Dutch
python ytx.py VIDEO_ID --json       # JSON output
python ytx.py VIDEO_ID -O           # AI-optimized output
python ytx.py VIDEO_ID -O --push    # Optimized + push to PromptManager
```

## Flags

| Flag | Description |
|------|-------------|
| `-o FILE` | Save to file (default: stdout) |
| `-l LANG` | Preferred language code (default: en) |
| `--json` | Output as JSON instead of markdown |
| `-O, --optimize` | AI-optimized: cleans filler, extracts tools/steps/sections |
| `--push` | Push output to PromptManager. Requires configuration (see below). Skips with a warning if not configured |
| `--project NAME` | PromptManager project name (default: `YouTube transcripts`) |

## Output

**Standard:** Markdown with title, channel, URL, date, description, and transcript.

**Optimized (-O):** Adds extracted tools/products, key steps, table of contents, and logical sections.

## Configuration

Configuration is only needed when using `--push` to send transcripts to PromptManager.

### Config file

Create a `.ytx.json` file in the project root or your home directory (`~/.ytx.json`):

```json
{
  "promptmanager_url": "http://localhost:8503",
  "promptmanager_token": "your-api-token"
}
```

**Priority order** (highest wins):

1. Environment variables
2. `.ytx.json` (project root)
3. `~/.ytx.json` (home directory)

### Environment variables

| Variable | Description |
|----------|-------------|
| `YTX_PROMPTMANAGER_URL` | PromptManager API base URL |
| `YTX_PROMPTMANAGER_TOKEN` | PromptManager API bearer token |

Set these in your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export YTX_PROMPTMANAGER_URL="http://localhost:8503"
export YTX_PROMPTMANAGER_TOKEN="your-api-token"
```

> **Security:** Store the token in an environment variable rather than in a project config file. The URL is not sensitive and can go in `.ytx.json`.

### Token rotation

Generate or rotate a PromptManager API token:

```bash
docker compose exec pma_yii php yii user/rotate-token <user-id>
```
