# ytx - YouTube Transcript Extractor

CLI tool that extracts YouTube video transcripts via API and outputs clean markdown. No browser automation needed - just pass a URL, get formatted text in seconds.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python ytx.py "<youtube_url>" -o output.md
python ytx.py VIDEO_ID -l nl        # Dutch
python ytx.py VIDEO_ID --json       # JSON output
python ytx.py VIDEO_ID -O           # AI-optimized output
```

## Flags

| Flag | Description |
|------|-------------|
| `-o FILE` | Save to file (default: stdout) |
| `-l LANG` | Preferred language code (default: en) |
| `--json` | Output as JSON instead of markdown |
| `-O, --optimize` | AI-optimized: cleans filler, extracts tools/steps/sections |

## Output

**Standard:** Markdown with title, channel, URL, date, description, and transcript.

**Optimized (-O):** Adds extracted tools/products, key steps, table of contents, and logical sections.

## Dependencies

- `youtube-transcript-api` - transcript fetching (required)
- `yt-dlp` - metadata + description (optional, falls back to oEmbed)
