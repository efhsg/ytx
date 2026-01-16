# ytx - YouTube Transcript Extractor

## Prime Directive

Before making any code changes:
1. Read `.claude/rules/coding-standards.md` for Python conventions
2. Follow `.claude/rules/commits.md` for commit messages
3. Run `/finalize-changes` before committing

## Intent

ytx extracts YouTube video transcripts via API and outputs clean markdown. No browser automation needed - just pass a URL, get formatted text in seconds.

## Usage
```bash
python ytx.py "<url>" -o output.md
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
| `-O, --optimize` | AI-optimized: cleans filler, extracts sections/tools/steps |

## Output

**Standard:** Markdown with title (H1), channel, URL, date, description, transcript text (no timestamps).

**Optimized (-O):** Adds tools mentioned, key steps, table of contents, and logical sections.

## Dependencies

- `youtube-transcript-api` - transcript fetching (required)
- `yt-dlp` - metadata + description (optional, falls back to oEmbed)
