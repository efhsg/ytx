# Testing Standards

## Framework

pytest — run with `python -m pytest tests/`

## Test Naming

- Files: `tests/test_{domain}.py` (e.g., `test_extraction.py`, `test_optimization.py`)
- Functions: `test_{action}_{condition}` (e.g., `test_extract_video_id_from_short_url`)

## What to Test

- Video ID extraction from all URL formats
- Transcript processing and cleaning
- Optimization pipeline (filler removal, section extraction, tool extraction)
- CLI argument parsing and output formatting
- PromptManager integration (push, config loading)
- Error handling (missing transcripts, invalid URLs, API failures)

## Mocking Strategy

- Mock external calls: YouTube API, yt-dlp subprocess, oEmbed API, PromptManager HTTP
- Don't mock internal logic — test real functions with real inputs
- Use `unittest.mock.patch` for external dependencies

## Test Structure

```python
def test_extract_video_id_from_watch_url():
    """Single behavior per test, descriptive name."""
    result = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert result == "dQw4w9WgXcQ"
```

## Running Tests

```bash
python -m pytest tests/                     # all tests
python -m pytest tests/test_extraction.py   # single file
python -m pytest tests/ -k "test_extract"   # pattern match
```
