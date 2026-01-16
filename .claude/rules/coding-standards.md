# Coding Standards

Python conventions for ytx.

## Type Hints

Add type hints for all function signatures:

```python
def extract_video_id(url: str) -> str | None:
    ...
```

## Docstrings

Public functions get docstrings (one-liner for simple functions):

```python
def fetch_transcript(video_id: str) -> list[dict]:
    """Fetch transcript for a YouTube video."""
    ...
```

## Error Handling

- No bare `except:` clauses - always catch specific exceptions
- Use `except Exception as e` if truly generic handling needed
- Print errors to stderr: `print(f"ERROR: {msg}", file=sys.stderr)`

## String Formatting

- Prefer f-strings: `f"Video: {title}"`
- Avoid `.format()` and `%` formatting

## File Operations

- Prefer `pathlib.Path` for file paths when practical
- For simple cases, `open()` with context manager is fine

## Imports

Order: stdlib → third-party → local (alphabetical within groups)

```python
import json
import sys
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
```

## CLI Conventions

- Use `argparse` (already in use)
- Exit codes: 0 success, 1 error
- Errors to stderr, output to stdout (or file with `-o`)
