# ytx: PromptManager Integration

## Overview
Add `--push` flag to send transcripts directly to PromptManager's ScratchPad API.

---

## 1. Add CLI Flags

**File:** `ytx.py`

```python
parser.add_argument('--push', action='store_true',
                    help='Push to PromptManager scratch_pad')
parser.add_argument('--project', default='YouTube transcripts',
                    help='PromptManager project name (default: YouTube transcripts)')
```

---

## 2. Config File Support

**File:** `~/.ytx.json`

```json
{
  "promptmanager_url": "http://localhost:8503",
  "promptmanager_token": "your-api-token"
}
```

**Alternative:** Environment variables
```bash
export YTX_PROMPTMANAGER_URL=http://localhost:8503
export YTX_PROMPTMANAGER_TOKEN=your-api-token
```

**Note:** Prefer env vars for secrets; if using `~/.ytx.json`, restrict permissions (e.g., `chmod 600 ~/.ytx.json`).

---

## 3. Config Loading + Validation

**Behavior:**
- If `--push` is set, require both `promptmanager_url` and `promptmanager_token`.
- Normalize the URL to avoid trailing slash issues.
- If config JSON is invalid, print a clear error and exit.

```python
def normalize_base_url(url: str) -> str:
    return url.rstrip('/')

def load_config() -> dict:
    """Load config from ~/.ytx.json or environment."""
    config = {}
    config_path = os.path.expanduser('~/.ytx.json')
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid JSON in {config_path}: {exc}", file=sys.stderr)
            sys.exit(1)
    # Environment overrides
    config['promptmanager_url'] = os.environ.get(
        'YTX_PROMPTMANAGER_URL', config.get('promptmanager_url', ''))
    config['promptmanager_token'] = os.environ.get(
        'YTX_PROMPTMANAGER_TOKEN', config.get('promptmanager_token', ''))
    config['promptmanager_url'] = normalize_base_url(
        config.get('promptmanager_url', ''))
    return config
```

---

## 4. API Client Function

**File:** `ytx.py`

```python
def push_to_promptmanager(name: str, content: str, project: str, config: dict) -> dict:
    """Push content to PromptManager scratch_pad API."""
    url = f"{config['promptmanager_url']}/api/scratch-pad/create"
    headers = {
        'Authorization': f"Bearer {config['promptmanager_token']}",
        'Content-Type': 'application/json',
    }
    data = json.dumps({
        'name': name,
        'content': content,
        'project_name': project,
    }).encode('utf-8')

    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        return {'error': f'HTTP {exc.code}: {exc.reason}'}
    except urllib.error.URLError as exc:
        return {'error': f'Network error: {exc.reason}'}
```

---

## 5. Integration in main()

```python
# Validate config early (before fetching transcript) to fail fast
config = None
if args.push:
    config = load_config()
    if not config.get('promptmanager_url'):
        print("ERROR: No PromptManager URL configured", file=sys.stderr)
        print("Set YTX_PROMPTMANAGER_URL or add to ~/.ytx.json", file=sys.stderr)
        sys.exit(1)
    if not config.get('promptmanager_token'):
        print("ERROR: No PromptManager token configured", file=sys.stderr)
        print("Set YTX_PROMPTMANAGER_TOKEN or add to ~/.ytx.json", file=sys.stderr)
        sys.exit(1)

# ... fetch transcript and generate output ...

# Push after output is ready
if args.push:
    result = push_to_promptmanager(
        name=metadata['title'],
        content=output,
        project=args.project,
        config=config
    )
    # API returns {"id": <int>} on success, or {"error": "..."} on failure
    if 'id' in result:
        print(f"Pushed to PromptManager: ID {result['id']}", file=sys.stderr)
    else:
        print(f"ERROR: {result.get('error', result)}", file=sys.stderr)
        sys.exit(1)
```

---

## 6. Behavior Notes

- **Output format:** `--push` works with both markdown (default) and `--json` output
- **Duplicate names:** PromptManager creates a new entry; no deduplication is performed
- **Combining flags:** `--push` can be combined with `-o` to both save locally and push remotely

---

## Usage

```bash
# Basic push (uses default project "YouTube transcripts")
python ytx.py "VIDEO_URL" -O --push

# Custom project
python ytx.py "VIDEO_URL" -O --push --project "AI Research"

# With file output AND push
python ytx.py "VIDEO_URL" -O -o transcript.md --push

# Push JSON output
python ytx.py "VIDEO_URL" --json --push
```

---

## Files to Modify

| File | Action |
|------|--------|
| `ytx.py` | Add --push, --project flags, config loading, API client |

---

## Verification

1. Create `~/.ytx.json` with token from PromptManager
2. Run: `python ytx.py "VIDEO_URL" -O --push`
3. Check PromptManager web UI for new scratch_pad
4. Test negative cases:
   - Missing URL → clear error, exits before fetching transcript
   - Missing token → clear error, exits before fetching transcript
   - Invalid JSON in config → parse error with line info
   - Network error → connection refused / timeout message
   - API error (401, 500) → HTTP status code in error
