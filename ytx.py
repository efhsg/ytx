#!/usr/bin/env python3
"""
ytx - YouTube Transcript Extractor

Usage:
    python ytx.py <youtube_url_or_id> [-o FILE] [-l LANG] [--json] [-O] [--push]

Examples:
    python ytx.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python ytx.py dQw4w9WgXcQ -o transcript.md
    python ytx.py dQw4w9WgXcQ -l nl
    python ytx.py dQw4w9WgXcQ -O --push --project "AI Research"
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        VideoUnavailable,
    )
except ImportError:
    print("ERROR: youtube-transcript-api not installed.")
    print("Run: pip install youtube-transcript-api")
    sys.exit(1)


def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from URL or return as-is if already an ID."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def youtube_url(video_id: str) -> str:
    """Build canonical YouTube watch URL."""
    return f"https://www.youtube.com/watch?v={video_id}"


def normalize_base_url(url: str) -> str:
    """Remove trailing slash from URL."""
    return url.rstrip('/')


def load_config() -> dict:
    """Load config from ~/.ytx.json or environment variables."""
    config = {}
    config_path = os.path.expanduser('~/.ytx.json')
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid JSON in {config_path}: {exc}", file=sys.stderr)
            sys.exit(1)
    # Environment overrides file config
    config['promptmanager_url'] = normalize_base_url(
        os.environ.get('YTX_PROMPTMANAGER_URL', config.get('promptmanager_url', '')))
    config['promptmanager_token'] = os.environ.get(
        'YTX_PROMPTMANAGER_TOKEN', config.get('promptmanager_token', ''))
    return config


def push_to_promptmanager(name: str, content: str, project: str, config: dict,
                          content_format: str = 'md') -> dict:
    """Push content to PromptManager scratch_pad API."""
    url = f"{config['promptmanager_url']}/api/scratch-pad/create"
    headers = {
        'Authorization': f"Bearer {config['promptmanager_token']}",
        'Content-Type': 'application/json',
    }
    data = json.dumps({
        'name': name,
        'content': content,
        'format': content_format,
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


def fetch_video_metadata_ytdlp(video_id: str) -> dict | None:
    """Fetch video metadata using yt-dlp (includes description)."""
    try:
        result = subprocess.run(
            ['yt-dlp', '--dump-json', '--no-download', youtube_url(video_id)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                'title': data.get('title', 'Unknown Title'),
                'channel': data.get('channel', data.get('uploader', 'Unknown Channel')),
                'channel_url': data.get('channel_url', data.get('uploader_url', '')),
                'description': data.get('description', ''),
                'tags': data.get('tags', []),
                'upload_date': data.get('upload_date', ''),
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def fetch_video_metadata_oembed(video_id: str) -> dict:
    """Fetch video metadata using oembed API (fallback, no description)."""
    oembed_url = f"https://www.youtube.com/oembed?url={youtube_url(video_id)}&format=json"

    try:
        with urllib.request.urlopen(oembed_url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            return {
                'title': data.get('title', 'Unknown Title'),
                'channel': data.get('author_name', 'Unknown Channel'),
                'channel_url': data.get('author_url', ''),
                'description': '',
                'tags': [],
                'upload_date': '',
            }
    except (urllib.error.URLError, json.JSONDecodeError):
        return {
            'title': 'Unknown Title',
            'channel': 'Unknown Channel',
            'channel_url': '',
            'description': '',
            'tags': [],
            'upload_date': '',
        }


def fetch_video_metadata(video_id: str) -> dict:
    """Fetch video metadata, trying yt-dlp first, then oembed fallback."""
    metadata = fetch_video_metadata_ytdlp(video_id)
    if metadata:
        return metadata
    return fetch_video_metadata_oembed(video_id)


def get_transcript(video_id: str, lang: str = 'en') -> list[dict]:
    """Fetch transcript, trying requested language first, then fallbacks.

    Prefers manual transcripts over auto-generated ones.
    """
    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except TranscriptsDisabled:
        raise Exception(f"Transcripts are disabled for video: {video_id}")
    except VideoUnavailable:
        raise Exception(f"Video unavailable: {video_id}")

    # Separate manual and auto-generated transcripts
    manual = [t for t in transcript_list if not t.is_generated]
    auto = [t for t in transcript_list if t.is_generated]

    # Try to find transcript in preferred order: manual first, then auto
    for transcripts in [manual, auto]:
        # First try requested language
        for t in transcripts:
            if t.language_code == lang:
                return list(t.fetch())
        # Then try English
        for t in transcripts:
            if t.language_code == 'en':
                return list(t.fetch())

    # Fall back to any available (manual preferred)
    for transcripts in [manual, auto]:
        if transcripts:
            return list(transcripts[0].fetch())

    raise Exception(f"No transcripts found for video: {video_id}")


def transcript_to_text(transcript: list) -> str:
    """Convert transcript entries to clean text without timestamps."""
    return ' '.join(entry.text.replace('\n', ' ') for entry in transcript)


# Filler patterns to remove
FILLER_PATTERNS = [
    r'\[music\]',
    r'\[applause\]',
    r'\[laughter\]',
    r'\[♪♪♪?\]',
    r'♪[^♪]*♪',
    r'\b(um|uh|er|ah)\b',
    r'\byou know\b',
    r'\blike,?\s+',
    r'\bso,?\s+(?=so\b)',  # repeated "so so"
]

# Section marker phrases
SECTION_MARKERS = [
    r"(?:^|\.\s+)(first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|lastly)[,\s]",
    r"(?:^|\.\s+)(now,?\s+let'?s?|next,?|moving on|let's talk about|here'?s?\s+(?:the|a|how))",
    r"(?:^|\.\s+)(step\s+\d+|the\s+(?:first|second|third|next)\s+(?:step|thing|point))",
    r"(?:^|\.\s+)(before we|to (?:start|begin|end)|in conclusion)",
]

# Known AI/tech tools (case-insensitive matching)
KNOWN_TOOLS = {
    # AI models & companies
    'claude', 'gpt', 'gemini', 'llama', 'mistral', 'qwen', 'anthropic', 'openai',
    'deepseek', 'cohere', 'meta', 'google',
    # AI frameworks & concepts
    'langchain', 'langgraph', 'langsmith', 'llamaindex', 'autogen', 'crewai',
    'rag', 'repl', 'cot', 'rlhf',
    # Vector DBs & search
    'pinecone', 'weaviate', 'chromadb', 'faiss', 'milvus', 'qdrant',
    # Platforms
    'huggingface', 'ollama', 'groq', 'perplexity', 'copilot', 'cursor',
    'replit', 'vercel', 'supabase', 'firebase', 'mongodb', 'postgres',
    'redis', 'elasticsearch', 'docker', 'kubernetes', 'terraform',
    # Dev tools
    'github', 'gitlab', 'vscode', 'neovim', 'vim',
    # Note-taking & productivity
    'notion', 'obsidian', 'roam', 'logseq', 'notebooklm',
    # Data science
    'jupyter', 'colab', 'pandas', 'numpy', 'pytorch', 'tensorflow',
    # Web frameworks
    'streamlit', 'gradio', 'fastapi', 'flask', 'django', 'react', 'nextjs',
    # Languages
    'typescript', 'python', 'javascript', 'nodejs', 'deno', 'bun', 'rust', 'golang',
}

# Tool name patterns for regex extraction
TOOL_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Multi-word: "Google Docs"
    r'\b([A-Z][a-z]+(?:LM|AI|ML|DB|JS|API))\b',  # Compound: "NotebookLM", "ChromaDB"
    r'\b([A-Z]{2,6}(?:-?\d+(?:\.\d+)?)?)\b',  # Acronyms + versions: "LLM", "GPT-4"
    r'\b([A-Z][a-z]+[A-Z][a-z]+)\b',  # CamelCase: "LangChain", "LangGraph"
]

# Common English words/acronyms to exclude from tool detection
TOOL_EXCLUDE = {
    'The', 'This', 'That', 'What', 'When', 'Where', 'How', 'Why', 'And', 'But',
    'For', 'Now', 'Here', 'There', 'First', 'Next', 'Finally', 'Also', 'Just',
    'Well', 'Look', 'Click', 'Let', 'Before', 'After', 'Don', 'Okay', 'These',
    'They', 'Use', 'You', 'Yes', 'Right', 'So', 'We', 'It', 'I', 'A', 'In', 'On',
    'UK', 'US', 'CEO', 'API', 'PDF', 'URL', 'AI',
    'LM', 'ML',
    'SWAT', 'SWOT', 'ROI', 'KPI', 'FAQ', 'TBD', 'ETA', 'FYI',
}

# Common transcription errors for tool names
TRANSCRIPTION_FIXES = {
    'notebookm': 'NotebookLM',
    'notebook alm': 'NotebookLM',
    'notebook lm': 'NotebookLM',
    'lang graph': 'LangGraph',
    'lang chain': 'LangChain',
}

# Acronyms that should always be uppercase
UPPERCASE_ACRONYMS = {'llm', 'rag', 'rlm', 'gpt', 'api', 'repl', 'cot', 'rlhf', 'sql', 'nlp'}

# Patterns for extracting actionable steps
STEP_PATTERNS = [
    r"(?:you (?:need to|want to|should|can|must)|let's|go ahead and|click on|hit)\s+([^.!?]+)",
    r"(?:step \d+[:\s]+)([^.!?]+)",
    r"(?:the (?:key|secret|trick) is)\s+([^.!?]+)",
]


def clean_transcript(text: str) -> str:
    """Remove filler phrases and noise from transcript."""
    cleaned = text
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def extract_sections(text: str) -> list[dict]:
    """Split transcript into logical sections based on transitional phrases."""
    sections = []

    # Find all section markers
    markers = []
    for pattern in SECTION_MARKERS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            markers.append(match.start())

    if not markers:
        return [{'title': 'Content', 'text': text}]

    markers = sorted(set(markers))

    # Split at markers
    for i, start in enumerate(markers):
        end = markers[i + 1] if i + 1 < len(markers) else len(text)
        section_text = text[start:end].strip()

        # Clean leading punctuation and extract first sentence as title
        section_text_clean = re.sub(r'^[.\s,]+', '', section_text)
        sentences = re.split(r'(?<=[.!?])\s+', section_text_clean)
        first_sentence = sentences[0][:80] if sentences else 'Section'
        # Capitalize first letter
        if first_sentence:
            first_sentence = first_sentence[0].upper() + first_sentence[1:]
        sections.append({
            'title': first_sentence,
            'text': section_text_clean
        })

    # Add intro if first marker isn't at start
    if markers[0] > 50:
        intro_text = text[:markers[0]].strip()
        sections.insert(0, {'title': 'Introduction', 'text': intro_text})

    return sections


def extract_tools(text: str) -> list[str]:
    """Extract tool and product names from transcript text."""
    potential_tools = []

    # Find tools via regex patterns
    for pattern in TOOL_PATTERNS:
        potential_tools.extend(re.findall(pattern, text))

    # Find known tools by case-insensitive search
    text_lower = text.lower()
    for tool in KNOWN_TOOLS:
        if tool in text_lower:
            pattern = re.compile(re.escape(tool), re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                potential_tools.extend(matches)

    # Apply transcription fixes
    for error, fix in TRANSCRIPTION_FIXES.items():
        if error in text_lower:
            potential_tools.append(fix)

    # Count occurrences and normalize casing
    tool_counts: dict[str, dict] = {}
    for tool in potential_tools:
        tool_clean = tool.strip()
        if tool_clean in TOOL_EXCLUDE or len(tool_clean) < 2:
            continue
        key = tool_clean.lower()
        if key in TRANSCRIPTION_FIXES:
            key = TRANSCRIPTION_FIXES[key].lower()
            tool_clean = TRANSCRIPTION_FIXES[key]
        if key not in tool_counts:
            tool_counts[key] = {'count': 0, 'display': tool_clean}
        tool_counts[key]['count'] += 1
        # Prefer version with more uppercase letters
        if sum(1 for c in tool_clean if c.isupper()) > sum(1 for c in tool_counts[key]['display'] if c.isupper()):
            tool_counts[key]['display'] = tool_clean

    # Keep tools appearing 2+ times, normalize acronym casing
    tools = []
    for key, val in tool_counts.items():
        if val['count'] >= 2:
            display = key.upper() if key in UPPERCASE_ACRONYMS else val['display']
            tools.append(display)

    return sorted(set(tools))


def extract_urls(text: str) -> list[str]:
    """Extract URLs from transcript text."""
    url_pattern = r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)'
    return list(set(re.findall(url_pattern, text)))


def extract_steps(text: str) -> list[str]:
    """Extract actionable steps from transcript text."""
    steps = []
    for pattern in STEP_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        steps.extend([m.strip() for m in matches if len(m.strip()) > 10])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_steps = []
    for step in steps[:15]:
        step_lower = step.lower()
        if step_lower not in seen:
            seen.add(step_lower)
            unique_steps.append(step)
    return unique_steps


def extract_metadata(text: str) -> dict:
    """Extract tools, URLs, and actionable steps from transcript."""
    return {
        'tools': extract_tools(text),
        'urls': extract_urls(text),
        'steps': extract_steps(text),
    }


def build_markdown_header(video_id: str, metadata: dict) -> list[str]:
    """Build common markdown header with title, channel, URL, date, and description."""
    date_extracted = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# {metadata['title']}",
        "",
        f"**Channel:** [{metadata['channel']}]({metadata['channel_url']})  ",
        f"**URL:** {youtube_url(video_id)}  ",
        f"**Extracted:** {date_extracted}",
        "",
    ]
    if metadata.get('description'):
        lines.extend([
            "## Description",
            "",
            metadata['description'],
            "",
        ])
    return lines


def format_optimized_markdown(video_id: str, metadata: dict, text: str,
                               sections: list[dict], extracted: dict) -> str:
    """Format transcript as AI-optimized markdown with metadata."""
    lines = build_markdown_header(video_id, metadata)

    # Tools mentioned
    if extracted['tools']:
        lines.append("## Tools & Products Mentioned")
        lines.append("")
        for tool in extracted['tools']:
            lines.append(f"- {tool}")
        lines.append("")

    # Key steps
    if extracted['steps']:
        lines.append("## Key Steps & Actions")
        lines.append("")
        for i, step in enumerate(extracted['steps'], 1):
            lines.append(f"{i}. {step.capitalize()}")
        lines.append("")

    # Table of contents
    if len(sections) > 1:
        lines.append("## Table of Contents")
        lines.append("")
        for i, section in enumerate(sections, 1):
            title = section['title'][:60] + ('...' if len(section['title']) > 60 else '')
            lines.append(f"{i}. {title}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Sections
    if len(sections) > 1:
        for i, section in enumerate(sections, 1):
            lines.append(f"## {i}. {section['title'][:60]}")
            lines.append("")
            lines.append(section['text'])
            lines.append("")
    else:
        lines.append("## Transcript")
        lines.append("")
        lines.append(text)

    return '\n'.join(lines)


def format_markdown(video_id: str, metadata: dict, transcript_text: str) -> str:
    """Format transcript as markdown document."""
    lines = build_markdown_header(video_id, metadata)
    lines.extend([
        "---",
        "",
        "## Transcript",
        "",
        transcript_text,
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='ytx - YouTube Transcript Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('video', help='YouTube URL or video ID')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-l', '--lang', default='en', help='Preferred language (default: en)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('-O', '--optimize', action='store_true',
                        help='AI-optimized output: clean filler, extract sections/tools/steps')
    parser.add_argument('--push', action='store_true',
                        help='Push to PromptManager scratch_pad')
    parser.add_argument('--project', default='YouTube transcripts',
                        help='PromptManager project name (default: YouTube transcripts)')

    args = parser.parse_args()

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

    try:
        video_id = extract_video_id(args.video)

        # Fetch metadata and transcript in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            metadata_future = executor.submit(fetch_video_metadata, video_id)
            transcript_future = executor.submit(get_transcript, video_id, args.lang)
            metadata = metadata_future.result()
            transcript = transcript_future.result()

        transcript_text = transcript_to_text(transcript)

        if args.optimize:
            transcript_text = clean_transcript(transcript_text)
            sections = extract_sections(transcript_text)
            extracted = extract_metadata(transcript_text)

            if args.json:
                output = json.dumps({
                    'video_id': video_id,
                    'title': metadata['title'],
                    'channel': metadata['channel'],
                    'description': metadata.get('description', ''),
                    'tags': metadata.get('tags', []),
                    'url': youtube_url(video_id),
                    'tools': extracted['tools'],
                    'steps': extracted['steps'],
                    'sections': [s['title'] for s in sections],
                    'transcript': transcript_text,
                }, indent=2)
            else:
                output = format_optimized_markdown(video_id, metadata, transcript_text,
                                                    sections, extracted)
        elif args.json:
            output = json.dumps({
                'video_id': video_id,
                'title': metadata['title'],
                'channel': metadata['channel'],
                'description': metadata.get('description', ''),
                'tags': metadata.get('tags', []),
                'url': youtube_url(video_id),
                'transcript': transcript_text,
            }, indent=2)
        else:
            output = format_markdown(video_id, metadata, transcript_text)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Saved to: {args.output}", file=sys.stderr)
        elif not args.push:
            print(output)

        # Push to PromptManager if requested
        if args.push:
            result = push_to_promptmanager(
                name=metadata['title'],
                content=output,
                project=args.project,
                config=config,
                content_format='text' if args.json else 'md'
            )
            if 'id' in result:
                print(f"Pushed to PromptManager: ID {result['id']}", file=sys.stderr)
            else:
                print(f"ERROR: {result.get('error', result)}", file=sys.stderr)
                sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
