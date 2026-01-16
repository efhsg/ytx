#!/usr/bin/env python3
"""
ytx - YouTube Transcript Extractor

Usage:
    python ytx.py <youtube_url_or_id> [-o FILE] [-l LANG] [--json]

Examples:
    python ytx.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python ytx.py dQw4w9WgXcQ -o transcript.md
    python ytx.py dQw4w9WgXcQ -l nl
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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


def fetch_video_metadata_ytdlp(video_id: str) -> dict | None:
    """Fetch video metadata using yt-dlp (includes description)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        result = subprocess.run(
            ['yt-dlp', '--dump-json', '--no-download', url],
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
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

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


def extract_metadata(text: str) -> dict:
    """Extract tools, links, and actionable steps from transcript."""

    # Known AI/tech tools (case-insensitive matching)
    KNOWN_TOOLS = {
        # AI models & companies
        'claude', 'gpt', 'gemini', 'llama', 'mistral', 'qwen', 'anthropic', 'openai',
        'deepseek', 'cohere', 'meta', 'google',
        # AI frameworks & concepts
        'langchain', 'langgraph', 'langsmith', 'llamaindex', 'autogen', 'crewai',
        'rag', 'repl', 'cot', 'rlhf',  # Common acronyms
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

    # Tool patterns
    tool_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Multi-word: "Google Docs"
        r'\b([A-Z][a-z]+(?:LM|AI|ML|DB|JS|API))\b',  # Compound: "NotebookLM", "ChromaDB"
        r'\b([A-Z]{2,6}(?:-?\d+(?:\.\d+)?)?)\b',  # Acronyms + versions: "LLM", "GPT-4", "GPT5"
        r'\b([A-Z][a-z]+[A-Z][a-z]+)\b',  # CamelCase: "LangChain", "LangGraph"
    ]
    potential_tools = []
    for pattern in tool_patterns:
        potential_tools.extend(re.findall(pattern, text))

    # Also find known tools by case-insensitive search
    text_lower = text.lower()
    for tool in KNOWN_TOOLS:
        if tool in text_lower:
            # Find original casing in text
            pattern = re.compile(re.escape(tool), re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                potential_tools.extend(matches)

    # Common English words/acronyms to exclude
    exclude = {
        'The', 'This', 'That', 'What', 'When', 'Where', 'How', 'Why', 'And', 'But',
        'For', 'Now', 'Here', 'There', 'First', 'Next', 'Finally', 'Also', 'Just',
        'Well', 'Look', 'Click', 'Let', 'Before', 'After', 'Don', 'Okay', 'These',
        'They', 'Use', 'You', 'Yes', 'Right', 'So', 'We', 'It', 'I', 'A', 'In', 'On',
        'UK', 'US', 'CEO', 'API', 'PDF', 'URL', 'AI',  # Too generic
        'LM', 'ML',  # Too short without prefix
        'SWAT', 'SWOT', 'ROI', 'KPI', 'FAQ', 'TBD', 'ETA', 'FYI',  # Business acronyms
    }

    # Normalize common transcription errors
    TRANSCRIPTION_FIXES = {
        'notebookm': 'NotebookLM',
        'notebook alm': 'NotebookLM',
        'notebook lm': 'NotebookLM',
        'lang graph': 'LangGraph',
        'lang chain': 'LangChain',
    }

    # Apply transcription fixes to text for better matching
    text_fixed = text.lower()
    for error, fix in TRANSCRIPTION_FIXES.items():
        if error in text_fixed:
            potential_tools.append(fix)

    # Count and filter
    tool_counts = {}
    for tool in potential_tools:
        tool_clean = tool.strip()
        if tool_clean not in exclude and len(tool_clean) >= 2:
            # Normalize: lowercase for counting, keep best casing
            key = tool_clean.lower()
            # Apply transcription fix if available
            if key in TRANSCRIPTION_FIXES:
                key = TRANSCRIPTION_FIXES[key].lower()
                tool_clean = TRANSCRIPTION_FIXES[key]
            if key not in tool_counts:
                tool_counts[key] = {'count': 0, 'display': tool_clean}
            tool_counts[key]['count'] += 1
            # Prefer version with more caps (e.g., "LangGraph" over "langgraph")
            if sum(1 for c in tool_clean if c.isupper()) > sum(1 for c in tool_counts[key]['display'] if c.isupper()):
                tool_counts[key]['display'] = tool_clean

    # Acronyms that should always be uppercase
    UPPERCASE_ACRONYMS = {'llm', 'rag', 'rlm', 'gpt', 'api', 'repl', 'cot', 'rlhf', 'sql', 'nlp'}

    # Keep tools that appear 2+ times, normalize acronym casing
    tools = []
    for k, v in tool_counts.items():
        if v['count'] >= 2:
            display = v['display']
            if k in UPPERCASE_ACRONYMS:
                display = k.upper()
            tools.append(display)

    # Extract URLs
    url_pattern = r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)'
    urls = list(set(re.findall(url_pattern, text)))

    # Extract key steps/actions
    step_patterns = [
        r"(?:you (?:need to|want to|should|can|must)|let's|go ahead and|click on|hit)\s+([^.!?]+)",
        r"(?:step \d+[:\s]+)([^.!?]+)",
        r"(?:the (?:key|secret|trick) is)\s+([^.!?]+)",
    ]
    steps = []
    for pattern in step_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        steps.extend([m.strip() for m in matches if len(m.strip()) > 10])

    # Deduplicate steps while preserving order
    seen = set()
    unique_steps = []
    for step in steps[:15]:  # Limit to top 15
        step_lower = step.lower()
        if step_lower not in seen:
            seen.add(step_lower)
            unique_steps.append(step)

    return {
        'tools': sorted(set(tools)),
        'urls': urls,
        'steps': unique_steps,
    }


def format_optimized_markdown(video_id: str, metadata: dict, text: str,
                               sections: list[dict], extracted: dict) -> str:
    """Format transcript as AI-optimized markdown with metadata."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    date_extracted = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"# {metadata['title']}",
        "",
        f"**Channel:** [{metadata['channel']}]({metadata['channel_url']})  ",
        f"**URL:** {url}  ",
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
    url = f"https://www.youtube.com/watch?v={video_id}"
    date_extracted = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"# {metadata['title']}",
        "",
        f"**Channel:** [{metadata['channel']}]({metadata['channel_url']})  ",
        f"**URL:** {url}  ",
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
    
    args = parser.parse_args()
    
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
                    'url': f"https://www.youtube.com/watch?v={video_id}",
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
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'transcript': transcript_text,
            }, indent=2)
        else:
            output = format_markdown(video_id, metadata, transcript_text)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Saved to: {args.output}", file=sys.stderr)
        else:
            print(output)
            
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
