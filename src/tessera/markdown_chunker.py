"""Break-point markdown chunker ported from QMD (tobi/qmd).

Uses distance-decay scored break points to produce semantically coherent
chunks. Key idea: scan all potential break points upfront, then when splitting,
search backward from the target position and pick the highest-quality break
point with squared-distance decay so headings far back still beat blank lines
at the boundary.
"""

import re
from dataclasses import dataclass

from .document import DocumentChunk

# -- Defaults (matching QMD) --------------------------------------------------

CHUNK_SIZE_CHARS = 3600      # ~900 tokens * 4 chars/token
CHUNK_OVERLAP_CHARS = 540    # 15% of chunk size
CHUNK_WINDOW_CHARS = 800     # ~200 tokens search window
DECAY_FACTOR = 0.7


# -- Data structures -----------------------------------------------------------

@dataclass
class BreakPoint:
    pos: int
    score: int
    type: str


@dataclass
class CodeFenceRegion:
    start: int
    end: int


# -- Break point patterns (score descending, position-deduped) -----------------
# Patterns match against \n-prefixed text so they find line starts.
# Higher scores = better split points.

BREAK_PATTERNS: list[tuple[str, int, str]] = [
    (r'\n#{1}(?!#)',               100, 'h1'),
    (r'\n#{2}(?!#)',                90, 'h2'),
    (r'\n#{3}(?!#)',                80, 'h3'),
    (r'\n#{4}(?!#)',                70, 'h4'),
    (r'\n#{5}(?!#)',                60, 'h5'),
    (r'\n#{6}(?!#)',                50, 'h6'),
    (r'\n```',                      80, 'codeblock'),
    (r'\n(?:---|\*\*\*|___)\s*\n',  60, 'hr'),
    (r'\n\n+',                      20, 'blank'),
    (r'\n[-*]\s',                    5, 'list'),
    (r'\n\d+\.\s',                   5, 'numlist'),
    (r'\n',                          1, 'newline'),
]


def scan_break_points(text: str) -> list[BreakPoint]:
    """Scan text for all potential break points, dedup by position (keep highest score)."""
    seen: dict[int, BreakPoint] = {}

    for pattern, score, bp_type in BREAK_PATTERNS:
        for match in re.finditer(pattern, text):
            pos = match.start()
            existing = seen.get(pos)
            if existing is None or score > existing.score:
                seen[pos] = BreakPoint(pos=pos, score=score, type=bp_type)

    return sorted(seen.values(), key=lambda bp: bp.pos)


def find_code_fences(text: str) -> list[CodeFenceRegion]:
    """Find all code fence regions. Unclosed fences extend to end of text."""
    regions: list[CodeFenceRegion] = []
    in_fence = False
    fence_start = 0

    for match in re.finditer(r'\n```', text):
        if not in_fence:
            fence_start = match.start()
            in_fence = True
        else:
            regions.append(CodeFenceRegion(start=fence_start, end=match.start() + len(match.group())))
            in_fence = False

    if in_fence:
        regions.append(CodeFenceRegion(start=fence_start, end=len(text)))

    return regions


def is_inside_code_fence(pos: int, fences: list[CodeFenceRegion]) -> bool:
    """Check if a position falls strictly inside a code fence region."""
    return any(f.start < pos < f.end for f in fences)


def find_best_cutoff(
    break_points: list[BreakPoint],
    target_pos: int,
    window_chars: int = CHUNK_WINDOW_CHARS,
    decay_factor: float = DECAY_FACTOR,
    code_fences: list[CodeFenceRegion] | None = None,
) -> int:
    """Find the best cut position using scored break points with distance decay.

    Squared distance decay: gentle near target, steep at window edge.
    A heading at 50% back scores 82.5 vs a blank line at 0% back scoring 20.
    """
    if code_fences is None:
        code_fences = []

    window_start = target_pos - window_chars
    best_score = -1.0
    best_pos = target_pos

    for bp in break_points:
        if bp.pos < window_start:
            continue
        if bp.pos > target_pos:
            break  # sorted, so done

        if is_inside_code_fence(bp.pos, code_fences):
            continue

        distance = target_pos - bp.pos
        normalized_dist = distance / window_chars if window_chars > 0 else 0
        multiplier = 1.0 - (normalized_dist ** 2) * decay_factor
        final_score = bp.score * multiplier

        if final_score > best_score:
            best_score = final_score
            best_pos = bp.pos

    return best_pos


def _find_headings_before(text: str, pos: int) -> tuple[str, str]:
    """Find the most recent section_heading and parent_section before pos."""
    header_stack: dict[int, str] = {}  # level -> text
    current_heading = ""
    parent_section = ""

    for match in re.finditer(r'^(#{1,6})\s+(.+)', text[:pos], re.MULTILINE):
        level = len(match.group(1))
        heading_text = match.group(2).strip()

        # Clear lower levels
        header_stack = {k: v for k, v in header_stack.items() if k <= level}
        header_stack[level] = heading_text

        current_heading = heading_text
        if level == 1:
            parent_section = ""
        elif level == 2:
            parent_section = header_stack.get(1, "")
        else:
            parent_section = header_stack.get(level - 1, header_stack.get(1, ""))

    return current_heading, parent_section


def chunk_markdown_breakpoint(
    text: str,
    max_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    window_chars: int = CHUNK_WINDOW_CHARS,
    decay_factor: float = DECAY_FACTOR,
) -> list[DocumentChunk]:
    """Chunk markdown using QMD's break-point algorithm with distance-decay scoring.

    Args:
        text: Markdown text to chunk.
        max_chars: Maximum chunk size in characters (default 3600, ~900 tokens).
        overlap_chars: Character overlap between adjacent chunks (default 540, 15%).
        window_chars: How far back to search for break points (default 800, ~200 tokens).
        decay_factor: Distance decay strength (default 0.7).

    Returns:
        List of DocumentChunk objects with heading hierarchy metadata.
    """
    if not text or not text.strip():
        return []

    if len(text) <= max_chars:
        section_heading, parent_section = _find_headings_before(text, len(text))
        return [DocumentChunk(
            content=text,
            source_type="markdown",
            section_heading=section_heading,
            parent_section=parent_section,
            start_line=0,
            end_line=text.count('\n'),
        )]

    break_points = scan_break_points(text)
    code_fences = find_code_fences(text)

    chunks: list[DocumentChunk] = []
    char_pos = 0

    while char_pos < len(text):
        target_end = min(char_pos + max_chars, len(text))
        end_pos = target_end

        # If not at document end, find best break point
        if end_pos < len(text):
            best_cutoff = find_best_cutoff(
                break_points, target_end, window_chars, decay_factor, code_fences,
            )
            if char_pos < best_cutoff <= target_end:
                end_pos = best_cutoff

        # Progress guarantee
        if end_pos <= char_pos:
            end_pos = min(char_pos + max_chars, len(text))

        chunk_text = text[char_pos:end_pos]
        start_line = text[:char_pos].count('\n')
        end_line = start_line + chunk_text.count('\n')
        section_heading, parent_section = _find_headings_before(text, end_pos)

        chunks.append(DocumentChunk(
            content=chunk_text,
            source_type="markdown",
            section_heading=section_heading,
            parent_section=parent_section,
            start_line=start_line,
            end_line=end_line,
        ))

        if end_pos >= len(text):
            break

        # Overlap: step back by overlap_chars
        char_pos = end_pos - overlap_chars

        # Prevent infinite loop — must advance past last chunk's start
        if chunks and char_pos <= (len(text) - len(chunks[-1].content)):
            # More precise: ensure we advance past previous char_pos
            prev_start = end_pos - len(chunk_text)
            if char_pos <= prev_start:
                char_pos = end_pos

    return chunks
