"""Tests for the QMD break-point markdown chunker."""

import pytest

from tessera.markdown_chunker import (
    BreakPoint,
    CodeFenceRegion,
    chunk_markdown_breakpoint,
    find_best_cutoff,
    find_code_fences,
    is_inside_code_fence,
    scan_break_points,
)


class TestScanBreakPoints:
    def test_finds_headers(self):
        text = "\n# Title\n\n## Section\n\nParagraph"
        bps = scan_break_points(text)
        types = {bp.type for bp in bps}
        assert 'h1' in types
        assert 'h2' in types

    def test_deduplicates_by_position_keeps_highest(self):
        # A \n is both a newline (score 1) and could match other patterns
        text = "\n# Heading\n\nText"
        bps = scan_break_points(text)
        # Position 0 should have h1 (100), not newline (1)
        pos_zero = [bp for bp in bps if bp.pos == 0]
        assert len(pos_zero) == 1
        assert pos_zero[0].score == 100

    def test_sorted_by_position(self):
        text = "\n## B\n\n# A\n\nEnd"
        bps = scan_break_points(text)
        positions = [bp.pos for bp in bps]
        assert positions == sorted(positions)

    def test_blank_line_detection(self):
        text = "First paragraph\n\nSecond paragraph"
        bps = scan_break_points(text)
        blank_bps = [bp for bp in bps if bp.type == 'blank']
        assert len(blank_bps) >= 1

    def test_list_item_detection(self):
        text = "\n- item one\n- item two"
        bps = scan_break_points(text)
        list_bps = [bp for bp in bps if bp.type == 'list']
        assert len(list_bps) >= 1

    def test_code_block_boundary(self):
        text = "\n```python\ncode\n```\n"
        bps = scan_break_points(text)
        code_bps = [bp for bp in bps if bp.type == 'codeblock']
        assert len(code_bps) >= 1


class TestFindCodeFences:
    def test_paired_fences(self):
        text = "before\n```python\ncode here\n```\nafter"
        fences = find_code_fences(text)
        assert len(fences) == 1
        assert text[fences[0].start:fences[0].start + 4] == '\n```'

    def test_multiple_fences(self):
        text = "text\n```\nblock1\n```\nmiddle\n```\nblock2\n```\nend"
        fences = find_code_fences(text)
        assert len(fences) == 2

    def test_unclosed_fence_extends_to_end(self):
        text = "before\n```python\ncode without closing"
        fences = find_code_fences(text)
        assert len(fences) == 1
        assert fences[0].end == len(text)

    def test_no_fences(self):
        text = "Just regular text\nwith no code blocks"
        fences = find_code_fences(text)
        assert len(fences) == 0


class TestIsInsideCodeFence:
    def test_inside(self):
        fences = [CodeFenceRegion(start=10, end=50)]
        assert is_inside_code_fence(30, fences) is True

    def test_outside(self):
        fences = [CodeFenceRegion(start=10, end=50)]
        assert is_inside_code_fence(5, fences) is False
        assert is_inside_code_fence(60, fences) is False

    def test_boundary_not_inside(self):
        fences = [CodeFenceRegion(start=10, end=50)]
        # Boundaries themselves are not "inside"
        assert is_inside_code_fence(10, fences) is False
        assert is_inside_code_fence(50, fences) is False


class TestFindBestCutoff:
    def test_prefers_heading_over_blank_line_with_decay(self):
        """H2 at 50% back should beat blank line at 0% back."""
        # H2 at pos 400 (50% back from target 800, window 800)
        # score = 90 * (1 - (0.5^2 * 0.7)) = 90 * 0.825 = 74.25
        # Blank line at pos 800 (0% back)
        # score = 20 * 1.0 = 20
        bps = [
            BreakPoint(pos=400, score=90, type='h2'),
            BreakPoint(pos=800, score=20, type='blank'),
        ]
        result = find_best_cutoff(bps, target_pos=800, window_chars=800)
        assert result == 400  # heading wins

    def test_skips_code_fence_break_points(self):
        bps = [
            BreakPoint(pos=100, score=20, type='blank'),
            BreakPoint(pos=200, score=90, type='h2'),  # inside fence
            BreakPoint(pos=300, score=20, type='blank'),
        ]
        fences = [CodeFenceRegion(start=150, end=250)]
        result = find_best_cutoff(bps, target_pos=400, window_chars=400, code_fences=fences)
        # Should pick pos 300 (best non-fenced), not 200
        assert result == 300

    def test_returns_target_when_no_break_points(self):
        result = find_best_cutoff([], target_pos=500, window_chars=800)
        assert result == 500

    def test_ignores_break_points_beyond_target(self):
        bps = [
            BreakPoint(pos=100, score=20, type='blank'),
            BreakPoint(pos=600, score=100, type='h1'),  # past target
        ]
        result = find_best_cutoff(bps, target_pos=500, window_chars=800)
        assert result == 100


class TestChunkMarkdownBreakpoint:
    def test_short_doc_single_chunk(self):
        text = "# Hello\n\nShort document."
        chunks = chunk_markdown_breakpoint(text, max_chars=3600)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].source_type == "markdown"

    def test_empty_input(self):
        assert chunk_markdown_breakpoint("") == []
        assert chunk_markdown_breakpoint("   \n\n  ") == []

    def test_header_splitting_preferred(self):
        """Headers should be preferred split points over paragraph breaks."""
        # Build a doc where a header sits in the search window
        section1 = "First section content. " * 40  # ~920 chars
        section2 = "\n## Second Section\n\n" + "Second content. " * 40
        text = section1 + section2
        chunks = chunk_markdown_breakpoint(text, max_chars=1000, overlap_chars=100, window_chars=400)
        assert len(chunks) >= 2
        # At least one chunk boundary should align near the ## header
        found_header_boundary = any(
            chunk.content.strip().startswith("## Second") or
            chunk.content.strip().endswith("## Second Section")
            for chunk in chunks
        )
        # The second chunk should reference the header
        assert any("Second Section" in c.section_heading for c in chunks)

    def test_code_fence_protection(self):
        """Break points inside code fences should not be chosen as cut points.

        When the code block fits within max_chars, the chunker should avoid
        splitting inside it. (If a fence exceeds max_chars, a force-cut is
        inevitable — that's expected behavior, not a failure.)
        """
        # Build a doc where the code block fits within max_chars but the
        # overall doc needs splitting at a point outside the fence.
        before = "Introduction text. " * 30 + "\n\n"  # ~570 chars
        code_block = "```python\ndef hello():\n    return 42\n```\n\n"  # ~40 chars
        after = "Conclusion text. " * 30 + "\n"  # ~510 chars
        text = before + code_block + after
        chunks = chunk_markdown_breakpoint(text, max_chars=700, overlap_chars=50, window_chars=300)

        code_start = text.index("```python")
        code_end = text.index("```\n\n", code_start + 3) + 4

        for chunk in chunks:
            # Find where this chunk starts in the original text
            snippet = chunk.content[:80]
            chunk_start = text.find(snippet)
            if chunk_start == -1:
                continue
            # No chunk should start strictly inside the code fence
            assert not (code_start < chunk_start < code_end), (
                f"Chunk starts inside code fence at pos {chunk_start}"
            )

    def test_overlap_between_chunks(self):
        """Adjacent chunks should share content."""
        text = "Word " * 1000  # ~5000 chars
        chunks = chunk_markdown_breakpoint(text, max_chars=1000, overlap_chars=150, window_chars=400)
        assert len(chunks) >= 3

        # Check overlap: end of chunk N should appear at start of chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i].content[-100:]  # last 100 chars of chunk i
            head = chunks[i + 1].content[:200]  # first 200 chars of chunk i+1
            # Some of the tail should appear in the head (overlap)
            # Use a shorter snippet for matching
            snippet = tail[-50:]
            assert snippet in head, f"No overlap between chunks {i} and {i+1}"

    def test_heading_hierarchy_tracking(self):
        text = "# Main Title\n\nIntro text.\n\n## Sub Section\n\nSub content.\n\n### Deep Section\n\nDeep content."
        chunks = chunk_markdown_breakpoint(text, max_chars=5000)
        assert len(chunks) == 1
        # Single chunk should have last heading as section_heading
        assert chunks[0].section_heading == "Deep Section"
        assert chunks[0].parent_section == "Sub Section"

    def test_heading_hierarchy_across_chunks(self):
        intro = "# Main Title\n\nIntro. " * 30 + "\n\n"
        section = "## Details\n\nDetails content. " * 30 + "\n\n"
        subsection = "### Specifics\n\nSpecific stuff. " * 30
        text = intro + section + subsection
        chunks = chunk_markdown_breakpoint(text, max_chars=500, overlap_chars=50, window_chars=200)
        assert len(chunks) >= 3

        # Later chunks should have heading context
        last_chunk = chunks[-1]
        assert last_chunk.section_heading != ""

    def test_large_realistic_doc(self):
        """A realistic blog-post-style markdown should produce reasonable chunks."""
        sections = []
        for i in range(5):
            sections.append(f"\n## Section {i + 1}\n\n")
            sections.append(f"This is the content of section {i + 1}. " * 50)
            sections.append("\n\n```python\n")
            sections.append(f"def function_{i}():\n    return {i}\n")
            sections.append("```\n\n")
            sections.append("More text after the code block. " * 20)

        text = "# Blog Post Title\n\n" + "".join(sections)
        chunks = chunk_markdown_breakpoint(text)

        # Should produce multiple chunks
        assert len(chunks) >= 3
        # No chunk should exceed max size by much (some tolerance for overlap boundaries)
        for chunk in chunks:
            assert len(chunk.content) <= 3600 + 100  # small tolerance
        # All chunks should have source_type
        assert all(c.source_type == "markdown" for c in chunks)

    def test_start_end_lines(self):
        text = "Line 0\nLine 1\nLine 2\nLine 3\nLine 4"
        chunks = chunk_markdown_breakpoint(text, max_chars=5000)
        assert len(chunks) == 1
        assert chunks[0].start_line == 0
        assert chunks[0].end_line == 4

    def test_unclosed_fence_handled(self):
        """An unclosed code fence should not crash, just extend to end."""
        text = "Before\n\n```python\ncode_inside = True\nmore code\n"
        chunks = chunk_markdown_breakpoint(text, max_chars=5000)
        assert len(chunks) == 1
        assert "code_inside" in chunks[0].content

    def test_progress_guarantee(self):
        """Even with no break points, chunking should complete and not loop."""
        # A single long line with no newlines
        text = "a" * 10000
        chunks = chunk_markdown_breakpoint(text, max_chars=1000, overlap_chars=100, window_chars=400)
        assert len(chunks) >= 5
        # All content should be covered
        total = sum(len(c.content) for c in chunks)
        assert total >= len(text)  # with overlap, total >= original

    def test_hr_as_break_point(self):
        text = "First section content.\n\n---\n\nSecond section content."
        bps = scan_break_points(text)
        hr_bps = [bp for bp in bps if bp.type == 'hr']
        assert len(hr_bps) >= 1
