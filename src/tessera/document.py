"""Document extraction and chunking module.

Supports extraction and chunking of various document formats:
- PDF: Extract text using pymupdf4llm
- Markdown: Split on headers with hierarchy tracking
- YAML: Split by top-level keys with safe loading
- JSON: Split by top-level keys with safe loading

All exceptions are wrapped in DocumentExtractionError for uniform error handling.
"""

import asyncio
import json
import yaml
from dataclasses import dataclass


class DocumentExtractionError(Exception):
    """Raised when document extraction fails."""
    pass


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata."""
    content: str
    source_type: str  # e.g. 'markdown', 'pdf', 'yaml', 'json'
    source_file: str = ""
    section_heading: str = ""
    key_path: str = ""
    page_number: int | None = None
    parent_section: str = ""
    start_line: int = 0
    end_line: int = 0


async def extract_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using pymupdf4llm.

    CRITICAL: Uses asyncio.to_thread() to avoid blocking the event loop.
    The extraction is run in a thread pool, not via asyncio.run().

    Args:
        pdf_path: Path to PDF file

    Returns:
        Markdown text extracted from PDF

    Raises:
        DocumentExtractionError: If extraction fails
    """
    try:
        import pymupdf4llm
    except ImportError:
        raise DocumentExtractionError(
            "pymupdf4llm not installed. Install with: pip install pymupdf4llm"
        )

    try:
        # Use asyncio.to_thread to run blocking I/O without blocking event loop
        markdown_text = await asyncio.to_thread(
            pymupdf4llm.to_markdown, pdf_path
        )
        return markdown_text
    except Exception as e:
        raise DocumentExtractionError(
            f"Failed to extract PDF from {pdf_path}: {e}"
        ) from e


def chunk_markdown(
    markdown_text: str,
    max_chunk_size: int = 1024,
    overlap: int = 128,
    split_headers: list[str] | None = None,
) -> list[DocumentChunk]:
    """
    Chunk markdown text on header boundaries with hierarchy tracking.

    Splits on lines starting with header patterns (default: "#", "##", "###").
    Maintains section hierarchy (h1 -> h2 -> h3) and tracks parent_section.
    Chunks exceeding max_chunk_size are split with overlap.

    Args:
        markdown_text: Markdown text to chunk
        max_chunk_size: Maximum chunk size in characters (default 1024)
        overlap: Character overlap for large chunks (default 128)
        split_headers: Header patterns to split on (default ["#", "##", "###"])

    Returns:
        List of DocumentChunk objects
    """
    if split_headers is None:
        split_headers = ["#", "##", "###"]

    lines = markdown_text.split("\n")
    chunks = []
    current_chunk_lines = []
    current_heading = ""
    parent_heading = ""
    start_line_idx = 0
    header_stack = {}  # Track header hierarchy: {level: header_text}

    for line_idx, line in enumerate(lines):
        # Check if line is a header
        is_header = False
        header_level = None
        header_text = ""

        for header_pattern in split_headers:
            if line.startswith(header_pattern + " "):
                is_header = True
                header_level = len(header_pattern)
                header_text = line[len(header_pattern) :].strip()
                break

        if is_header:
            # Flush current chunk if it has content
            if current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines)
                if len(chunk_content) > max_chunk_size:
                    # Split large chunks with overlap
                    chunks.extend(
                        _split_large_chunk(
                            chunk_content,
                            current_heading,
                            parent_heading,
                            start_line_idx,
                            line_idx - 1,
                            max_chunk_size,
                            overlap,
                        )
                    )
                else:
                    chunks.append(
                        DocumentChunk(
                            content=chunk_content,
                            source_type="markdown",
                            section_heading=current_heading,
                            parent_section=parent_heading,
                            start_line=start_line_idx,
                            end_line=line_idx - 1,
                        )
                    )
                current_chunk_lines = []

            # Update header tracking
            if header_level == 1:
                header_stack = {1: header_text}
                current_heading = header_text
                parent_heading = ""
            elif header_level == 2:
                header_stack = {
                    k: v for k, v in header_stack.items() if k < 2
                }
                header_stack[2] = header_text
                current_heading = header_text
                parent_heading = header_stack.get(1, "")
            elif header_level == 3:
                header_stack = {
                    k: v for k, v in header_stack.items() if k < 3
                }
                header_stack[3] = header_text
                current_heading = header_text
                parent_heading = header_stack.get(2, header_stack.get(1, ""))

            start_line_idx = line_idx
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)

    # Flush remaining chunk
    if current_chunk_lines:
        chunk_content = "\n".join(current_chunk_lines)
        if len(chunk_content) > max_chunk_size:
            chunks.extend(
                _split_large_chunk(
                    chunk_content,
                    current_heading,
                    parent_heading,
                    start_line_idx,
                    len(lines) - 1,
                    max_chunk_size,
                    overlap,
                )
            )
        else:
            chunks.append(
                DocumentChunk(
                    content=chunk_content,
                    source_type="markdown",
                    section_heading=current_heading,
                    parent_section=parent_heading,
                    start_line=start_line_idx,
                    end_line=len(lines) - 1,
                )
            )

    return chunks


def _split_large_chunk(
    content: str,
    section_heading: str,
    parent_section: str,
    start_line: int,
    end_line: int,
    max_chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """
    Split a large chunk into overlapping pieces.

    Args:
        content: Content to split
        section_heading: Section heading for this chunk
        parent_section: Parent section name
        start_line: Starting line number
        end_line: Ending line number
        max_chunk_size: Maximum size for each piece
        overlap: Character overlap between pieces

    Returns:
        List of DocumentChunk objects
    """
    chunks = []
    offset = 0
    line_offset = start_line

    while offset < len(content):
        # Take up to max_chunk_size characters
        end_offset = min(offset + max_chunk_size, len(content))
        chunk_text = content[offset:end_offset]

        # Count newlines to estimate line number
        lines_in_chunk = chunk_text.count("\n")
        chunk_end_line = line_offset + lines_in_chunk

        chunks.append(
            DocumentChunk(
                content=chunk_text,
                source_type="markdown",
                section_heading=section_heading,
                parent_section=parent_section,
                start_line=line_offset,
                end_line=chunk_end_line,
            )
        )

        # Move offset by (max_chunk_size - overlap) for next chunk
        offset = end_offset - overlap
        line_offset = chunk_end_line

        # Stop if we've reached the end
        if end_offset >= len(content):
            break

    return chunks


def chunk_yaml(
    yaml_path: str, max_chunk_size: int = 2048
) -> list[DocumentChunk]:
    """
    Chunk YAML file by top-level keys.

    Uses safe_load for security. Chunks by top-level keys with recursive
    splitting if content exceeds max_chunk_size.

    Args:
        yaml_path: Path to YAML file
        max_chunk_size: Maximum chunk size in characters (default 2048)

    Returns:
        List of DocumentChunk objects

    Raises:
        DocumentExtractionError: If YAML parsing fails
    """
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise DocumentExtractionError(
            f"YAML file not found: {yaml_path}"
        ) from e
    except yaml.YAMLError as e:
        raise DocumentExtractionError(
            f"Failed to parse YAML file {yaml_path}: {e}"
        ) from e
    except Exception as e:
        raise DocumentExtractionError(
            f"Failed to read YAML file {yaml_path}: {e}"
        ) from e

    if data is None:
        return []

    if not isinstance(data, dict):
        # If root is not a dict, wrap the entire content
        content = yaml.dump(data, default_flow_style=False)
        return [
            DocumentChunk(
                content=content,
                source_type="yaml",
                source_file=yaml_path,
                key_path="",
                parent_section="",
                start_line=0,
                end_line=len(content.split("\n")) - 1,
            )
        ]

    chunks = []
    for key, value in data.items():
        # Dump this key-value pair to YAML
        content = yaml.dump({key: value}, default_flow_style=False)

        # Check if chunk exceeds max_chunk_size
        if len(content) > max_chunk_size:
            # Recursively chunk nested structures
            sub_chunks = _chunk_yaml_nested(
                {key: value}, max_chunk_size, key, ""
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                DocumentChunk(
                    content=content,
                    source_type="yaml",
                    source_file=yaml_path,
                    key_path=key,
                    parent_section="",
                    start_line=0,
                    end_line=len(content.split("\n")) - 1,
                )
            )

    return chunks


def _chunk_yaml_nested(
    data: dict, max_chunk_size: int, key_path: str, parent_section: str
) -> list[DocumentChunk]:
    """
    Recursively chunk nested YAML structures.

    Args:
        data: Dict to chunk (single key-value pair)
        max_chunk_size: Maximum chunk size
        key_path: Current key path (dot-separated)
        parent_section: Parent section name

    Returns:
        List of DocumentChunk objects
    """
    chunks = []
    for key, value in data.items():
        current_key_path = f"{key_path}.{key}" if key_path else key

        if isinstance(value, dict):
            # Recursively process nested dict
            for nested_key, nested_value in value.items():
                nested_content = yaml.dump(
                    {nested_key: nested_value}, default_flow_style=False
                )
                if len(nested_content) > max_chunk_size:
                    # Further recursion
                    chunks.extend(
                        _chunk_yaml_nested(
                            {nested_key: nested_value},
                            max_chunk_size,
                            current_key_path,
                            key_path,
                        )
                    )
                else:
                    chunks.append(
                        DocumentChunk(
                            content=nested_content,
                            source_type="yaml",
                            key_path=f"{current_key_path}.{nested_key}",
                            parent_section=key_path,
                            start_line=0,
                            end_line=len(nested_content.split("\n")) - 1,
                        )
                    )
        else:
            # Leaf value
            content = yaml.dump({key: value}, default_flow_style=False)
            chunks.append(
                DocumentChunk(
                    content=content,
                    source_type="yaml",
                    key_path=current_key_path,
                    parent_section=parent_section,
                    start_line=0,
                    end_line=len(content.split("\n")) - 1,
                )
            )

    return chunks


def chunk_json(
    json_path: str, max_chunk_size: int = 2048
) -> list[DocumentChunk]:
    """
    Chunk JSON file by top-level keys.

    Uses json.loads() for safe parsing. Chunks by top-level keys with
    recursive splitting if content exceeds max_chunk_size.

    Args:
        json_path: Path to JSON file
        max_chunk_size: Maximum chunk size in characters (default 2048)

    Returns:
        List of DocumentChunk objects

    Raises:
        DocumentExtractionError: If JSON parsing fails
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise DocumentExtractionError(
            f"JSON file not found: {json_path}"
        ) from e
    except json.JSONDecodeError as e:
        raise DocumentExtractionError(
            f"Failed to parse JSON file {json_path}: {e}"
        ) from e
    except Exception as e:
        raise DocumentExtractionError(
            f"Failed to read JSON file {json_path}: {e}"
        ) from e

    if data is None:
        return []

    if not isinstance(data, dict):
        # If root is not a dict, wrap the entire content
        content = json.dumps(data, indent=2)
        return [
            DocumentChunk(
                content=content,
                source_type="json",
                source_file=json_path,
                key_path="",
                parent_section="",
                start_line=0,
                end_line=len(content.split("\n")) - 1,
            )
        ]

    chunks = []
    for key, value in data.items():
        # Dump this key-value pair to JSON
        content = json.dumps({key: value}, indent=2)

        # Check if chunk exceeds max_chunk_size
        if len(content) > max_chunk_size:
            # Recursively chunk nested structures
            sub_chunks = _chunk_json_nested(
                {key: value}, max_chunk_size, key, ""
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                DocumentChunk(
                    content=content,
                    source_type="json",
                    source_file=json_path,
                    key_path=key,
                    parent_section="",
                    start_line=0,
                    end_line=len(content.split("\n")) - 1,
                )
            )

    return chunks


def _chunk_json_nested(
    data: dict, max_chunk_size: int, key_path: str, parent_section: str
) -> list[DocumentChunk]:
    """
    Recursively chunk nested JSON structures.

    Args:
        data: Dict to chunk (single key-value pair)
        max_chunk_size: Maximum chunk size
        key_path: Current key path (dot-separated)
        parent_section: Parent section name

    Returns:
        List of DocumentChunk objects
    """
    chunks = []
    for key, value in data.items():
        current_key_path = f"{key_path}.{key}" if key_path else key

        if isinstance(value, dict):
            # Recursively process nested dict
            for nested_key, nested_value in value.items():
                nested_content = json.dumps(
                    {nested_key: nested_value}, indent=2
                )
                if len(nested_content) > max_chunk_size:
                    # Further recursion
                    chunks.extend(
                        _chunk_json_nested(
                            {nested_key: nested_value},
                            max_chunk_size,
                            current_key_path,
                            key_path,
                        )
                    )
                else:
                    chunks.append(
                        DocumentChunk(
                            content=nested_content,
                            source_type="json",
                            key_path=f"{current_key_path}.{nested_key}",
                            parent_section=key_path,
                            start_line=0,
                            end_line=len(nested_content.split("\n")) - 1,
                        )
                    )
        else:
            # Leaf value
            content = json.dumps({key: value}, indent=2)
            chunks.append(
                DocumentChunk(
                    content=content,
                    source_type="json",
                    key_path=current_key_path,
                    parent_section=parent_section,
                    start_line=0,
                    end_line=len(content.split("\n")) - 1,
                )
            )

    return chunks
