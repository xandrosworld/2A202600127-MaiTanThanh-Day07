from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on sentence boundaries: ". ", "! ", "? ", or ".\n"
        sentences = re.split(r'(?<=[.!?]) +|(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(' '.join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: fits within chunk_size
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separators left — force character-level split
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        # Empty-string separator: character-level split
        if sep == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        parts = current_text.split(sep)
        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split(part, next_seps))
        return result if result else [current_text]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed = FixedSizeChunker(chunk_size=chunk_size).chunk(text)
        by_sentences = SentenceChunker().chunk(text)
        recursive = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0.0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": stats(fixed),
            "by_sentences": stats(by_sentences),
            "recursive": stats(recursive),
        }


class HeaderAwareChunker:
    """
    Custom chunker for Markdown policy documents.

    Strategy: split on Markdown headers (lines starting with '#').
    Each chunk = one complete section, with the header kept at the top.
    If a section exceeds max_chunk_size, it is further split on blank lines
    (paragraphs) to avoid oversized embeddings.

    Additional quality filters (controlled by min_chunk_length):
        - Chunks shorter than min_chunk_length are discarded (likely bare headers).
        - Chunks whose non-empty lines consist almost entirely of URLs / metadata
          markers are discarded (e.g. "Source: https://..." only sections).

    Design rationale:
        VinUni policy documents are authored with a strict header hierarchy
        (##, ###). Each section expresses one coherent policy clause.
        Splitting at header boundaries preserves semantic completeness —
        a chunk always contains exactly one policy idea — which significantly
        improves retrieval precision compared to fixed-size or sentence-based
        chunking that can cut across clause boundaries.
    """

    def __init__(self, max_chunk_size: int = 1000, min_chunk_length: int = 80) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_length = min_chunk_length

    def _is_low_quality(self, text: str) -> bool:
        """Return True if the chunk contains too little real content."""
        if len(text) < self.min_chunk_length:
            return True
        # Count lines that carry real content (not just URLs / metadata markers)
        content_lines = [
            line for line in text.splitlines()
            if line.strip() and not line.strip().startswith(
                ("Source:", "http", "**VINUNIVERSITY**", "==>", "Title:")
            )
        ]
        return len(content_lines) < 2

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        lines = text.splitlines(keepends=True)
        sections: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#") and current:
                # New header found → save previous section, start new one
                sections.append(current)
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append(current)

        chunks: list[str] = []
        for section_lines in sections:
            section_text = "".join(section_lines).strip()
            if not section_text:
                continue
            if len(section_text) <= self.max_chunk_size:
                if not self._is_low_quality(section_text):
                    chunks.append(section_text)
            else:
                # Section too large → split further on blank lines (paragraphs)
                paragraphs = re.split(r"\n{2,}", section_text)
                for para in paragraphs:
                    para = para.strip()
                    if para and not self._is_low_quality(para):
                        chunks.append(para)

        return chunks


