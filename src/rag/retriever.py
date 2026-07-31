"""
Very simple keyword-based retriever over the FAQ markdown files.

This is intentionally NOT using embeddings or a vector database (no
OpenAI/Chroma/Pinecone in Version 1). Instead it splits each FAQ file
into sections (by "## " headings) and scores sections by how many
query words they share. This keeps the RAG concept clear without any
external dependencies.
"""

import re
from dataclasses import dataclass

from src.config import FAQ_DIR


@dataclass
class FaqSection:
    source: str      # filename, e.g. "shipping.md"
    heading: str      # section heading, e.g. "How long does shipping take?"
    text: str         # section body text


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def load_faq_sections() -> list[FaqSection]:
    """Read every .md file in data/faq and split it into sections."""
    sections: list[FaqSection] = []

    if not FAQ_DIR.exists():
        return sections

    for path in sorted(FAQ_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")

        # Split on markdown "## " headings, keeping the heading text
        parts = re.split(r"^## (.+)$", content, flags=re.MULTILINE)

        # parts[0] is the top-level title/intro before the first "##"; skip it
        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append(FaqSection(source=path.name, heading=heading, text=body))

    return sections


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """
    Return up to `top_k` FAQ snippets most relevant to `query`, using
    simple word-overlap scoring between the query and each section's
    heading + body.
    """
    query_words = _tokenize(query)
    if not query_words:
        return []

    scored: list[tuple[int, FaqSection]] = []
    for section in load_faq_sections():
        section_words = _tokenize(section.heading + " " + section.text)
        overlap = len(query_words & section_words)
        if overlap > 0:
            scored.append((overlap, section))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    evidence = []
    for _, section in scored[:top_k]:
        evidence.append(f"[{section.source}] {section.heading}: {section.text}")

    return evidence
