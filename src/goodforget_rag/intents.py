"""Simple heuristic intent extraction for non-oracle comparison."""

from __future__ import annotations

import re


NEGATIVE_MARKERS = (
    "without relying on",
    "without using",
    "while avoiding",
    "but do not use",
    "do not use",
    "avoiding",
    "avoid",
)


def heuristic_positive_intent(query: str) -> str:
    """Remove simple negative clauses from a query.

    This intentionally weak heuristic is included to make oracle positive-intent
    assumptions visible. It is not a production query rewriter.
    """

    lowered = query.lower()
    cut_positions = [
        lowered.find(marker)
        for marker in NEGATIVE_MARKERS
        if lowered.find(marker) >= 0
    ]
    if not cut_positions:
        return query.strip()
    return query[: min(cut_positions)].strip(" .,;:")


def heuristic_forget_set(query: str) -> list[str]:
    """Extract simple forget-intent phrases from negative clauses."""

    lowered = query.lower()
    phrases: list[str] = []
    for marker in NEGATIVE_MARKERS:
        pattern = re.compile(re.escape(marker) + r"\s+(.+?)(?:[.;]|\Z)", re.IGNORECASE)
        for match in pattern.finditer(query):
            phrase = match.group(1).strip(" .,;:")
            if phrase:
                phrases.append(phrase)
    if phrases:
        return phrases

    if any(word in lowered for word in ("confidential", "secret", "leaked", "private")):
        return [query]
    return []
