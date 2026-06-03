from __future__ import annotations

import re


def normalize_fact(fact: str) -> str:
    return " ".join((fact or "").strip().split())


def extract_candidate_facts(text: str) -> list[str]:
    text = " ".join((text or "").strip().split())
    if not text:
        return []

    candidate_patterns = [
        r"\bI like\b[^.?!]*",
        r"\bI love\b[^.?!]*",
        r"\bI prefer\b[^.?!]*",
        r"\bmy favorite\b[^.?!]*",
        r"\bI enjoy\b[^.?!]*",
        r"\bI don't like\b[^.?!]*",
        r"\bI do not like\b[^.?!]*",
        r"\bI hate\b[^.?!]*",
    ]
    facts: list[str] = []
    seen: set[str] = set()
    for pattern in candidate_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            normalized = normalize_fact(match)
            if len(normalized) < 8 or normalized in seen:
                continue
            seen.add(normalized)
            facts.append(normalized)
    return facts
