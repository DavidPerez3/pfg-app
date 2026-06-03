from __future__ import annotations

from collections import Counter
from typing import Any


REFINEMENT_MARKERS = (
    "more ",
    "less ",
    "another ",
    "something ",
    "make it ",
    "make them ",
    "but ",
    "instead",
    "without ",
    "with more ",
    "with less ",
    "focus on ",
    "only ",
    "prefer ",
    "not so ",
    "similar but ",
)


def is_recommendation_refinement(text: str) -> bool:
    normalized = " ".join(text.lower().strip().split())
    if not normalized:
        return False
    return any(marker in normalized for marker in REFINEMENT_MARKERS)


def build_hitl_refinement_context(
    *,
    latest_user_prompt: str | None,
    latest_assistant_text: str | None,
    latest_feedback: dict[str, Any] | None,
) -> str | None:
    parts: list[str] = []

    if latest_user_prompt and latest_user_prompt.strip():
        parts.append(f"Previous recommendation request: {latest_user_prompt.strip()}")

    if latest_assistant_text and latest_assistant_text.strip():
        parts.append(
            "Previous assistant recommendation summary: "
            + latest_assistant_text.strip().replace("\n", " ")
        )

    if latest_feedback:
        rating = latest_feedback.get("rating")
        comment = str(latest_feedback.get("comment") or "").strip()
        if isinstance(rating, int):
            if rating <= 2:
                parts.append(
                    "The user marked the previous recommendation as needing work. "
                    "Adjust the direction more strongly and avoid repeating the same angle."
                )
            elif rating >= 4:
                parts.append(
                    "The user marked the previous recommendation as useful. "
                    "Preserve the overall direction unless the new request explicitly changes it."
                )
        if comment:
            parts.append(f"Stored feedback note: {comment}")

    if not parts:
        return None

    parts.append(
        "Treat the current message as a refinement of the previous recommendation, "
        "not as an unrelated request."
    )
    return "\n".join(parts)


def build_follow_up_prompts(
    *,
    items: list[dict[str, Any]],
    cold_start: bool,
) -> list[str]:
    suggestions = [
        "Give me a more mainstream version of these recommendations",
        "Give me a more niche version of these recommendations",
        "Give me a more recent version of these recommendations",
    ]

    genre_counter: Counter[str] = Counter()
    for item in items[:5]:
        for raw_genre in str(item.get("genres") or "").split("|"):
            genre = raw_genre.strip()
            if genre:
                genre_counter[genre] += 1

    for genre, _ in genre_counter.most_common(2):
        suggestions.append(f"Refine this list toward more {genre.lower()} items")

    if cold_start:
        suggestions.append("Use a safer and broader recommendation strategy")
    else:
        suggestions.append("Keep the same profile but make the list more diverse")

    deduped: list[str] = []
    for suggestion in suggestions:
        if suggestion not in deduped:
            deduped.append(suggestion)
    return deduped[:5]
