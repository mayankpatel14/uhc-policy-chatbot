"""Prompt construction, context formatting, and chunk deduplication."""

from chatbot.config import MAX_CONTEXT_CHARS, MAX_CHUNK_CHARS
from chatbot.retriever import ChunkResult

SYSTEM_PROMPT = """\
You are a UHC medical policy assistant. Users are doctors and hospital staff.

RULES:
1. Answer ONLY from the policy excerpts below. No outside knowledge.
2. Be BRIEF: 2-4 bullet points max. One short paragraph for summary if needed.
3. Cite sources as (policy-name, Section). Example: (bariatric-surgery, Coverage Rationale).
4. If context lacks the answer, say "I don't have enough policy information to answer this."
5. If something is "unproven and not medically necessary," say it is NOT covered.
6. Do NOT repeat or paraphrase the same point multiple times. State each fact once.
7. Coverage Rationale is the authoritative source for what IS and IS NOT covered. \
If Coverage Rationale says a treatment is unproven/not medically necessary for a condition, \
clearly state it is NOT covered — even if Clinical Evidence discusses studies about it."""


def deduplicate_chunks(chunks: list[ChunkResult]) -> list[ChunkResult]:
    """Keep the highest-scoring chunk per (policy_name, section) pair."""
    seen: dict[tuple[str, str], ChunkResult] = {}
    for c in chunks:
        key = (c.policy_name, c.section)
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    return sorted(seen.values(), key=lambda c: c.score, reverse=True)


def _truncate_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    """Truncate chunk text to max_chars, breaking at sentence boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    if last_period > max_chars // 2:
        return cut[:last_period + 1]
    return cut + "..."


def format_context(chunks: list[ChunkResult]) -> str:
    """
    Render retrieved chunks into a numbered context block for the LLM.
    Each chunk is truncated to MAX_CHUNK_CHARS, total capped at MAX_CONTEXT_CHARS.
    """
    deduped = deduplicate_chunks(chunks)

    parts = []
    char_count = 0

    for i, c in enumerate(deduped, 1):
        header = f"[{i}] {c.policy_name} | {c.section} | {c.plan_type}"
        body = _truncate_text(c.text)
        block = f"{header}\n{body}\n"

        if char_count + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        char_count += len(block)

    return "\n---\n".join(parts)


def build_messages(
    query: str,
    context: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """Build the chat messages list with system prompt, history, and context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    user_content = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer briefly in 2-4 bullet points with citations."
    )
    messages.append({"role": "user", "content": user_content})

    return messages
