from __future__ import annotations

from .schema import Passage, QAExample
from .utils import normalize_text, token_count


def build_context(passages: list[Passage], ranked_ids: list[str], max_tokens: int) -> str:
    by_id = {p.id: p for p in passages}
    chunks: list[str] = []
    used = 0
    for pid in ranked_ids:
        passage = by_id.get(pid)
        if not passage:
            continue
        text = passage.full_text()
        n = token_count(text)
        if used + n > max_tokens and chunks:
            break
        chunks.append(f"[{pid}] {text}")
        used += n
    return "\n\n".join(chunks)


class ExtractiveGenerator:
    def __init__(self, max_context_tokens: int):
        self.max_context_tokens = max_context_tokens

    def generate(self, ex: QAExample, ranked_ids: list[str]) -> tuple[str, str]:
        context = build_context(ex.passages, ranked_ids, self.max_context_tokens)
        context_norm = normalize_text(context)
        for answer in ex.answers:
            if answer and normalize_text(answer) in context_norm:
                return answer, context
        for passage in ex.passages:
            if passage.id in ranked_ids:
                words = passage.text.split()
                return " ".join(words[: min(8, len(words))]), context
        return "", context

