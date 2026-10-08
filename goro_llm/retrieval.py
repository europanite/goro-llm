from __future__ import annotations

from dataclasses import asdict, dataclass

from .dictionary import DictionaryWord
from .phonetics import consonant_string, similarity, split_mora, vowel_string


@dataclass(frozen=True)
class RetrievalPlan:
    plan_id: str
    word_id: int
    replacement: str
    replacement_reading: str
    start: int
    end: int
    target_span: str
    similarity: float
    target_vowels: str
    replacement_vowels: str
    target_consonants: str
    replacement_consonants: str
    profile: str

    def to_dict(self) -> dict:
        return asdict(self)


def _blocked_indices(target_morae: list[str], protected: list[str]) -> set[int]:
    blocked: set[int] = set()
    for item in protected:
        lock = split_mora(item)
        for i in range(len(target_morae) - len(lock) + 1):
            if target_morae[i : i + len(lock)] == lock:
                blocked.update(range(i, i + len(lock)))
    return blocked


def retrieve(
    target_reading: str,
    words: list[DictionaryWord],
    *,
    profile: str = "consonant",
    threshold: float = 0.55,
    limit: int = 12,
    protected: list[str] | None = None,
) -> list[RetrievalPlan]:
    target = split_mora(target_reading)
    blocked = _blocked_indices(target, protected or [])
    plans: list[RetrievalPlan] = []

    for word in words:
        wm = split_mora(word.reading)
        if not 2 <= len(wm) <= 16:
            continue
        best: RetrievalPlan | None = None
        min_size = max(1, len(wm) - 1)
        max_size = min(len(target), len(wm) + 1)
        for size in range(min_size, max_size + 1):
            for start in range(len(target) - size + 1):
                end = start + size
                if blocked.intersection(range(start, end)):
                    continue
                span = "".join(target[start:end])
                score = similarity(span, word.reading, profile=profile)
                if score < threshold:
                    continue
                plan = RetrievalPlan(
                    plan_id=f"w{word.id}-m{start}-{end}",
                    word_id=word.id,
                    replacement=word.surface,
                    replacement_reading=word.reading,
                    start=start,
                    end=end,
                    target_span=span,
                    similarity=round(score, 4),
                    target_vowels=vowel_string(span),
                    replacement_vowels=vowel_string(word.reading),
                    target_consonants=consonant_string(span),
                    replacement_consonants=consonant_string(word.reading),
                    profile=profile,
                )
                if best is None or (plan.similarity, -abs(size - len(wm))) > (
                    best.similarity,
                    -abs((best.end - best.start) - len(wm)),
                ):
                    best = plan
        if best:
            plans.append(best)

    plans.sort(key=lambda p: (-p.similarity, p.word_id, p.start))
    # Prefer distinct dictionary words by construction; each word contributes one plan.
    return plans[:limit]
