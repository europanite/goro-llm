from __future__ import annotations

import math


def length_score(text: str) -> float:
    # Broad preference for compact cues; intentionally weak and transparent.
    n = len(text.strip())
    if n == 0:
        return 0.0
    center = 24
    return math.exp(-abs(n - center) / 28)


def score_candidate(candidate: dict, plans: dict[str, dict]) -> dict:
    plan_id = candidate.get("plan_id")
    plan = plans.get(plan_id) if plan_id else None
    phonetic = float(plan["similarity"]) if plan else 0.0
    anchor = 1.0 if plan and plan["replacement"] in candidate.get("text", "") else 0.0
    concise = length_score(candidate.get("text", ""))
    # Primary metric is phonetic fidelity; surface heuristics only break close ties.
    score = 0.75 * phonetic + 0.15 * anchor + 0.10 * concise
    return {
        **candidate,
        "rank_score": round(score, 4),
        "phonetic_score": round(phonetic, 4),
        "anchor_present": bool(anchor),
        "length_score": round(concise, 4),
    }


def rank_candidates(candidates: list[dict], plans: list[dict]) -> list[dict]:
    plan_map = {p["plan_id"]: p for p in plans}
    scored = [score_candidate(c, plan_map) for c in candidates]
    return sorted(scored, key=lambda c: (-c["rank_score"], c["text"]))
