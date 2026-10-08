from __future__ import annotations

import csv
from pathlib import Path

from .dictionary import load_dictionary
from .experiment import load_benchmark
from .retrieval import retrieve


def compare_profiles(
    benchmark_path: str | Path,
    dictionary_path: str | Path,
    output_path: str | Path,
    *,
    threshold: float = 0.55,
    limit: int = 5,
) -> list[dict]:
    tasks = load_benchmark(benchmark_path)
    words = load_dictionary(dictionary_path)
    rows: list[dict] = []
    for task in tasks:
        protected = [x for x in (task.get("protected") or "").split("|") if x]
        for profile in ("template", "balanced", "consonant"):
            plans = retrieve(
                task["target_reading"],
                words,
                profile=profile,
                threshold=threshold,
                limit=limit,
                protected=protected,
            )
            top = plans[0] if plans else None
            rows.append(
                {
                    "task_id": task["id"],
                    "profile": profile,
                    "candidate_count": len(plans),
                    "top_word": top.replacement if top else "",
                    "top_word_reading": top.replacement_reading if top else "",
                    "target_span": top.target_span if top else "",
                    "top_similarity": top.similarity if top else "",
                    "target_consonants": top.target_consonants if top else "",
                    "replacement_consonants": top.replacement_consonants if top else "",
                    "target_vowels": top.target_vowels if top else "",
                    "replacement_vowels": top.replacement_vowels if top else "",
                }
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows
