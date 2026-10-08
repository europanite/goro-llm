from __future__ import annotations

import csv
import json
from pathlib import Path

from .dictionary import load_dictionary
from .generator import generate_llm_only, generate_with_plan, overgenerate
from .rank import rank_candidates
from .retrieval import retrieve


def load_benchmark(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def run_experiment(
    benchmark_path: str | Path,
    dictionary_path: str | Path,
    output_path: str | Path,
    *,
    profile: str = "consonant",
    threshold: float = 0.55,
    dry_run: bool = False,
    overgenerate_count: int = 8,
) -> list[dict]:
    tasks = load_benchmark(benchmark_path)
    words = load_dictionary(dictionary_path)
    rows: list[dict] = []

    for task in tasks:
        protected = [x for x in (task.get("protected") or "").split("|") if x]
        plans = [
            p.to_dict()
            for p in retrieve(
                task["target_reading"],
                words,
                profile=profile,
                threshold=threshold,
                limit=12,
                protected=protected,
            )
        ]
        common = {
            "task_id": task["id"],
            "target": task["target"],
            "fact": task["fact"],
            "target_reading": task["target_reading"],
            "profile": profile,
            "retrieval_plans": plans,
        }

        if dry_run:
            rows.append({**common, "system": "retrieval_only", "output": None})
            continue

        llm = generate_llm_only(task["target"], task["fact"], task["target_reading"])
        rows.append({**common, "system": "llm_only", "output": llm})

        if plans:
            guided = generate_with_plan(
                task["target"], task["fact"], task["target_reading"], plans[0]
            )
            rows.append({**common, "system": "retrieval_top1", "output": guided})

            generated = overgenerate(
                task["target"],
                task["fact"],
                task["target_reading"],
                plans,
                count=overgenerate_count,
            )
            ranked = rank_candidates(generated, plans)
            rows.append(
                {
                    **common,
                    "system": "overgenerate_rank",
                    "output": ranked[0] if ranked else None,
                    "all_candidates": ranked,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
