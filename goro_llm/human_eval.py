from __future__ import annotations

import csv
import json
from pathlib import Path


def make_eval_sheet(results_jsonl: str | Path, output_csv: str | Path) -> int:
    records = []
    with Path(results_jsonl).open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            output = row.get("output") or {}
            if not output.get("text"):
                continue
            records.append(
                {
                    "task_id": row["task_id"],
                    "system": row["system"],
                    "target": row["target"],
                    "fact": row["fact"],
                    "mnemonic": output["text"],
                    "naturalness_1_5": "",
                    "memorability_1_5": "",
                    "fact_recall_1_5": "",
                    "fact_correct_0_1": "",
                    "notes": "",
                }
            )
    fields = list(records[0].keys()) if records else [
        "task_id", "system", "target", "fact", "mnemonic",
        "naturalness_1_5", "memorability_1_5", "fact_recall_1_5",
        "fact_correct_0_1", "notes"
    ]
    with Path(output_csv).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return len(records)
