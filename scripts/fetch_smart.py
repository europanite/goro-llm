#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from goro_llm.research_data import SMART_DATASETS


def main() -> None:
    p = argparse.ArgumentParser(description="Download public SMART mnemonic datasets from Hugging Face")
    p.add_argument("--output-dir", type=Path, default=Path("external_data/smart"))
    p.add_argument("--dataset", choices=["all", *SMART_DATASETS], default="all")
    args = p.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install research extras: pip install -e '.[research]'") from exc

    selected = SMART_DATASETS if args.dataset == "all" else {args.dataset: SMART_DATASETS[args.dataset]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for short, repo in selected.items():
        ds = load_dataset(repo)
        manifest[short] = {"repo": repo, "splits": {}}
        for split, table in ds.items():
            path = args.output_dir / f"{short}-{split}.jsonl"
            table.to_json(path, force_ascii=False)
            manifest[short]["splits"][split] = {"rows": len(table), "path": str(path)}
            print(f"{repo}:{split} -> {path} ({len(table)} rows)")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
