#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from goro_llm.research_data import prepare_dpo_pairs, read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Pair human puns (chosen) with SFT generations (rejected)")
    p.add_argument("--human", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/pun_dpo/train.jsonl"))
    args = p.parse_args()
    rows = prepare_dpo_pairs(read_jsonl(args.human), read_jsonl(args.generated))
    print(f"wrote {write_jsonl(rows, args.output)} pairs to {args.output}")


if __name__ == "__main__":
    main()
