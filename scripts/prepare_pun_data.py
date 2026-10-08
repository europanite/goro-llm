#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from goro_llm.research_data import grouped_split, load_tabular, prepare_pun_sft, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Convert an authorized pun/paraphrase table to leakage-safe SFT JSONL")
    p.add_argument("input", type=Path)
    p.add_argument("--pun-col", default="pun")
    p.add_argument("--paraphrase-col", default="paraphrase")
    p.add_argument("--rating-col")
    p.add_argument("--min-rating", type=float)
    p.add_argument("--semantic-sim-col", help="column containing a precomputed semantic cosine similarity")
    p.add_argument("--min-semantic-sim", type=float)
    p.add_argument("--min-edit-distance", type=int)
    p.add_argument("--group-col")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("data/pun_sft"))
    args = p.parse_args()
    rows = prepare_pun_sft(
        load_tabular(args.input), pun_col=args.pun_col, paraphrase_col=args.paraphrase_col,
        rating_col=args.rating_col, min_rating=args.min_rating,
        semantic_similarity_col=args.semantic_sim_col,
        min_semantic_similarity=args.min_semantic_sim,
        min_edit_distance=args.min_edit_distance, group_col=args.group_col,
    )
    splits = grouped_split(rows, seed=args.seed)
    for split, part in splits.items():
        path = args.output_dir / f"{split}.jsonl"
        print(split, write_jsonl(part, path), path)


if __name__ == "__main__":
    main()
