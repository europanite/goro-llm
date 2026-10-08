from __future__ import annotations

import argparse
import json

from .ablation import compare_profiles
from .dictionary import load_dictionary
from .experiment import run_experiment
from .human_eval import make_eval_sheet
from .retrieval import retrieve


def retrieve_main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve Japanese words with similar phonetics")
    parser.add_argument("reading")
    parser.add_argument("--dictionary", default="data/dictionary.csv")
    parser.add_argument("--profile", choices=["template", "balanced", "consonant"], default="consonant")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    words = load_dictionary(args.dictionary)
    plans = retrieve(args.reading, words, profile=args.profile, threshold=args.threshold, limit=args.limit)
    print(json.dumps([p.to_dict() for p in plans], ensure_ascii=False, indent=2))


def experiment_main() -> None:
    parser = argparse.ArgumentParser(description="Run Japanese mnemonic baselines")
    parser.add_argument("--benchmark", default="data/benchmark.csv")
    parser.add_argument("--dictionary", default="data/dictionary.csv")
    parser.add_argument("--output", default="results/results.jsonl")
    parser.add_argument("--profile", choices=["template", "balanced", "consonant"], default="consonant")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--overgenerate-count", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = run_experiment(
        args.benchmark,
        args.dictionary,
        args.output,
        profile=args.profile,
        threshold=args.threshold,
        dry_run=args.dry_run,
        overgenerate_count=args.overgenerate_count,
    )
    print(f"wrote {len(rows)} rows to {args.output}")


def eval_main() -> None:
    parser = argparse.ArgumentParser(description="Create a human-evaluation CSV")
    parser.add_argument("results")
    parser.add_argument("--output", default="results/human_eval.csv")
    args = parser.parse_args()
    n = make_eval_sheet(args.results, args.output)
    print(f"wrote {n} items to {args.output}")


def ablation_main() -> None:
    parser = argparse.ArgumentParser(description="Compare phonetic weighting profiles without an LLM")
    parser.add_argument("--benchmark", default="data/benchmark.csv")
    parser.add_argument("--dictionary", default="data/dictionary.csv")
    parser.add_argument("--output", default="results/profile_ablation.csv")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    rows = compare_profiles(
        args.benchmark, args.dictionary, args.output,
        threshold=args.threshold, limit=args.limit,
    )
    print(f"wrote {len(rows)} rows to {args.output}")
