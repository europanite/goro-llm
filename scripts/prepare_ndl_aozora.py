#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from goro_llm.research_data import NDL_AOZORA_URL, iter_ndl_aozora_pairs, write_ndl_jsonl


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch/convert NDL Aozora furigana corpus")
    p.add_argument("--archive", type=Path, default=Path("external_data/ndl/aozora_dataset.zip"))
    p.add_argument("--extract-dir", type=Path, default=Path("external_data/ndl/aozora_dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("data/ndl_reading"))
    p.add_argument("--url", default=NDL_AOZORA_URL)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--group-by-file", action="store_true",
                   help="stricter split than the 2024 paper: keep each work in one split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    if not args.archive.exists():
        if args.no_download:
            p.error(f"archive not found: {args.archive}")
        print(f"downloading {args.url}")
        download(args.url, args.archive)
    if not args.extract_dir.exists():
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(args.extract_dir)
    counts = write_ndl_jsonl(
        iter_ndl_aozora_pairs(args.extract_dir), args.output_dir,
        seed=args.seed, group_by_file=args.group_by_file, limit=args.limit,
    )
    print(counts)


if __name__ == "__main__":
    main()
