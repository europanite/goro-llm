from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

KANA_RE = re.compile(r"^[ぁ-ゖァ-ヶー\s、。！？!?・,.…（）()「」『』\-0-9]+$")
KANJI_RE = re.compile(r"[一-龯々〆ヵヶ]")
LINE_NO_RE = re.compile(r"^行番号\s*[:：]\s*(\d+)\s*$")

NDL_AOZORA_URL = "https://lab.ndl.go.jp/dataset/huriganacorpus/aozora_dataset.zip"
SMART_DATASETS = {
    "sft": "nbalepur/Mnemonic_SFT",
    "preferences": "nbalepur/Mnemonic_Pref",
    "chosen_rejected": "nbalepur/Mnemonic_Chosen_Rejected",
    "test": "nbalepur/Mnemonic_Test",
}


@dataclass(frozen=True)
class ReadingPair:
    text: str
    reading: str
    source_file: str
    line_number: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_reading(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def _looks_like_pair(text: str, reading: str) -> bool:
    if not text or not reading or not KANJI_RE.search(text):
        return False
    return bool(KANA_RE.fullmatch(reading))


def parse_ndl_aozora_text(content: str, source_file: str = "") -> list[ReadingPair]:
    """Parse NDL Aozora furigana corpus files into sentence/readings.

    The corpus stores a marker line ("行番号: N"), then the full input sentence,
    then the full reading, followed by token-level annotations. Only the first
    two payload lines are needed for the reading fine-tuning task used by Minami
    et al. (JSAI 2024).
    """
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[ReadingPair] = []
    i = 0
    while i < len(lines):
        match = LINE_NO_RE.match(lines[i].strip())
        if not match:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break
        text = lines[j].strip()
        j += 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break
        reading = _clean_reading(lines[j])
        if _looks_like_pair(text, reading):
            out.append(ReadingPair(text=text, reading=reading, source_file=source_file,
                                   line_number=int(match.group(1))))
        i = j + 1
    return out


def iter_ndl_aozora_pairs(root: str | Path) -> Iterator[ReadingPair]:
    root = Path(root)
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")
        rel = str(path.relative_to(root))
        yield from parse_ndl_aozora_text(text, rel)


def _stable_bucket(value: str, seed: int = 42) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(pair: ReadingPair, *, seed: int = 42, group_by_file: bool = False) -> str:
    """Return train/validation/test with a 90/5/5 deterministic split.

    `group_by_file=False` mirrors the pair-level split described in Minami et al.
    (2024). `group_by_file=True` is a stricter leakage-resistant alternative.
    """
    key = pair.source_file if group_by_file else f"{pair.source_file}:{pair.line_number}:{pair.text}"
    x = _stable_bucket(key, seed)
    if x < 0.90:
        return "train"
    if x < 0.95:
        return "validation"
    return "test"


def write_ndl_jsonl(
    pairs: Iterable[ReadingPair], output_dir: str | Path, *, seed: int = 42,
    group_by_file: bool = False, limit: int | None = None,
) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: (output_dir / f"{s}.jsonl").open("w", encoding="utf-8")
               for s in ("train", "validation", "test")}
    counts = {s: 0 for s in handles}
    try:
        for idx, pair in enumerate(pairs):
            if limit is not None and idx >= limit:
                break
            split = assign_split(pair, seed=seed, group_by_file=group_by_file)
            row = pair.to_dict() | {"split": split}
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for h in handles.values():
            h.close()
    return counts


def load_tabular(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else None
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            delimiter = ","
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]


def character_edit_distance(a: str, b: str) -> int:
    """Plain character-level Levenshtein distance used for pair filtering."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def prepare_pun_sft(
    rows: Iterable[dict[str, str]], *, pun_col: str, paraphrase_col: str,
    rating_col: str | None = None, min_rating: float | None = None,
    semantic_similarity_col: str | None = None,
    min_semantic_similarity: float | None = None,
    min_edit_distance: int | None = None, group_col: str | None = None,
) -> list[dict]:
    """Convert a licensed/local pun-paraphrase table to SFT examples.

    This intentionally does not fetch or redistribute the Araki/Hokkaido pun DB.
    It only transforms a copy the researcher is authorized to use.
    """
    out = []
    for i, row in enumerate(rows):
        pun = (row.get(pun_col) or "").strip()
        paraphrase = (row.get(paraphrase_col) or "").strip()
        if not pun or not paraphrase:
            continue
        if rating_col and min_rating is not None:
            try:
                if float(row.get(rating_col, "nan")) < min_rating:
                    continue
            except (TypeError, ValueError):
                continue
        if min_semantic_similarity is not None:
            if not semantic_similarity_col:
                raise ValueError("semantic_similarity_col is required with min_semantic_similarity")
            try:
                if float(row.get(semantic_similarity_col, "nan")) < min_semantic_similarity:
                    continue
            except (TypeError, ValueError):
                continue
        edit_distance = character_edit_distance(pun, paraphrase)
        if min_edit_distance is not None and edit_distance < min_edit_distance:
            continue
        out.append({
            "id": str(i),
            "input": paraphrase,
            "output": pun,
            "group_id": (row.get(group_col) or pun).strip() if group_col else pun,
            "task": "pun_paraphrase_sft",
            "edit_distance": edit_distance,
        })
    return out


def grouped_split(rows: list[dict], *, seed: int = 42,
                  ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)) -> dict[str, list[dict]]:
    """Leakage-resistant split: all rows with the same group_id stay together."""
    if abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("ratios must sum to 1")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("group_id", row.get("id", ""))), []).append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    result = {"train": [], "validation": [], "test": []}
    for key in keys:
        x = _stable_bucket(key, seed)
        split = "train" if x < ratios[0] else "validation" if x < ratios[0] + ratios[1] else "test"
        result[split].extend(groups[key])
    return result


def prepare_dpo_pairs(human_rows: Iterable[dict], generated_rows: Iterable[dict], *,
                      key: str = "input", human_output: str = "output",
                      generated_output: str = "output") -> list[dict]:
    """Pair human puns (chosen) with SFT model generations (rejected)."""
    human = {}
    for row in human_rows:
        if row.get(key) and row.get(human_output):
            human.setdefault(str(row[key]), []).append(str(row[human_output]))
    out = []
    for row in generated_rows:
        prompt = str(row.get(key, "")).strip()
        rejected = str(row.get(generated_output, "")).strip()
        if not prompt or not rejected or prompt not in human:
            continue
        for chosen in human[prompt]:
            if chosen.strip() and chosen.strip() != rejected:
                out.append({"prompt": prompt, "chosen": chosen.strip(), "rejected": rejected})
    return out


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
