from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .phonetics import normalize_kana


@dataclass(frozen=True)
class DictionaryWord:
    id: int
    surface: str
    reading: str
    category: str = "general"
    source: str = "user"


def load_dictionary(path: str | Path) -> list[DictionaryWord]:
    rows: list[DictionaryWord] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"surface", "reading"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("dictionary CSV needs surface,reading columns")
        seen: set[tuple[str, str]] = set()
        for index, raw in enumerate(reader, 1):
            surface = (raw.get("surface") or "").strip()
            reading = normalize_kana(raw.get("reading") or "")
            if not surface:
                raise ValueError(f"empty surface at row {index + 1}")
            key = (surface, reading)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                DictionaryWord(
                    id=int(raw.get("id") or index),
                    surface=surface,
                    reading=reading,
                    category=(raw.get("category") or "general").strip(),
                    source=(raw.get("source") or "user").strip(),
                )
            )
    return rows
