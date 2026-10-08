from __future__ import annotations

from dataclasses import dataclass
import unicodedata

SMALL = "ゃゅょぁぃぅぇぉゎ"
SMALL_VOWELS = dict(zip(SMALL, "auoaiueoa"))

FEATURES: dict[str, tuple[str, str]] = {}
for consonant, chars in [
    ("", "あいうえお"),
    ("k", "かきくけこ"),
    ("g", "がぎぐげご"),
    ("s", "さしすせそ"),
    ("z", "ざじずぜぞ"),
    ("t", "たちつてと"),
    ("d", "だぢづでど"),
    ("n", "なにぬねの"),
    ("h", "はひふへほ"),
    ("b", "ばびぶべぼ"),
    ("p", "ぱぴぷぺぽ"),
    ("m", "まみむめも"),
    ("r", "らりるれろ"),
]:
    for kana, vowel in zip(chars, "aiueo"):
        FEATURES[kana] = (consonant, vowel)

FEATURES.update(
    {
        "し": ("sh", "i"),
        "じ": ("j", "i"),
        "ち": ("ch", "i"),
        "ぢ": ("j", "i"),
        "つ": ("ts", "u"),
        "づ": ("z", "u"),
        "ふ": ("f", "u"),
        "や": ("y", "a"),
        "ゆ": ("y", "u"),
        "よ": ("y", "o"),
        "わ": ("w", "a"),
        "を": ("", "o"),
        "ゐ": ("w", "i"),
        "ゑ": ("w", "e"),
        "ゔ": ("v", "u"),
        "ん": ("N", "N"),
        "っ": ("Q", "Q"),
    }
)

CONSONANT_FAMILIES = [
    {"k", "g"},
    {"s", "z"},
    {"sh", "j", "ch"},
    {"t", "d", "ts"},
    {"h", "f", "b", "p", "v"},
]


@dataclass(frozen=True)
class PhoneticWeights:
    """Heuristic substitution weights.

    template: reproduces the supplied prototype's vowel-heavy weighting.
    consonant: research hypothesis that emphasizes consonant similarity.
    """

    vowel: float
    consonant: float
    related_consonant_multiplier: float = 0.35


PROFILES = {
    "template": PhoneticWeights(vowel=0.58, consonant=0.42),
    "balanced": PhoneticWeights(vowel=0.50, consonant=0.50),
    "consonant": PhoneticWeights(vowel=0.35, consonant=0.65),
}


def normalize_kana(value: str) -> str:
    s = unicodedata.normalize("NFKC", value).strip()
    s = "".join(chr(ord(c) - 96) if "ァ" <= c <= "ヶ" else c for c in s)
    if not s:
        raise ValueError("reading is empty")
    invalid = [c for c in s if c not in FEATURES and c not in SMALL and c != "ー"]
    if invalid:
        raise ValueError(f"reading must be kana/long mark only: {value!r}; invalid={invalid[:3]}")
    return s


def split_mora(value: str) -> list[str]:
    result: list[str] = []
    for c in normalize_kana(value):
        if c in SMALL and result and result[-1] not in {"っ", "ん", "ー"}:
            result[-1] += c
        else:
            result.append(c)
    return result


def features(moras: list[str]) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for mora in moras:
        if mora == "ー":
            vowel = output[-1][1] if output and output[-1][1] in "aiueo" else "LONG"
            output.append(("", vowel))
            continue
        onset, vowel = FEATURES[mora[0]]
        if len(mora) > 1:
            vowel = SMALL_VOWELS[mora[-1]]
            if mora[-1] in "ゃゅょ" and onset not in {"sh", "j", "ch"}:
                onset += "y"
        output.append((onset, vowel))
    return output


def vowel_string(value: str) -> str:
    return "".join(v for _, v in features(split_mora(value)) if v in "aiueo")


def consonant_string(value: str) -> str:
    return "-".join(c or "∅" for c, _ in features(split_mora(value)))


def _same_family(a: str, b: str) -> bool:
    return any(a in group and b in group for group in CONSONANT_FAMILIES)


def replacement_cost(
    a: tuple[str, str], b: tuple[str, str], weights: PhoneticWeights
) -> float:
    if a == b:
        return 0.0
    if a[1] in {"N", "Q", "LONG"} or b[1] in {"N", "Q", "LONG"}:
        return 1.0
    if a[0] == b[0]:
        onset_cost = 0.0
    elif _same_family(a[0], b[0]):
        onset_cost = weights.related_consonant_multiplier
    else:
        onset_cost = 1.0
    return weights.vowel * (a[1] != b[1]) + weights.consonant * onset_cost


def similarity_features(
    a: list[tuple[str, str]],
    b: list[tuple[str, str]],
    weights: PhoneticWeights,
) -> float:
    if not a or not b:
        return 0.0
    row = list(map(float, range(len(b) + 1)))
    for i, x in enumerate(a, 1):
        new = [float(i)]
        for j, y in enumerate(b, 1):
            new.append(
                min(
                    new[-1] + 1,
                    row[j] + 1,
                    row[j - 1] + replacement_cost(x, y, weights),
                )
            )
        row = new
    return max(0.0, 1.0 - row[-1] / max(len(a), len(b)))


def similarity(a: str, b: str, profile: str = "consonant") -> float:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}; choose from {sorted(PROFILES)}")
    return similarity_features(features(split_mora(a)), features(split_mora(b)), PROFILES[profile])
