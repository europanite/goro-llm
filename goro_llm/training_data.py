from __future__ import annotations

READING_TO_KANA = (
    "日本語の文章を入力します。入力した文章の読み方をひらがなで出力してください。"
)
KANA_TO_READING = (
    "ひらがなのみで書かれた文章を入力します。これらを漢字を含めた自然な日本語の文章に書き換えてください。"
)
PUN_PROMPT = (
    "あなたは日本語の扱いに長けた高度な人工知能です。特に日本語の文章を入力として、"
    "その意味やニュアンスをできるだけ維持したまま面白い駄洒落に変換することが得意です。"
    "以下に示す文の意味を変えずに、駄洒落を交えた文に書き換えてください。"
)


def reading_messages(row: dict, reverse: bool = False) -> list[dict[str, str]]:
    if reverse:
        return [
            {"role": "user", "content": f"{KANA_TO_READING}\n{row['reading']}"},
            {"role": "assistant", "content": row["text"]},
        ]
    return [
        {"role": "user", "content": f"{READING_TO_KANA}\n{row['text']}"},
        {"role": "assistant", "content": row["reading"]},
    ]


def pun_sft_messages(row: dict) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"{PUN_PROMPT}\n{row['input']}"},
        {"role": "assistant", "content": row["output"]},
    ]


def dpo_row(row: dict) -> dict[str, str]:
    return {
        "prompt": f"{PUN_PROMPT}\n{row['prompt']}",
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }
