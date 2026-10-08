from goro_llm.research_data import (
    ReadingPair,
    assign_split,
    character_edit_distance,
    grouped_split,
    parse_ndl_aozora_text,
    prepare_dpo_pairs,
    prepare_pun_sft,
)


def test_parse_ndl_block():
    sample = """行番号: 3
名前はまだ無い。
なまえわ まだ ない。
名前
なまえ
漢字
は
わ
ひらがな
行番号: 4
猫である。
ねこである。
猫
ねこ
漢字
"""
    rows = parse_ndl_aozora_text(sample, "wagahai.txt")
    assert [r.text for r in rows] == ["名前はまだ無い。", "猫である。"]
    assert rows[0].reading == "なまえわまだない。"
    assert rows[0].line_number == 3


def test_split_is_deterministic_90_5_5_space():
    row = ReadingPair("名前はまだ無い。", "なまえわまだない。", "x.txt", 3)
    assert assign_split(row, seed=42) == assign_split(row, seed=42)
    assert assign_split(row, seed=42) in {"train", "validation", "test"}


def test_pun_filter_and_group_split_keeps_group_together():
    rows = [
        {"pun": "布団が吹っ飛んだ", "paraphrase": "布団が飛んだ", "rating": "4"},
        {"pun": "布団が吹っ飛んだ", "paraphrase": "布団が風で飛ばされた", "rating": "4"},
        {"pun": "低評価", "paraphrase": "低い", "rating": "2"},
    ]
    prepared = prepare_pun_sft(rows, pun_col="pun", paraphrase_col="paraphrase",
                               rating_col="rating", min_rating=2.5)
    assert len(prepared) == 2
    splits = grouped_split(prepared, seed=1)
    locations = [name for name, items in splits.items() if items]
    assert len(locations) == 1
    assert len(splits[locations[0]]) == 2


def test_prepare_dpo_pairs_human_is_chosen():
    human = [{"input": "布団が飛んだ", "output": "布団が吹っ飛んだ"}]
    generated = [{"input": "布団が飛んだ", "output": "布団が飛んで布団だ"}]
    pairs = prepare_dpo_pairs(human, generated)
    assert pairs == [{
        "prompt": "布団が飛んだ",
        "chosen": "布団が吹っ飛んだ",
        "rejected": "布団が飛んで布団だ",
    }]


def test_minami_pair_filters():
    rows = [
        {"pun": "布団が吹っ飛んだ", "paraphrase": "強風で寝具が空へ飛ばされました",
         "rating": "4", "sim": "0.82"},
        {"pun": "近い文", "paraphrase": "近い文だ", "rating": "4", "sim": "0.91"},
        {"pun": "意味違い", "paraphrase": "十分に長く異なる文章です", "rating": "4", "sim": "0.30"},
    ]
    assert character_edit_distance("abc", "adc") == 1
    prepared = prepare_pun_sft(
        rows, pun_col="pun", paraphrase_col="paraphrase",
        rating_col="rating", min_rating=2.5,
        semantic_similarity_col="sim", min_semantic_similarity=0.7,
        min_edit_distance=7,
    )
    assert len(prepared) == 1
    assert prepared[0]["edit_distance"] >= 7
