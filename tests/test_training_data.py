from goro_llm.training_data import dpo_row, pun_sft_messages, reading_messages


def test_reading_messages_both_directions():
    row = {"text": "名前はまだ無い。", "reading": "なまえわまだない。"}
    assert reading_messages(row)[1]["content"] == row["reading"]
    assert reading_messages(row, reverse=True)[1]["content"] == row["text"]


def test_pun_and_dpo_templates():
    sft = pun_sft_messages({"input": "布団が飛んだ", "output": "布団が吹っ飛んだ"})
    assert sft[1]["content"] == "布団が吹っ飛んだ"
    dpo = dpo_row({"prompt": "布団が飛んだ", "chosen": "A", "rejected": "B"})
    assert dpo["chosen"] == "A" and dpo["rejected"] == "B"
