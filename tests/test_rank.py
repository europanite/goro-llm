from goro_llm.rank import rank_candidates


def test_anchor_and_phonetics_rank():
    plans = [
        {"plan_id": "a", "similarity": 0.95, "replacement": "キス"},
        {"plan_id": "b", "similarity": 0.70, "replacement": "カス"},
    ]
    candidates = [
        {"text": "キスで覚える", "explanation": "", "plan_id": "a"},
        {"text": "カスで覚える", "explanation": "", "plan_id": "b"},
    ]
    ranked = rank_candidates(candidates, plans)
    assert ranked[0]["plan_id"] == "a"
