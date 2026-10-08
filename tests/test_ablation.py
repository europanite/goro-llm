from goro_llm.ablation import compare_profiles


def test_profile_ablation_writes_all_three_profiles(tmp_path):
    output = tmp_path / "ablation.csv"
    rows = compare_profiles("data/benchmark.csv", "data/dictionary.csv", output)
    assert output.exists()
    assert len(rows) == 12
    assert {row["profile"] for row in rows} == {"template", "balanced", "consonant"}
