from goro_llm.phonetics import consonant_string, similarity, split_mora, vowel_string


def test_normalization_and_mora():
    assert split_mora("ｷｬｯﾄ") == ["きゃ", "っ", "と"]
    assert vowel_string("コーヒー") == "ooii"
    assert consonant_string("きす").startswith("k-")


def test_similarity_profiles():
    assert similarity("きす", "きす") == 1.0
    assert similarity("きす", "きず", profile="consonant") > similarity("きす", "かす", profile="consonant")
    assert similarity("こー", "こお") == 1.0
