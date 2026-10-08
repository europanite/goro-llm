from goro_llm.dictionary import DictionaryWord
from goro_llm.retrieval import retrieve

WORDS = [
    DictionaryWord(1, "キス", "きす"),
    DictionaryWord(2, "傷", "きず"),
    DictionaryWord(3, "カス", "かす"),
]


def test_retrieve_exact_span():
    plans = retrieve("ひくきすたす", WORDS, profile="consonant", threshold=0.5)
    assert plans
    assert plans[0].replacement == "キス"
    assert plans[0].target_span == "きす"
    assert plans[0].similarity == 1.0


def test_protected_span_blocks_replacement():
    plans = retrieve("ひくきすたす", WORDS, protected=["きす"], threshold=0.8)
    assert all(p.target_span != "きす" for p in plans)
