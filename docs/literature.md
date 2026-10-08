# Literature survey: Japanese puns, 語呂合わせ, and LLM mnemonics

This repository separates three related but non-identical tasks:

1. **語呂合わせ / mnemonic cue generation**: produce an easy-to-recall phrase tied to a fact, number, formula, or sequence.
2. **駄洒落 / pun generation**: create humorous text exploiting homophony or phonetic similarity.
3. **Keyword mnemonics**: connect a difficult target term to a simpler sound-alike keyword and a memorable association.

The literature below is used as design evidence, not as a claim that all three tasks are equivalent.

## Japanese phonetic and pun research

| Year | Work | Relevance to this repository |
|---|---|---|
| 2002 | T. Yokogawa, *Generation of Japanese puns based on similarity of articulation* | Early Japanese pun generator based on articulatory sound similarity; motivates explicit phonetic features. https://doi.org/10.1109/nafips.2001.944423 |
| 2009 | Kawahara & Shinohara, *The role of psychoacoustic similarity in Japanese puns: A corpus study* | Corpus evidence that imperfect Japanese puns prefer psychoacoustically similar consonants. https://doi.org/10.1017/S0022226708005537 |
| 2010 | *品詞による文評価を用いた日本語語呂自動生成手法* | Japanese mnemonic generation aimed at memorability and recall of original items; useful pre-LLM baseline perspective. https://ipsj.ixsq.nii.ac.jp/records/140161 |
| 2016 | Yatsu & Araki, *A Method for Detecting Japanese Puns using a Support Vector Machine and Consonantal Phonetic Similar Features* | Consonantal phonetic-similarity features improved Japanese pun detection over bag-of-words baseline. https://doi.org/10.3156/jsoft.28.875 |
| 2024 | Minami et al., *Generating Japanese Puns via Paraphrasing Using a Language Model Fine-Tuned with Furigana-Annotated Corpus* | Fine-tuning with furigana/phonetic information reported a BLEU improvement of about 0.03 over the comparison model without the reading-oriented pretraining. https://doi.org/10.11517/pjsai.JSAI2024.0_2G5GS603 |
| 2024 | Tsukami & Quan, *強化学習を用いた駄洒落の自動生成* | Japanese GPT-2 + RLHF/PPO; reported pun-generation rate around 10% versus 1% for the reference LM on 500 seed words. https://www.ieice.org/publications/conferences/summary.php?ConfCd=F&conf_type=F&expandable=2&id=FIT0000016632&lecture_number=E-026&session_num=5e&year=2024 |
| 2025 | Mibayashi, Yamamoto & Ohshima, *モーラの類似性と単語の生成確率を考慮したライムフレーズ生成* | Combines mora similarity with language-model generation probability for Japanese rhyme phrase generation; supports treating phonetic fit and linguistic fluency as separate objectives. https://rerank-lab.org/publication/ |
| 2025 | Wang, Yamamoto & Ohshima, *音素の類似性による対話型駄洒落の生成* | Uses phoneme similarity for interactive Japanese pun generation, reinforcing the value of an explicit sound-similarity layer. https://yamamotolab.net/publications/ |
| 2025 | Minami et al., *学習済み言語モデルからの生成データを活用した文駄洒落化の性能改善* | SFT followed by DPO using human puns as preferred examples and model outputs as dispreferred examples; the paper reports improved pun-like quality and human-evaluation gains. https://doi.org/10.1527/tjsai.40-5_C-P44 |
| 2025 | Takigawa et al., *生成系AIを用いた数学における時代に即した語呂合わせ教育手法に関する考察* | Directly connects generative AI, mathematics education, and mnemonic methods. Public bibliographic information is limited, so this repository does not infer experimental details that are not visible in the record. https://jglobal.jst.go.jp/detail?JGLOBAL_ID=202602287234172932 |
| 2026 | Nishihara & Ichikawa, *強化学習を用いた自然な数字語呂合わせの自動生成* | Explicitly combines LLMs and reinforcement learning for natural Japanese number mnemonics; confirms this is now an active LLM research direction. https://www.ipsj.or.jp/event/fit/fit2026/abstract/data/html/program/e.html |

## LLM mnemonic research

| Year | Work | Relevance |
|---|---|---|
| 2024 | Lee, McNichols & Lan, *Exploring Automated Keyword Mnemonics Generation with Large Language Models via Overgenerate-and-Rank* | Generate many cues, then rank using psycholinguistic measures and pilot-study findings. This motivates the `overgenerate_rank` condition. https://aclanthology.org/2024.findings-emnlp.316/ |
| 2024 | Balepur et al., *A SMART Mnemonic Sounds like “Glue Tonic”* | Uses student feedback and DPO. Important warning: expressed preference and observed learning effectiveness can disagree. https://aclanthology.org/2024.emnlp-main.786/ |
| 2025 | Lee, Scarlatos & Lan, *Interpretable Mnemonic Generation for Kanji Learning via Expectation-Maximization* | Japanese-learning mnemonic generation with explicit interpretable construction rules, though focused on kanji decomposition rather than sound-based 語呂合わせ. https://aclanthology.org/2025.emnlp-main.1294/ |

## Research gap used here

A useful, testable gap is not “LLMs have never generated Japanese wordplay.” They have. The narrower gap for this project, based on the literature search above, is:

> **Can an explicit Japanese phonetic retriever, especially one emphasizing consonant similarity, improve sound fidelity and human-rated recall usefulness of local-LLM-generated study mnemonics compared with LLM-only generation?**

I did not find a public, reproducible benchmark that directly compares LLM-only generation against explicit Japanese phonetic retrieval for general study mnemonics. That is a search finding, not proof that no such work exists.

The repository therefore keeps phonetic retrieval transparent, exposes multiple weight profiles, and treats human memory performance as a separate evaluation target rather than assuming that a fluent or funny output is a good mnemonic.


BibTeX for the main cited works is available in [`references.bib`](references.bib).

## Reproducibility details now wired into the code

The repository now encodes the following paper-grounded pieces rather than only citing them:

- NDL Aozora furigana corpus parser and 90/5/5 split for the reading task.
- CALM2-7B-Chat LoRA configuration (`r=16`, `alpha=64`) with query/key, embedding, and LM-head targets and the paper's 20% warmup + cosine schedule.
- Leakage-resistant converter for local pun/paraphrase data and the 2.5/5 rating threshold used in Minami et al. 2024.
- SFT -> DPO pair construction in which a human pun is `chosen` and an SFT-generated pun is `rejected`, matching Minami et al. 2025 at the data-structure level.
- Public SMART datasets from Hugging Face for SFT, student preferences, Bayesian chosen/rejected pairs, and testing.
- 4-bit QLoRA configs intended for a single 24 GB GPU, kept explicitly separate from paper-reproduction settings.

See [`datasets-frameworks.md`](datasets-frameworks.md) for commands and caveats.
