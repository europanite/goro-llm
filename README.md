# Japanese Goro LLM Research

A small, reproducible research repository for **Japanese 語呂合わせ / mnemonic generation with local LLMs**.

The core idea is deliberately hybrid:

> **Python searches for phonetically useful Japanese anchor words; the LLM turns those anchors into short memorable cues.**

This repo was built from an existing local `goro` prototype that already used kana normalization, mora segmentation, weighted phonetic edit distance, CSV dictionaries, and Ollama. The research version makes the assumptions explicit and adds baselines, ablations, experiment output, and human-evaluation scaffolding.

## Why this is a research problem

Recent work already shows that adjacent tasks are viable: Japanese pun generation has used furigana-aware language models and RLHF, and FIT 2026 includes LLM + reinforcement learning for Japanese number mnemonics. LLM mnemonic research also shows that “generate many, then rank” is useful, while student preference does not necessarily equal actual learning effectiveness. See [`docs/literature.md`](docs/literature.md).

The specific question here is narrower:

> **Does explicit Japanese phonetic retrieval—especially consonant-aware retrieval—improve the quality of LLM-generated study mnemonics?**

## Repository layout

```text
japanese-goro-llm/
├── data/
│   ├── benchmark.csv
│   └── dictionary.csv
├── configs/
│   ├── minami2024_reading.json
│   ├── minami2024_pun_sft.json
│   ├── gemma2_jpn_sft_3090.json
│   ├── gemma2_jpn_dpo_3090.json
│   └── smart_dpo_adapter.json
├── docs/
│   ├── datasets-frameworks.md
│   ├── experiment-design.md
│   ├── literature.md
│   └── references.bib
├── external_data/
│   └── README.md
├── goro_llm/
│   ├── cli.py
│   ├── dictionary.py
│   ├── experiment.py
│   ├── generator.py
│   ├── human_eval.py
│   ├── ollama.py
│   ├── phonetics.py
│   ├── rank.py
│   ├── research_data.py
│   ├── retrieval.py
│   └── training_data.py
├── scripts/
│   ├── fetch_smart.py
│   ├── prepare_dpo_pairs.py
│   ├── prepare_ndl_aozora.py
│   └── prepare_pun_data.py
├── training/
│   ├── train_dpo.py
│   └── train_sft.py
├── prompts/
├── tests/
├── compose.yml
├── compose.gpu.yml
└── pyproject.toml
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

## 2. Test phonetic retrieval without an LLM

```bash
goro-retrieve ひくきすたす \
  --dictionary data/dictionary.csv \
  --profile consonant \
  --threshold 0.55
```

Three phonetic profiles are included:

- `template`: vowel 0.58 / consonant 0.42 (the supplied prototype baseline)
- `balanced`: 0.50 / 0.50
- `consonant`: vowel 0.35 / consonant 0.65 (research hypothesis)

The weights are **heuristics**, not published fitted parameters.

## 3. Compare phonetic profiles without an LLM

```bash
goro-profile-ablation \
  --benchmark data/benchmark.csv \
  --dictionary data/dictionary.csv \
  --output results/profile_ablation.csv
```

This is the first ablation to run: it shows whether changing the consonant/vowel weights actually changes retrieval before LLM variability is introduced.

## 4. Dry-run the benchmark

This creates retrieval plans only and requires no model.

```bash
goro-experiment \
  --benchmark data/benchmark.csv \
  --dictionary data/dictionary.csv \
  --profile consonant \
  --dry-run \
  --output results/dry-run.jsonl
```

## 5. Run with a local Ollama model

Start Ollama directly, or with Docker:

```bash
docker compose -f compose.yml -f compose.gpu.yml up -d
```

Pull a model using your normal Ollama workflow, then set its installed name:

```bash
export GORO_MODEL='<your-installed-model>'
export GORO_OLLAMA_URL='http://127.0.0.1:11434'

goro-experiment \
  --profile consonant \
  --output results/consonant.jsonl
```

The experiment runs three conditions when retrieval is available:

1. `llm_only`
2. `retrieval_top1`
3. `overgenerate_rank`


## Research datasets and training frameworks

The repository now includes adapters for research data and runnable LoRA/SFT/DPO scaffolding. See [`docs/datasets-frameworks.md`](docs/datasets-frameworks.md).

### NDL furigana corpus (Minami et al. 2024)

```bash
python scripts/prepare_ndl_aozora.py
python training/train_sft.py configs/minami2024_reading.json
```

### Authorized Japanese pun/paraphrase data

```bash
python scripts/prepare_pun_data.py local_puns.csv \
  --pun-col pun --paraphrase-col paraphrase \
  --rating-col rating --min-rating 2.5 \
  --min-edit-distance 7
python training/train_sft.py configs/gemma2_jpn_sft_3090.json
```

For closer Minami-2024 data filtering, add `--semantic-sim-col <column> --min-semantic-sim 0.7` when your authorized table already contains the paper-style semantic cosine score.

### DPO pairs: human pun preferred, SFT generation rejected

```bash
python scripts/prepare_dpo_pairs.py \
  --human data/pun_sft/train.jsonl \
  --generated results/sft_generations.jsonl
python training/train_dpo.py configs/gemma2_jpn_dpo_3090.json
```

### SMART public mnemonic data

```bash
pip install -e '.[research]'
python scripts/fetch_smart.py
```

External corpora are downloaded into ignored paths and are not silently redistributed.

## 6. Create a human-evaluation sheet

```bash
goro-make-eval results/consonant.jsonl \
  --output results/human_eval.csv
```

Rate naturalness, memorability, factual recall support, and factual correctness. For a serious memory claim, use an actual delayed-recall experiment; perceived usefulness alone is not enough.

## Design inherited from the supplied prototype

The supplied project already normalized kana, split Japanese into morae, represented each mora with consonant/vowel features, used weighted edit distance, and treated similar consonant families such as `k/g` as closer than arbitrary substitutions. It also used a local Ollama endpoint and CSV dictionary storage. This repository keeps those ideas but turns them into explicit research variables rather than a single fixed implementation.

## Important limitations

- This is a **baseline research scaffold**, not evidence that the method improves memory.
- Pitch accent, devoicing, detailed acoustic distance, and full Japanese phonology are not modeled.
- The included starter dictionary is intentionally tiny; serious experiments need a larger, licensed dictionary.
- The automatic ranker is transparent but crude. It mainly rewards phonetic fidelity and anchor use.
- LLM judges can be added, but they should not replace blinded human evaluation.
- The benchmark is illustrative. Expand it before making statistical claims.

## Research references

See [`docs/literature.md`](docs/literature.md) for the full survey and links. Key starting points:

- Kawahara & Shinohara (2009), Japanese imperfect puns and psychoacoustic similarity.
- Minami et al. (JSAI 2024), furigana-aware Japanese pun generation.
- Lee et al. (Findings of EMNLP 2024), overgenerate-and-rank mnemonic generation.
- Balepur et al. (EMNLP 2024), student-feedback-aligned mnemonic generation.
- Nishihara & Ichikawa (FIT 2026), LLM + RL for Japanese number mnemonics.

## License

MIT. Literature and external datasets retain their own licenses; this repository does not redistribute them.
