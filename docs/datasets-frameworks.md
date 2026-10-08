# Research datasets and frameworks integrated in this repository

This page distinguishes **paper-derived reproduction settings**, **public external datasets**, and **local/practical adaptations**.

## 1. NDL Aozora furigana corpus

**Used in:** Minami et al., JSAI 2024.

The paper used the NDL furigana-annotated Aozora corpus to teach a Japanese LLM bidirectional reading conversion before pun fine-tuning. It reports 640,449 sentence/readings pairs split 90/5/5.

This repository provides:

```bash
python scripts/prepare_ndl_aozora.py
```

The script downloads the official NDL ZIP, extracts the corpus, parses the documented `行番号 -> sentence -> full reading -> token annotations` format, and emits:

```text
data/ndl_reading/train.jsonl
data/ndl_reading/validation.jsonl
data/ndl_reading/test.jsonl
```

By default it uses a deterministic pair-level 90/5/5 split. `--group-by-file` keeps a whole literary work in one split and is stricter than the split described in the 2024 paper.

The external corpus is **not committed** to this repository. NDL's repository identifies the work as free of known copyright restrictions, but users should still read the upstream license and notes before redistribution.

## 2. Araki / Hokkaido pun database adapter

**Used in:** Minami et al. 2024/2025 and related Japanese pun work.

The 2024 paper describes a pun database of about 67,000 entries. For its pun-paraphrase training set, it selected 25,672 puns with average human rating >= 2.5/5, generated five non-pun paraphrases per pun, filtered for semantic similarity and surface difference, and retained 79,960 pairs.

This repository does **not** bundle that database because a redistribution grant was not verified during this build. If you have an authorized local copy or a derived table, convert it with:

```bash
python scripts/prepare_pun_data.py local_puns.csv \
  --pun-col pun \
  --paraphrase-col paraphrase \
  --rating-col rating \
  --min-rating 2.5 \
  --semantic-sim-col semantic_similarity \
  --min-semantic-sim 0.7 \
  --min-edit-distance 7 \
  --output-dir data/pun_sft
```

Rows sharing the same pun stay in the same split to prevent train/test leakage. The converter can reproduce the paper's character-edit-distance threshold (`>= 7`) and, when your authorized table contains a precomputed semantic-cosine column, its semantic-similarity threshold (`>= 0.7`). The repository does not call the paper's proprietary embedding API automatically.

## 3. SMART mnemonic datasets

**Used in:** Balepur et al., EMNLP 2024.

The authors publicly released four Hugging Face datasets:

- `nbalepur/Mnemonic_SFT`
- `nbalepur/Mnemonic_Pref`
- `nbalepur/Mnemonic_Chosen_Rejected`
- `nbalepur/Mnemonic_Test`

Fetch them with:

```bash
pip install -e '.[research]'
python scripts/fetch_smart.py
```

SMART is English GRE-vocabulary mnemonic research, not Japanese 語呂合わせ. It is included because its **preference-learning design** is directly useful here: the paper collected 2,684 student preferences from 45 students and showed that expressed preference and observed learning effectiveness can disagree.

Use SMART data as a framework/evaluation reference, not as evidence about Japanese phonetics.

## 4. CALM2 + LoRA reading/pun pipeline

**Used in:** Minami et al., JSAI 2024.

The paper used `cyberagent/calm2-7b-chat` and LoRA. For the reading stage it reports:

- LoRA rank `r=16`
- LoRA alpha `64`
- LoRA on attention query/key plus embedding and LM head (PEFT targets `q_proj`, `k_proj`, `embed_tokens`, `lm_head`)
- AdamW
- maximum learning rate `1e-4`
- 20% warmup followed by cosine decay
- effective batch size 20
- 1 epoch
- extra KL-divergence term to the pre-fine-tuning model with `lambda=0.05`

For the pun fine-tuning stage it reports:

- LoRA `r=16`, `alpha=64`
- query/key linear layers
- maximum learning rate `1e-4`
- 20% warmup + cosine
- batch size 32
- 5 epochs

Configs:

```text
configs/minami2024_reading.json
configs/minami2024_pun_sft.json
```

Run with:

```bash
pip install -e '.[train]'
python training/train_sft.py configs/minami2024_reading.json
python training/train_sft.py configs/minami2024_pun_sft.json
```

### Reproduction caveat

The paper's reading-stage loss includes a custom KL term. The generic TRL SFT runner in this repository does not duplicate that term because an exact implementation needs a frozen reference model and substantially more VRAM. Therefore `minami2024_reading.json` is a **near-reproduction recipe**, not bit-exact reproduction.

## 5. SFT -> DPO pun preference training

**Used in:** Minami et al., 2025.

The 2025 paper uses a two-stage approach:

1. SFT on pun/paraphrase pairs.
2. Preference optimization where **human-written puns are preferred** and **SFT-model-generated puns are dispreferred**.

This repository implements the same data structure:

```bash
python scripts/prepare_dpo_pairs.py \
  --human data/pun_sft/train.jsonl \
  --generated results/sft_generations.jsonl \
  --output data/pun_dpo/train.jsonl
```

Then:

```bash
python training/train_dpo.py configs/gemma2_jpn_dpo_3090.json
```

The bundled Gemma 2 configs are **24 GB RTX 3090 practical configs**, not claims of exact 2025-paper hyperparameters.

## 6. GPT-2 + PPO research path

**Used in:** Tsukami & Quan, FIT 2024.

That work fine-tuned a Japanese GPT-2 on the "北大駄洒落コーパス" and then used RLHF with PPO. The abstract reports roughly 10% pun generation for the PPO-trained active LM versus roughly 1% for its reference LM on 500 seeds.

The current repository does not ship a PPO trainer because the named corpus was not located as a clearly redistributable public dataset in this survey. The method remains documented as an experimental comparison target. If a licensed copy becomes available, it can be added as a fourth training condition.

## 7. Practical RTX 3090 configurations

The user-facing local configs use 4-bit QLoRA so that Japanese SFT/DPO experiments are realistic on a 24 GB GPU:

```text
configs/gemma2_jpn_sft_3090.json
configs/gemma2_jpn_dpo_3090.json
```

They use:

- Transformers
- PEFT / LoRA
- TRL (`SFTTrainer`, `DPOTrainer`)
- Hugging Face Datasets
- bitsandbytes 4-bit NF4
- gradient checkpointing

These are intentionally separated from paper-exact settings.

## Recommended experiment matrix

| Condition | Data | Training | Purpose |
|---|---|---|---|
| A | local benchmark | prompt only | LLM-only baseline |
| B | dictionary + benchmark | no fine-tuning | phonetic retrieval + LLM |
| C | NDL furigana | CALM2 LoRA | phonetic-reading pretraining reproduction |
| D | authorized pun pairs | SFT | Japanese pun-style learning |
| E | human/model pun pairs | DPO | preference optimization |
| F | SMART data | SFT/DPO reference | compare preference-learning machinery |

For claims about memorability, do not rely only on human ratings. Add delayed recall, following the methodological warning from SMART that stated preference and measured learning can diverge.
