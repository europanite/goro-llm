# Experiment design

## Research questions

- **RQ1:** Does phonetic retrieval increase measurable sound similarity between the target reading and the mnemonic anchor?
- **RQ2:** Does consonant-heavy weighting outperform the supplied prototype's vowel-heavy heuristic on human judgments of “sounds like the target”?
- **RQ3:** Does overgenerate-and-rank improve naturalness and memorability over taking the top retrieval plan once?
- **RQ4:** Do perceived usefulness scores correlate with delayed factual recall?

## Systems

- `llm_only`: no dictionary and no explicit phonetic plan.
- `retrieval_top1`: highest-scoring phonetic plan, one generation.
- `overgenerate_rank`: up to 12 retrieval plans, multiple LLM candidates, transparent deterministic ranking.

## Phonetic profiles

- `template`: vowel 0.58 / consonant 0.42 — preserves the supplied prototype as a baseline.
- `balanced`: 0.50 / 0.50.
- `consonant`: vowel 0.35 / consonant 0.65 — research hypothesis motivated by Japanese pun literature.

These are heuristic research settings, not values copied from prior papers.

## Automatic measurements

1. Weighted phonetic similarity of the selected target span and anchor reading.
2. Whether the required anchor surface actually appears in the generated mnemonic.
3. Candidate length and diversity.
4. Failure rate: invalid JSON, missing anchor, no retrieval plan.

Automatic scores are **not** treated as memory effectiveness.

## Human evaluation

For each mnemonic, ask raters for 1–5 scores on:

- naturalness,
- memorability,
- ability to recall the target fact,
- factual correctness (binary).

Randomize system labels for real evaluation. For learning-effect evaluation, add an immediate recall test and a delayed recall test (e.g. next day or later) without showing the mnemonic. The SMART paper is a useful warning that what users say they prefer may differ from what actually improves learning.

## Minimal analysis

Report per-system means/medians, confidence intervals, failure rates, and paired comparisons on the same tasks. Do not claim a memory benefit from automatic phonetic scores alone.
