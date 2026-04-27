# Joint Alignment Speculative Decoding for Real-Time LLM Role-Play

*Mixed persona + general draft training for deployment-robust acceleration*

Authors · DSCI/CSE-498 · Spring 2026

---

## 1. Problem Description / Motivation

**The goal**: deploy a persona-specialized LLM (e.g., a Jack Sparrow chatbot) with two simultaneous requirements:

- **Speed** — ≥ 40 tok/s to feel real-time.
- **Persona fidelity** — responses stay in character.

**The standard solution**: LoRA-finetune the target on persona data (fidelity) + use *speculative decoding* with a small draft model (speed).

**The open question**: what draft model should one use for a persona-specialized target?

| Option | Intuition | Hidden risk |
|---|---|---|
| Off-the-shelf general draft | Safe, broad | Leaves in-persona speed on the table |
| Naive specialization (train draft on persona data) | Max in-character speed | Draft overfits — *off-topic* user queries slow down silently |
| **Joint Alignment** (ours) | Train draft on mixed persona + general data | Matches the deployed target's actual query distribution |

Speculative decoding is mathematically lossless (rejection sampling preserves the target's output distribution), so **draft model choice is a pure speed dial**. The question is: does specialization of the draft help everywhere, or only in-persona?

---

## 2. Dataset Description

| Dataset | Role | Samples used | Source |
|---|---|---|---|
| **RoleBench — Jack Sparrow** | Target LoRA fine-tuning; in-domain draft training; in-domain acceptance evaluation; reference-similarity evaluation | 1,429 train / 349 test | HuggingFace `ZenMoore/RoleBench` |
| **UltraChat 200K** | General-domain draft training (mixed-data half); out-of-domain acceptance evaluation; reference-similarity evaluation | 1,429 train / 349 test_sft | HuggingFace `HuggingFaceH4/ultrachat_200k` |

Preprocessing: samples filtered to ≤ 1024 tokens under Qwen3 chat template (`enable_thinking=False`). RoleBench train/test splits follow the benchmark; UltraChat train and test samples drawn from disjoint `train_sft` / `test_sft` shards.

---

## 3. Experiments / Model Selection

### System

- **Target**: Qwen3-4B, LoRA-finetuned on RoleBench Jack Sparrow (fp16).
- **Draft**: Qwen3-4B EAGLE 3 (~0.8B params, shares embedding with target). Trained by 7-step KL-divergence between target and draft distributions (decay 0.8).
- **Inference**: EAGLE 3 tree attention (depth=7, total_token=60, draft_top_k=10).

### Proposed solution — Joint Alignment draft training

Train the draft on a 50/50 mixed corpus of persona data (RoleBench Jack Sparrow) and general conversation data (UltraChat 200K). The mix aligns the draft with the target's *deployment distribution* (users send both in-persona and off-topic queries), not the narrow persona fine-tuning distribution.

### Baselines and variants

| ID | Target | Draft training data | Draft epochs |
|---|---|---|---|
| B1  | Base Qwen3-4B | — (autoregressive) | — |
| B1' | Qwen3-4B-JS  | — (autoregressive) | — |
| B2  | Base Qwen3-4B | Off-the-shelf EAGLE 3 | — |
| B2' | Qwen3-4B-JS  | Off-the-shelf EAGLE 3 | — |
| B3  (naive) | Qwen3-4B-JS | 1,429 JS | 20 |
| Proposed-A  | Qwen3-4B-JS | 1,429 JS + 5,000 UC (22% char) | 8 |
| **Proposed-B (ours)** | **Qwen3-4B-JS** | **1,429 JS + 1,429 UC (50% char)** | **8** |
| Proposed-C  | Qwen3-4B-JS | Same as Proposed-B | 20 |

Draft hyperparameters (all trainings identical): AdamW lr 2e-4 cosine, batch 2 × grad accum 8, 7-step KL loss, decay 0.8, seed 42, 2-GPU target/draft split (2080 Ti, 11 GB).

---

## 4. Evaluation

### Objective

We ask whether the proposed Joint Alignment draft training achieves a **better speed–coverage trade-off** than naive specialization or off-the-shelf drafts, measured by:

1. **Acceptance length** (primary): average accepted-tokens per draft step, on both in-domain (JS) and out-of-domain (UC) prompts.
2. **Tokens per second** (secondary): end-to-end generation speed, fixed 256-token output.
3. **Reference similarity** (quality check): BLEU + ROUGE-1/2/L vs ground-truth reference responses — verifies speculative decoding is empirically lossless.

### Hero result — Joint-domain acceptance Pareto plot

*Insert `results/pareto_plot.png`* — (in-domain, out-of-domain) acceptance length per config.

**Proposed-B is the only point that Pareto-dominates the general-draft baseline (B2') on both axes.** B3 maximizes in-domain acceptance but collapses to baseline on out-of-domain (overfitting). Proposed-C (20 epochs) regresses on in-domain — training past ~8 epochs on mixed data over-fits.

### Speed results (fixed 256-token output, 349 prompts per domain)

| Config | TPS | Speedup | JS accept | UC accept | UC ↓ from JS |
|---|---|---|---|---|---|
| B1  | 24.45 | 1.00× | — | — | — |
| B1' | 25.31 | 1.04× | — | — | — |
| B2' | 62.03 | 2.54× | 3.84 | 3.63 | −5.5% |
| B3 (naive) | **66.89** | **2.74×** | **4.25** | 3.61 | **−15.1%** |
| **Proposed-B** | 58.75 | 2.40× | 4.01 | **3.86** | **−3.7%** |
| Proposed-C | 55.36 | 2.26× | 3.53 | 3.63 | +2.8% |

### Quality evaluation — speculative decoding is lossless

Reference similarity on free-length RoleBench test generations:

| Target | Config | BLEU | R-1 | R-2 | R-L |
|---|---|---|---|---|---|
| Base  | B1 (AR) | 2.84 | 22.51 | 6.94 | 15.52 |
| Base  | B2 (spec.) | 2.82 | 22.95 | 6.85 | 15.58 |
| **JS-finetuned** | B1' (AR) | 9.05 | 32.64 | 11.35 | 24.19 |
| **JS-finetuned** | B2' | 8.60 | 32.45 | 11.77 | 24.07 |
| **JS-finetuned** | B3 | 9.06 | 32.84 | 12.02 | 24.43 |
| **JS-finetuned** | **Proposed-B** | 8.52 | 32.73 | 11.54 | 23.99 |

- LoRA fine-tuning: **ROUGE-L 15.5 → 24.2 (+56%)** (persona fidelity).
- All JS-finetuned-target configs (AR and speculative): **ROUGE-L spread < 2.2%** — empirical confirmation of speculative-decoding losslessness.
- Same pattern holds cross-domain (UC test): speculative variants cluster within < 4% on ROUGE-L.

### Key findings

1. **The naive specialist (B3) silently degrades deployment speed**: peak on JS but −15% on UC.
2. **Joint Alignment (Proposed-B) is Pareto-optimal**: the only config that dominates the general-draft baseline on both in-domain and out-of-domain acceptance.
3. **Over-training on mixed data hurts**: Proposed-C (20 epochs) regresses to baseline on both axes despite better training-loss metrics.
4. **Speculative decoding is empirically lossless** at fixed quality (ROUGE-L < 2.2% spread), so draft-model choice is a pure speed dial — justifying a quality-agnostic speed-only optimization target.

---

## Assets referenced

- Pareto plot: `results/pareto_plot.png` (+ `.pdf`)
- Full numeric tables: `results/RESULTS.md`
- Reference similarity JSON: `results/reference_similarity.json`
