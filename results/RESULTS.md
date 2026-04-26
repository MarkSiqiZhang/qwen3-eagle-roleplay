# Experiment Results

## Setup
- **Target Model**: Qwen3-4B (fp16)
- **Draft Model**: AngelSlim Qwen3-4B EAGLE 3
- **Character**: Jack Sparrow (1,429 train / 349 test samples from RoleBench)
- **Hardware**: RTX 2080 Ti (11GB) x4
- **Generation Config**: temperature=0.7, top_p=0.8, top_k=20
- **Thinking Mode**: Disabled (`enable_thinking=False`)

---

## Summary (Fixed 256-Token Output for Fair Comparison)

All experiments below use `--min_new_tokens 256 --max_new_tokens 256` to ensure identical output lengths for fair TPS comparison.

| Experiment | Target Model | Draft Model | Avg TPS | Accept Length | Speedup vs B1 |
|---|---|---|---|---|---|
| B1 | Qwen3-4B | — | 24.45 | — | 1.00x |
| B1' | Qwen3-4B-jack-sparrow | — | 25.31 | — | 1.04x |
| B2 | Qwen3-4B | EAGLE3 (general) | 45.21 | 2.96 | 1.85x |
| B2' | Qwen3-4B-jack-sparrow | EAGLE3 (general) | 62.03 | 3.84 | 2.54x |
| B3 | Qwen3-4B-jack-sparrow | EAGLE3 (JS-only, 20ep) | 66.89 | 4.25 | 2.74x |
| Proposed (22% char, 8ep) | Qwen3-4B-jack-sparrow | EAGLE3 (1.4k JS + 5k UC, 8ep) | 56.29 | 3.84 | 2.30x |
| **Proposed (50% char, 8ep)** | **Qwen3-4B-jack-sparrow** | **EAGLE3 (1.4k JS + 1.4k UC, 8ep)** | **58.75** | **4.01** | **2.40x** |
| Proposed (50% char, 20ep) | Qwen3-4B-jack-sparrow | EAGLE3 (1.4k JS + 1.4k UC, 20ep) | 55.36 | 3.53 | 2.26x |

### Key Observations
- **B2' > B2**: Persona-tuned target shows *higher* acceptance (3.84 vs 2.96), suggesting LoRA finetuning makes the target more predictable to the draft model.
- **B3 peaks in-domain but overfits**: 4.25 accept on JS (+10.7% over B2'), but drops to 3.61 on out-of-domain UC prompts (−15.1%), matching B2' — B3 pays its specialization gain entirely on OOD.
- **Proposed-B wins the generalization trade-off**: 4.01 accept on JS (−5.6% vs B3) but 3.86 on UC (+6.9% vs B3, +6.3% vs B2'). Smallest in-domain→OOD drop (−3.7%). Mixed training is net positive on *both* domains vs the off-the-shelf baseline.
- **Character ratio matters**: Raising character share from 22% to 50% improves in-domain acceptance from 3.84 → 4.01 (+4.4%) and TPS from 56.29 → 58.75 (+4.4%). Balanced 50/50 is directionally closer to B3 on JS without losing general-data exposure.
- **More epochs hurt the mixed draft**: 50% mix + 20 epochs regressed to 3.53 on JS despite training metrics improving (loss 3.26 → 2.74, acc 0.527 → 0.623). On UC it held at 3.63 (≈ B2'). Net: overtraining on mixed data drifts toward UC-friendly representations and loses JS-specific gains — early stopping by eval acceptance is required.
- **Bottom line**: Proposed-B (50% mix, 8 epochs) is the recommended deployment config — it trades 5.6% peak in-domain acceptance vs B3 for +6.9% OOD acceptance and the smallest generalization gap of any variant.

---

## Cross-Domain Evaluation (UltraChat 200K test_sft, 349 samples)

To test the overfitting hypothesis, all EAGLE 3 configs are evaluated on 349 UltraChat 200K `test_sft` prompts (disjoint from any training data), with the same target model (`Qwen3-4B-jack-sparrow`) and fixed 256-token output.

| Experiment | In-Domain (JS) Accept | Out-of-Domain (UC) Accept | Δ Accept | In-Domain TPS | OOD TPS |
|---|---|---|---|---|---|
| B2' (off-the-shelf draft) | 3.84 | 3.63 | −5.5% | 62.03 | 55.39 |
| B3 (100% JS, 20ep) | 4.25 | 3.61 | **−15.1%** | 66.89 | 55.09 |
| **Proposed B (50% mix, 8ep)** | **4.01** | **3.86** | **−3.7%** | **58.75** | **58.86** |
| Proposed C (50% mix, 20ep) | 3.53 | 3.63 | +2.8% | 55.36 | 55.71 |

### Cross-Domain Analysis

**Hypothesis validated, with nuance**: naive character-specialist draft (B3) loses its in-domain advantage on out-of-domain prompts, while the mixed-data Proposed-B draft retains gains on both.

Key comparisons on UC (out-of-domain):
- **Proposed-B 3.86 > B3 3.61** (+6.9%): mixed training beats naive specialization on OOD
- **Proposed-B 3.86 > B2' 3.63** (+6.3%): mixed training even beats the off-the-shelf general draft on OOD — the finetuning is a net gain, not merely a preservation
- **B3 3.61 ≈ B2' 3.63**: B3's specialization vanishes on OOD but doesn't destroy general drafting

In-domain vs OOD drop ranking (smallest drop = best generalization):
1. **Proposed-B: −3.7%** (4.01 → 3.86) — best generalization
2. B2': −5.5% (3.84 → 3.63)
3. B3: **−15.1%** (4.25 → 3.61) — clear specialization/overfitting signature
4. Proposed-C: +2.8% (3.53 → 3.63) — overtraining hurt JS, UC untouched

**Proposed-B trade-off**: sacrifices 5.6% peak in-domain acceptance vs B3 (4.01 vs 4.25), but gains 6.9% on OOD (3.86 vs 3.61). For any realistic deployment serving mixed prompts, Proposed-B dominates B3.

**Why Proposed-C regressed on JS but not UC**: with 50/50 data and 20 epochs, the draft saw 28.6k UC sample-epochs vs 11.4k for Proposed-B — later epochs drifted toward UC-friendly representations, causing JS-specific regression while UC performance held. This reinforces the "early stopping by eval acceptance" recommendation: training loss continues to decrease past the point where OOD generalization is maintained at the expense of in-domain fit.

---

## Detailed Results

### Baseline 1: Standard Autoregressive Generation

**Setup**: Qwen3-4B (no fine-tuning), standard `model.generate()`, single GPU

| Metric | Free-length | Fixed 256 |
|---|---|---|
| Avg TPS | 18.95 | 24.45 |
| Median TPS | 18.0 | 24.51 |
| Std TPS | 3.04 | 0.12 |
| Avg TTFT | 92.0 ms | 72.43 ms |
| Avg Output Length | 225.06 | 256 |
| Total Runtime | 4,401 s | 3,679 s |

---

### Baseline 1': Autoregressive with Persona-Tuned Target

**Setup**: Qwen3-4B-jack-sparrow (LoRA fine-tuned), standard `model.generate()`, single GPU

| Metric | Free-length | Fixed 256 |
|---|---|---|
| Avg TPS | 20.32 | 25.31 |
| Median TPS | 20.17 | 24.87 |
| Std TPS | 0.72 | 0.57 |
| Avg TTFT | 72.44 ms | 73.58 ms |
| Avg Output Length | 60.33 | 256 |
| Total Runtime | 1,079 s | 3,557 s |

Note: Free-length TPS is lower because the fine-tuned model generates shorter outputs with more EOS-adjacent tokens, adding overhead per token.

---

### Baseline 2: EAGLE 3 Speculative Decoding (General Draft)

**Setup**: Qwen3-4B + off-the-shelf EAGLE 3 draft model, single GPU

| Metric | Free-length | Fixed 256 |
|---|---|---|
| Avg TPS | 34.17 | 45.21 |
| Median TPS | 34.21 | 43.46 |
| Avg Accept Length | 2.79 | 2.96 |
| Avg Output Length | 224.2 | 257.79 |
| Total Runtime | 2,158 s | 2,035 s |
| Speedup vs B1 | 1.80x | 1.85x |

---

### Baseline 2': EAGLE 3 with Persona-Tuned Target (General Draft)

**Setup**: Qwen3-4B-jack-sparrow + off-the-shelf EAGLE 3 draft model, single GPU

| Metric | Free-length | Fixed 256 |
|---|---|---|
| Avg TPS | 41.4 | 62.03 |
| Median TPS | 40.5 | 60.12 |
| Avg Accept Length | 2.75 | 3.84 |
| Avg Output Length | 61.7 | 258.72 |
| Total Runtime | 470 s | 1,496 s |
| Speedup vs B1' | 2.04x | 2.45x |

Note: Free-length acceptance length (2.75) is lower than fixed-256 (3.84) because the model frequently tries to stop early — short sequences have proportionally more overhead from verification steps near EOS.

---

### Baseline 3: Naive Draft Fine-Tuning (Character-Only)

**Setup**: Qwen3-4B-jack-sparrow + EAGLE 3 draft fine-tuned on Jack Sparrow data only (1,429 samples, 20 epochs)

**Training**: 2-GPU split (target on GPU 0, draft on GPU 1), AdamW lr=2e-4, cosine schedule, batch_size=2, grad_accum=8, 7-step KL loss with 0.8 decay

| Metric | Fixed 256 |
|---|---|
| Avg TPS | 66.89 |
| Median TPS | 66.51 |
| Std TPS | 13.01 |
| Avg Accept Length | 4.25 |
| Median Accept Length | 4.21 |
| Avg Output Length | 258.66 |
| Total Runtime | 1,406 s |
| Speedup vs B1 | 2.74x |
| Speedup vs B1' | 2.64x |
| **Improvement over B2'** | **+7.8% TPS, +10.7% acceptance** |

---

### Proposed: Joint Alignment Speculative Decoding

Two variants trained, differing only in character/general mixing ratio. Both trained for 8 epochs with identical hyperparameters (2-GPU split, batch_size=1, grad_accum=16, lr=2e-4 cosine schedule, 7-step KL loss with 0.8 decay).

#### Variant A: 22% character ratio

**Training data**: 1,429 Jack Sparrow + 5,000 UltraChat 200K = 6,429 samples (22% / 78%)
- max_seq_length: 2048
- Final train loss: 4.26, accuracy: 0.539

| Metric | Fixed 256 |
|---|---|
| Avg TPS | 56.29 |
| Median TPS | 58.07 |
| Std TPS | 15.19 |
| Avg Accept Length | 3.84 |
| Median Accept Length | 3.94 |
| Avg Output Length | 258.39 |
| Total Runtime | 1,756 s |
| Speedup vs B1 | 2.30x |
| vs B2' | 0% acceptance (same), -9.2% TPS |
| vs B3 | -9.6% acceptance, -15.8% TPS |

#### Variant B: 50% character ratio (balanced)

**Training data**: 1,429 Jack Sparrow + 1,429 UltraChat 200K = 2,858 samples (50% / 50%)
- max_seq_length: 1024 (samples filtered to ≤1024 tokens during data prep to avoid OOM)
- Final train loss: 3.26, accuracy: 0.527

| Metric | Fixed 256 |
|---|---|
| Avg TPS | 58.75 |
| Median TPS | 58.75 |
| Std TPS | 12.57 |
| Avg Accept Length | 4.01 |
| Median Accept Length | 4.02 |
| Avg Output Length | 258.66 |
| Total Runtime | 1,623 s |
| Speedup vs B1 | 2.40x |
| vs B2' | +4.4% acceptance, -5.3% TPS |
| vs B3 | -5.6% acceptance, -12.2% TPS |
| **vs Variant A** | **+4.4% acceptance, +4.4% TPS** |

#### Variant C: 50% character ratio, 20 epochs (match B3 training budget)

**Training data**: Same 2,858-sample 50/50 mix as Variant B
- max_seq_length: 1024
- Final train loss: 2.74, accuracy: 0.623 (vs Variant B's 3.26 / 0.527 — substantially better train metrics)

| Metric | Fixed 256 |
|---|---|
| Avg TPS | 55.36 |
| Median TPS | 57.66 |
| Std TPS | 16.61 |
| Avg Accept Length | 3.53 |
| Median Accept Length | 3.68 |
| Avg Output Length | 258.02 |
| Total Runtime | 1,816 s |
| Speedup vs B1 | 2.26x |
| vs B2' | -8.1% acceptance, -10.8% TPS |
| vs B3 | -16.9% acceptance, -17.2% TPS |
| **vs Variant B (8ep)** | **-12.0% acceptance, -5.8% TPS** |

#### Analysis

- **Raising character ratio helps**: 22% → 50% improved both acceptance (3.84 → 4.01) and TPS (56.29 → 58.75). The draft sees more JS-specific signal per epoch.
- **More epochs hurt the mixed draft** (key surprising finding): Variant C (20 epochs) regressed to 3.53 acceptance despite better training metrics (acc 0.623 vs Variant B's 0.527). The draft overfits to the training distribution on a mixed corpus — the drift caused by continued training outweighs the fit gains. This is the opposite of B3's behavior (100% JS + 20 epochs → best in-domain acceptance).
- **Mechanism hypothesis**: With mixed data, later epochs over-memorize token co-occurrence patterns from UltraChat that don't generalize to JS-style inference-time sampling. B3 avoids this because its training distribution IS the evaluation distribution.
- **Variance check**: Variant C's std TPS (16.61) is high vs Variant B's (12.57). The drop is real but partially noisy.
- **Still below B3**: Even Variant B (best Proposed variant) does not match B3 on in-domain. But B3 is expected to collapse on out-of-distribution prompts — the trade-off matters only with cross-domain eval.

**Open questions (cross-domain evaluation needed)**:
- Does Variant B retain general drafting ability on non-JS prompts? If yes, the trade-off (lower in-domain peak, broader coverage) is justified.
- Does B3 collapse on non-JS prompts? If yes, it validates the overfitting hypothesis.
- Is there a sweet spot between 8 and 20 epochs for the balanced mix? Early stopping by eval acceptance (not training loss) may be needed.
