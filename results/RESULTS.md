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
| B3 | Qwen3-4B-jack-sparrow | EAGLE3 (JS-only) | 66.89 | 4.25 | 2.74x |
| Proposed | Qwen3-4B-jack-sparrow | EAGLE3 (mixed data) | — | — | — |

### Key Observations
- **B2' > B2**: Persona-tuned target shows *higher* acceptance (3.84 vs 2.96), suggesting LoRA finetuning makes the target more predictable to the draft model.
- **B3 > B2'**: Character-specific draft training improves acceptance further (+10.7%), but overfits to one persona.
- **Gap for Proposed**: Can mixed-data training match B3's in-domain performance while maintaining general-purpose drafting?

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

**Setup**: Qwen3-4B-jack-sparrow + EAGLE 3 draft adapted via data mixing (UltraChat 200K + Jack Sparrow)

> Pending
