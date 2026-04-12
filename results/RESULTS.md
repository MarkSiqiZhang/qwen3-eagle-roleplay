# Experiment Results

## Setup
- **Target Model**: Qwen3-4B (fp16)
- **Draft Model**: AngelSlim Qwen3-4B EAGLE 3
- **Character**: Jack Sparrow (1,429 train / 349 test samples from RoleBench)
- **Hardware**: RTX 2080 Ti (11GB) x4
- **Generation Config**: temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=512
- **Thinking Mode**: Disabled (`enable_thinking=False`)

---

## Baseline 1: Standard Autoregressive Generation

**Setup**: Qwen3-4B (no fine-tuning), standard `model.generate()`, single GPU

| Metric | Value |
|---|---|
| Avg TPS | 18.95 |
| Median TPS | 18.0 |
| Std TPS | 3.04 |
| Avg TTFT | 92.0 ms |
| Avg Output Length | 225.06 tokens |
| Avg Total Time / Sample | 12,611 ms |
| Total Runtime | 4,401 s |

---

## Baseline 2: EAGLE 3 Speculative Decoding (General Draft)

**Setup**: Qwen3-4B (no fine-tuning) + off-the-shelf EAGLE 3 draft model, single GPU

| Metric | Value |
|---|---|
| Avg TPS | 34.17 |
| Median TPS | 34.21 |
| Avg Acceptance Length | 2.79 |
| Avg Output Length | 224.2 tokens |
| Total Runtime | 2,158 s |
| **Speedup vs B1** | **1.80x** |

---

## Baseline 1': Autoregressive with Persona-Tuned Target

**Setup**: Qwen3-4B-jack-sparrow (fine-tuned), standard `model.generate()`, single GPU

| Metric | Value |
|---|---|
| Avg TPS | 20.32 |
| Median TPS | 20.17 |
| Std TPS | 0.72 |
| Avg TTFT | 72.44 ms |
| Avg Output Length | 60.33 tokens |
| Avg Total Time / Sample | 3,091 ms |
| Total Runtime | 1,079 s |
| **Speedup vs B1** | **1.07x** (TPS comparable; faster total due to shorter outputs) |

---

## Baseline 2': EAGLE 3 with Persona-Tuned Target (General Draft)

**Setup**: Qwen3-4B-jack-sparrow (fine-tuned) + off-the-shelf EAGLE 3 draft model, single GPU

| Metric | Value |
|---|---|
| Avg TPS | 41.4 |
| Median TPS | 40.5 |
| Avg Acceptance Length | 2.75 |
| Avg Output Length | 61.7 tokens |
| Total Runtime | 470 s |
| **Speedup vs B1'** | **2.04x** |

---

## Baseline 3: Naive Draft Fine-Tuning

**Setup**: Qwen3-4B (fine-tuned) + EAGLE 3 draft fine-tuned only on Jack Sparrow data

> Pending

---

## Proposed: Joint Alignment Speculative Decoding

**Setup**: Qwen3-4B (fine-tuned) + EAGLE 3 draft adapted via data mixing (UltraChat 200K + Jack Sparrow)

> Pending
