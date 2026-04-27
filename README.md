# Joint Alignment Speculative Decoding for Real-Time LLM Role-Play

DSCI / CSE-498 — Spring 2026 final project.

## Project description

Persona-tuned LLMs (e.g. a Jack-Sparrow chatbot built by LoRA fine-tuning a base
model) deliver strong character fidelity but their token distribution shifts
away from the general distribution. **Speculative decoding** with EAGLE-3 can
accelerate generation 2–3× by using a small draft model to propose tokens that
the target verifies in parallel — but only if the draft is well aligned to the
target. A general-purpose, off-the-shelf draft model under-accelerates
persona-tuned targets; the obvious fix — retrain the draft on character-only
data — silently degrades general-prompt acceptance because real users send
**both** in-character and off-topic queries.

This project introduces **Joint Alignment**: train the draft on a 50 / 50 mix
of persona dialogue (RoleBench) and general dialogue (UltraChat 200K). The
resulting draft (called **Prop-B**) is the only configuration we tested that
Pareto-dominates the off-the-shelf baseline on **both** in-domain and
out-of-domain acceptance, achieving a 2.40× end-to-end speedup with no
quality cost (speculative decoding is mathematically lossless — output
distribution is identical across drafts).

The project deliverables are:

1. A persona-tuned target model (`Qwen3-4B-jack-sparrow`).
2. Three EAGLE-3 draft variants — off-the-shelf (B2'), character-only (B3),
   and Joint Alignment (Prop-B).
3. End-to-end benchmarks (TPS, accept length, BLEU/ROUGE-L) across
   in-domain (RoleBench Jack Sparrow test) and out-of-domain (UltraChat
   `test_sft`) sets.
4. A live three-column Gradio web demo that compares all three drafts on
   the same target.

### Headline results (fixed 256-token output, RTX 2080 Ti)

| Config | Draft | JS Accept | UC Accept | UC Δ | JS TPS | UC TPS |
|---|---|---|---|---|---|---|
| B1' (autoregressive reference) | — | — | — | — | 25.31 | 23.13 |
| B2' (off-the-shelf draft) | `Qwen3-4B_eagle3` | 3.84 | 3.63 | −5.5 % | 62.03 | 55.39 |
| B3 (persona-only draft) | `Qwen3-4B_eagle3_b3` | **4.25** | 3.61 | **−15.1 %** | **66.89** | 55.09 |
| **Prop-B (Joint Alignment, ours)** | `Qwen3-4B_eagle3_proposed_balanced` | 4.01 | **3.86** | **−3.7 %** | 58.75 | **58.86** |

B3 is fastest in-domain but loses 15 % of its acceptance gain on OOD
prompts. Prop-B is the only draft that beats B2' on both axes. Full tables
and ablations: [`results/RESULTS.md`](results/RESULTS.md).

## Data sources

All data is downloaded from HuggingFace — see [`data/readme_data.txt`](data/readme_data.txt)
for exact commands and target paths.

| Dataset | Used for | HuggingFace ID |
|---|---|---|
| RoleBench (English) | Character SFT + in-domain test | `ZenMoore/RoleBench` |
| UltraChat 200K | General mix-in (`train_sft`) + OOD test (`test_sft`) | `HuggingFaceH4/ultrachat_200k` |

Models pulled from HuggingFace:

| Model | Role | HuggingFace ID |
|---|---|---|
| Qwen3-4B base | Target before SFT | `Qwen/Qwen3-4B` |
| AngelSlim Qwen3-4B EAGLE-3 | Off-the-shelf draft (B2') and starting weights for B3 / Prop-B training | `AngelSlim/Qwen3-4B_eagle3` |

We do not redistribute any data or model weights; everything is pulled
on-demand from HuggingFace.

## Required packages

Tested with Python 3.10 on Linux + 4× RTX 2080 Ti (11 GB) + CUDA 12.x.

Pinned versions are in [`requirements-lock.txt`](requirements-lock.txt);
loose top-level pins are in [`requirements.txt`](requirements.txt). Core
deps are:

```
torch >= 2.5
transformers >= 4.40
accelerate >= 1.0
datasets >= 3.0
peft >= 0.10
bitsandbytes >= 0.43
sentencepiece, protobuf, safetensors, huggingface-hub, wandb, tqdm
gradio              # web demo
sacrebleu, rouge-score   # eval_reference_similarity.py
matplotlib          # plot_pareto.py
```

The vendored copy of [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
under `EAGLE/` is added to `sys.path` by the scripts that need it — no
separate install step.

### Conda environment setup

```bash
conda create -n eagle-roleplay python=3.10 -y
conda activate eagle-roleplay
pip install -r requirements.txt
pip install gradio sacrebleu rouge-score matplotlib
```

For exact reproduction:

```bash
pip install -r requirements-lock.txt
```

## Repository layout

```
qwen3-eagle-roleplay/
├── README.md                       # this file
├── requirements.txt                # loose top-level pins
├── requirements-lock.txt           # full pinned environment
├── data/
│   ├── readme_data.txt             # how to obtain the data
│   ├── rolebench/                  # RoleBench raw download (gitignored)
│   └── roleplay_data/              # processed JSONL train/test splits
│       ├── jack_sparrow_train.jsonl
│       ├── jack_sparrow_test.jsonl
│       ├── ultrachat_test.jsonl
│       ├── mixed_train_balanced.jsonl   # Prop-B training data (50/50)
│       └── mixed_train.jsonl            # earlier 22 % char ratio
├── models/                          # local model weights (gitignored)
│   ├── Qwen3-4B/                            # downloaded base
│   ├── Qwen3-4B-jack-sparrow/                # SFT target (output of sft_target.py)
│   ├── Qwen3-4B_eagle3/                      # AngelSlim draft (B2')
│   ├── Qwen3-4B_eagle3_b3/                   # B3 draft (output of train_draft.py)
│   └── Qwen3-4B_eagle3_proposed_balanced/    # Prop-B draft (output of train_draft.py)
├── scripts/
│   ├── preprocess_rolebench.py     # RoleBench → JSONL chat format
│   ├── prepare_mixed_data.py       # builds Prop-B training mix
│   ├── prepare_uc_test.py          # builds UltraChat OOD test split
│   ├── sft_target.py               # LoRA SFT to produce JS-FT target
│   ├── eagle3_draft_model.py       # draft-model wrapper used during training
│   ├── train_draft.py              # B3 / Prop-B draft trainer (KL distillation)
│   ├── baseline1_autoregressive.py # B1, B1' eval
│   ├── baseline2_eagle3_speculative.py # B2, B2', B3, Prop-B eval
│   ├── eval_reference_similarity.py    # BLEU / ROUGE-L
│   ├── plot_pareto.py              # generates results/pareto_plot.{png,pdf}
│   ├── webdemo.py                  # 3-draft Gradio demo (entry point)
│   ├── webdemo_models.py           # PlainWorker / EagleWorker streaming wrappers
│   ├── webdemo_presets.py          # JS / OOD prompt presets
│   ├── run_cross_domain_chain.sh   # serial cross-domain eval helper
│   └── run_free_chain.sh           # serial free-length eval helper
├── EAGLE/                          # vendored SafeAILab/EAGLE library
├── results/
│   ├── RESULTS.md                  # full results writeup
│   ├── pareto_plot.{png,pdf}       # Pareto figure used in poster
│   ├── reference_similarity.json   # BLEU / ROUGE-L numbers
│   ├── baseline*/                  # per-config eval outputs
│   └── cross_domain_*/             # OOD eval outputs
├── docs/
│   ├── poster.tex                  # LaTeX source of poster
│   └── proposed_balanced_training.md
└── logs/                           # training / eval log files
```

## How to run

All commands assume `cwd = repo root` and `conda activate eagle-roleplay`.
GPU placement uses `CUDA_VISIBLE_DEVICES`; most steps need a single 11 GB
card, the web demo needs three.

### 1. Get the data

See [`data/readme_data.txt`](data/readme_data.txt). Briefly:

```bash
huggingface-cli download ZenMoore/RoleBench --repo-type dataset \
    --local-dir data/rolebench

python scripts/preprocess_rolebench.py \
    --character "Jack Sparrow" \
    --output_dir data/roleplay_data

python scripts/prepare_mixed_data.py \
    --jack_sparrow_path data/roleplay_data/jack_sparrow_train.jsonl \
    --output_path data/roleplay_data/mixed_train_balanced.jsonl \
    --balanced

python scripts/prepare_uc_test.py \
    --output_path data/roleplay_data/ultrachat_test.jsonl
```

### 2. Get the base + off-the-shelf draft models

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
huggingface-cli download AngelSlim/Qwen3-4B_eagle3 --local-dir models/Qwen3-4B_eagle3
```

### 3. SFT the target on Jack Sparrow

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sft_target.py \
    --model_path models/Qwen3-4B \
    --data_path data/roleplay_data/jack_sparrow_train.jsonl \
    --output_dir models/Qwen3-4B-jack-sparrow
```

LoRA-tunes the base; the merged model is saved at
`models/Qwen3-4B-jack-sparrow` and used as the target everywhere downstream.

### 4. Train the two draft variants

B3 — naive 100 % Jack Sparrow training (20 epochs):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_draft.py \
    --target_model_path models/Qwen3-4B-jack-sparrow \
    --draft_model_path models/Qwen3-4B_eagle3 \
    --data_path data/roleplay_data/jack_sparrow_train.jsonl \
    --num_epochs 20 \
    --output_dir models/Qwen3-4B_eagle3_b3
```

Prop-B — Joint Alignment 50 / 50 mix (8 epochs):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_draft.py \
    --target_model_path models/Qwen3-4B-jack-sparrow \
    --draft_model_path models/Qwen3-4B_eagle3 \
    --data_path data/roleplay_data/mixed_train_balanced.jsonl \
    --num_epochs 8 \
    --output_dir models/Qwen3-4B_eagle3_proposed_balanced
```

For long runs use `nohup ... &` so the job survives session disconnects.

### 5. Reproduce the benchmarks

In-domain (Jack Sparrow test, fixed 256 tokens):

```bash
# B1' autoregressive reference
CUDA_VISIBLE_DEVICES=0 python scripts/baseline1_autoregressive.py \
    --model_path models/Qwen3-4B-jack-sparrow \
    --data_path data/roleplay_data/jack_sparrow_test.jsonl \
    --output_dir results/baseline1_prime_fixed256 \
    --max_new_tokens 256 --min_new_tokens 256

# B2' / B3 / Prop-B speculative
for name in eagle3 eagle3_b3 eagle3_proposed_balanced; do
  CUDA_VISIBLE_DEVICES=0 python scripts/baseline2_eagle3_speculative.py \
      --base_model_path models/Qwen3-4B-jack-sparrow \
      --ea_model_path models/Qwen3-4B_$name \
      --data_path data/roleplay_data/jack_sparrow_test.jsonl \
      --output_dir results/${name}_fixed256 \
      --max_new_tokens 256 --min_new_tokens 256
done
```

Cross-domain (UltraChat OOD test) — replace the data_path with
`data/roleplay_data/ultrachat_test.jsonl` and write to `results/cross_domain_*/`.
The helper `scripts/run_cross_domain_chain.sh` does this serially.

BLEU / ROUGE-L vs ground-truth references:

```bash
python scripts/eval_reference_similarity.py \
    --output_path results/reference_similarity.json
```

Pareto plot (acceptance JS vs UC for all configs):

```bash
python scripts/plot_pareto.py
```

### 6. Launch the web demo

Three GPUs, one per column. Each column = same persona-tuned target +
different draft. Switch the `Domain` dropdown between
`Jack Sparrow (in-domain)` and `General (out-of-domain)` to see B3
silently lose acceptance while Prop-B holds.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/webdemo.py --port 7860
```

Open `http://<host>:7860`. Use `--share` for a public Gradio tunnel,
`--skip_warmup` for faster startup during iteration.

CLI options:

```
--target_model    shared persona-tuned target (default: models/Qwen3-4B-jack-sparrow)
--draft_a         column A — B2' off-the-shelf draft
--draft_b         column B — B3 persona-only draft (naive)
--draft_c         column C — Prop-B Joint Alignment draft (ours)
--device_a/b/c    GPU placement (default cuda:0/1/2)
```

## Hardware

All runs were on a single workstation with **4× NVIDIA RTX 2080 Ti
(11 GB)**. The web demo uses three of them concurrently
(target + draft per card ≈ 9.5 GB). Training uses one card at a time.

## Limitations

- Single persona (Jack Sparrow) — generalization to other characters is untested.
- Single mix ratio validated end-to-end (50 / 50). The 22 % ratio was tried and is worse; finer sweeps were not run due to compute budget.
- EAGLE-3 only — Medusa, PLD, and other speculative-decoding families are out of scope.
- 11 GB cards force fp16 + total_token=60 / depth=7 — bigger trees were not tested.

## Citation / acknowledgements

- **EAGLE-3** — Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. Code vendored under `EAGLE/`.
- **RoleBench** — Wang et al., *RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of LLMs* (`ZenMoore/RoleBench`).
- **UltraChat 200K** — `HuggingFaceH4/ultrachat_200k`.
- **AngelSlim Qwen3-4B EAGLE-3** — pretrained draft used as starting weights for B2' / B3 / Prop-B.
