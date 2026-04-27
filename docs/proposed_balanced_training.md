# Proposed Method: Balanced Training Run (50/50 Mix)

Complete instructions to train and evaluate an EAGLE 3 draft model using a
balanced 50/50 mix of Jack Sparrow + UltraChat data. This doc is self-contained
so it can be run on any server with the repo checked out.

## Goal

Train Proposed draft model on:
- **1,429** Jack Sparrow training samples (all of them)
- **1,429** UltraChat 200K samples (matched quantity for 50/50 ratio)
- Total: **2,858** samples, **50% in-domain / 50% general**
- **8 epochs** (matches the first Proposed run for apples-to-apples comparison)

Compare to:
- Prior Proposed run: 1,429 JS + 5,000 UltraChat = 6,429 samples (22% in-domain), 8 epochs → 56.29 TPS / 3.84 accept
- B3: 1,429 JS only, 20 epochs → 66.89 TPS / 4.25 accept

---

## 1. Prerequisites

### 1.1 Repo layout (must exist on the server)

```
qwen3-eagle-roleplay/
├── EAGLE/                                      # EAGLE speculative decoding library
├── scripts/
│   ├── prepare_mixed_data.py                   # Data mixing
│   ├── train_draft.py                          # Draft model training
│   ├── eagle3_draft_model.py                   # Draft model architecture
│   └── baseline2_eagle3_speculative.py         # Evaluation
├── models/
│   ├── Qwen3-4B-jack-sparrow/                  # LoRA-merged target model
│   └── Qwen3-4B_eagle3/                        # Pretrained EAGLE 3 draft (AngelSlim)
├── data/
│   └── roleplay_data/
│       ├── jack_sparrow_train.jsonl            # 1,429 samples
│       └── jack_sparrow_test.jsonl             # 349 samples
└── results/
```

If `models/` or `data/roleplay_data/` are missing, `rsync` them from the main
server first. They are too large for git.

### 1.2 Conda environment

The training expects the `eagle-roleplay` conda env. All commands use `conda run -n eagle-roleplay`.

If the env doesn't exist on the new server, create it:

```bash
conda create -n eagle-roleplay python=3.10 -y
conda activate eagle-roleplay
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.1 accelerate peft safetensors tqdm
pip install datasets huggingface_hub
pip install wandb   # optional; the training script writes to W&B if available
```

### 1.3 Hardware

- **2× GPUs with ≥11 GB each** (2× RTX 2080 Ti works, as does any Ampere+).
- Target model stays on GPU 0 (fp16, ~8 GB).
- Draft model + activations on GPU 1 (fp32, ~8 GB with gradient checkpointing).
- Eval is single-GPU (needs ~10 GB free).

### 1.4 Hugging Face access

The data preparation step downloads `HuggingFaceH4/ultrachat_200k`. It is public
(no gated access required). On first run it caches to `~/.cache/huggingface/`.
The full split is ~1.5 GB on disk; the script streams and keeps only ~3,000
samples in memory, but downloads the full split.

If disk is tight, point the cache elsewhere:

```bash
export HF_HOME=/large/disk/hf_cache
```

---

## 2. Step 1: Prepare Balanced Mixed Data

The existing `scripts/prepare_mixed_data.py` accepts `--general_samples`. Pass
**1429** to match the Jack Sparrow count, and use a new output path so the
original 22%-mix dataset isn't overwritten.

```bash
cd qwen3-eagle-roleplay

conda run -n eagle-roleplay python scripts/prepare_mixed_data.py \
    --character_data data/roleplay_data/jack_sparrow_train.jsonl \
    --general_samples 1429 \
    --output_path data/roleplay_data/mixed_train_balanced.jsonl \
    --seed 42
```

### Expected output

```
Loading character data from data/roleplay_data/jack_sparrow_train.jsonl...
  1429 character samples
Downloading UltraChat 200K (train_sft split)...
  207865 total samples in UltraChat 200K train_sft
Converting UltraChat samples...
  1429 general samples after filtering

Mixed dataset saved to data/roleplay_data/mixed_train_balanced.jsonl
  1429 character + 1429 general = 2858 total
  Character ratio: 50.0%
```

### Verification

```bash
wc -l data/roleplay_data/mixed_train_balanced.jsonl    # expect 2858
head -1 data/roleplay_data/mixed_train_balanced.jsonl | python -m json.tool
```

Each line is `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`.

---

## 3. Step 2: Train Draft Model

Same hyperparameters as the prior Proposed run for a fair comparison. The only
change is the data path and output directory.

### Command

Run in a detached session (screen / tmux) or via nohup — training takes
~1,270 s × 8 epochs ≈ **2.8 hours** (expect ~1.2 hr with 2.8k samples
vs the prior 6.4k).

Actually, since the dataset is smaller, the epoch time will drop roughly
proportionally:
- Prior run: 5,824 tokenized samples × 8 epochs, ~1,272 s/epoch
- This run: ~2,600 tokenized samples × 8 epochs, expect ~570 s/epoch
- Total: **~1.3 hours**

```bash
cd qwen3-eagle-roleplay
mkdir -p logs

nohup conda run -n eagle-roleplay python -u scripts/train_draft.py \
    --target_model_path models/Qwen3-4B-jack-sparrow \
    --draft_model_path models/Qwen3-4B_eagle3 \
    --data_path data/roleplay_data/mixed_train_balanced.jsonl \
    --output_dir models/Qwen3-4B_eagle3_proposed_balanced \
    --epochs 8 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr 2e-4 \
    --max_seq_length 2048 \
    --num_pred_steps 7 \
    --loss_decay 0.8 \
    --seed 42 \
    --save_every 2 \
    --target_device cuda:0 \
    --draft_device cuda:1 \
    > logs/proposed_balanced.log 2>&1 &

echo "Training PID: $!"
```

### What happens

1. Loads tokenizer and 2,858 samples, tokenizes, filters samples where the
   assistant portion is empty.
2. Loads Qwen3-4B-jack-sparrow on `cuda:0` in fp16, frozen.
3. Loads EAGLE 3 draft on `cuda:1` in fp32, trainable. Embedding layer is
   loaded from the target model and frozen. `d2t` / `t2d` vocabulary mappings
   are loaded from the pretrained draft (not rebuilt).
4. Per batch:
   - Forward target model, extract hidden states from layers `[2, 18, 33]`,
     concatenate to shape `[B, L, 3 × hidden_size]`.
   - Shift logits / input IDs left by one (predict next token).
   - Move tensors to `cuda:1`.
   - 7-step autoregressive draft forward; KL-divergence loss at each step
     weighted by `0.8^i`.
5. Backward + grad accumulation × 16 → step.
6. Checkpoint every 2 epochs + final epoch.
7. Writes W&B run to project `qwen3-eagle-draft` (if W&B is logged in).

### Monitor

```bash
tail -f logs/proposed_balanced.log
```

Expected pattern (from prior Proposed run, adjust epoch time):

```
Epoch 1/8 | Loss: 5.99 | Acc: 0.388 | LR: 1.92e-04 | Time: ~570s
Epoch 2/8 | Loss: 5.24 | Acc: 0.445 | LR: 1.71e-04 | Time: ~570s
  Checkpoint saved (epoch 2) -> models/Qwen3-4B_eagle3_proposed_balanced/
...
Epoch 8/8 | Loss: ~4.2 | Acc: ~0.54 | LR: 0.00e+00 | Time: ~570s
  Checkpoint saved (epoch 8) -> models/Qwen3-4B_eagle3_proposed_balanced/

Training complete. Final model saved to models/Qwen3-4B_eagle3_proposed_balanced/
```

Epoch 1 may be ~10% slower due to CUDA kernel autotuning.

### Output artifacts

```
models/Qwen3-4B_eagle3_proposed_balanced/
├── config.json          # copied from pretrained draft
└── model.safetensors    # ~830 MB (fc, midlayer, norm, lm_head, d2t, t2d)
```

`embed_tokens` is intentionally NOT saved — the inference code loads it from
the target model. This keeps the draft file small and matches the AngelSlim
draft layout.

---

## 4. Step 3: Transfer Model Back (optional)

If you trained on a different server, `rsync` the trained draft back to the
main eval server:

```bash
# On main eval server:
rsync -avP remote:qwen3-eagle-roleplay/models/Qwen3-4B_eagle3_proposed_balanced/ \
    models/Qwen3-4B_eagle3_proposed_balanced/
```

---

## 5. Step 4: Evaluate

Run the standard EAGLE 3 evaluation with fixed 256-token output (for fair TPS
comparison to every other experiment in RESULTS.md).

### Command

```bash
cd qwen3-eagle-roleplay

CUDA_VISIBLE_DEVICES=0 conda run -n eagle-roleplay python -u \
    scripts/baseline2_eagle3_speculative.py \
    --base_model_path models/Qwen3-4B-jack-sparrow \
    --ea_model_path models/Qwen3-4B_eagle3_proposed_balanced \
    --data_path data/roleplay_data/jack_sparrow_test.jsonl \
    --output_dir results/proposed_balanced_fixed256 \
    --max_new_tokens 256 \
    --min_new_tokens 256
```

Eval takes ~25 min for 349 samples. If GPU 0 is occupied, set
`CUDA_VISIBLE_DEVICES` to another free GPU.

### Output artifacts

```
results/proposed_balanced_fixed256/
├── generation_outputs.jsonl    # per-sample TPS, accept length, generated text
└── metrics_summary.json        # aggregate stats
```

### Expected contents of metrics_summary.json

```json
{
  "model": "models/Qwen3-4B-jack-sparrow",
  "draft_model": "models/Qwen3-4B_eagle3_proposed_balanced",
  "method": "EAGLE 3 Speculative Decoding",
  ...
  "metrics": {
    "avg_tps": ...,
    "avg_accept_length": ...,
    ...
  }
}
```

---

## 6. Comparison Table (fill in after eval)

Insert into `results/RESULTS.md` next to the existing Proposed row:

| Variant | Mix | Epochs | Avg TPS | Accept Length |
|---|---|---|---|---|
| Proposed (22% JS) | 1,429 + 5,000 | 8 | 56.29 | 3.84 |
| **Proposed (50% JS)** | **1,429 + 1,429** | **8** | **?** | **?** |
| B3 | 1,429 only | 20 | 66.89 | 4.25 |
| B2' | (no draft training) | — | 62.03 | 3.84 |

### Hypotheses

- **If Proposed-50 beats Proposed-22 on JS acceptance**: the character ratio
  was the bottleneck. Doubling it to 50% gives the draft enough JS signal.
- **If Proposed-50 ≈ B3 on JS acceptance**: we've matched the in-domain peak
  with only 50% character data, showing mixed training is viable.
- **If Proposed-50 still below B3**: character ratio alone isn't the issue —
  total training steps or epoch count matters more. Consider raising epochs
  to 16–20 in a follow-up run.

---

## 7. Known Issues

### Loss explosion in epoch 1

If loss goes to NaN in the first few steps, the cause is usually that the
fp32 draft + fp16 target + autocast combination produces an overflow in the
KL computation on a specific sample. Reduce `--batch_size` to 1 (already the
default here) and ensure `gradient_checkpointing=True` (default).

### OOM on draft GPU

If GPU 1 OOMs, drop `--max_seq_length` from 2048 to 1536. The JS dataset
doesn't have samples longer than ~1,800 tokens anyway.

### `import datasets` fails

`pip install datasets huggingface_hub` inside the conda env.

### UltraChat download is slow

The first run downloads ~1.5 GB. Pre-warm the cache:

```bash
conda run -n eagle-roleplay python -c "
from datasets import load_dataset
load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft')
"
```

### Epoch time much higher than expected

Check `nvidia-smi` — another process sharing the GPU is the usual cause. Kill
it or pick different GPUs via `--target_device` / `--draft_device`.

---

## 8. TL;DR

```bash
# 1. Prepare balanced data (1 min)
conda run -n eagle-roleplay python scripts/prepare_mixed_data.py \
    --general_samples 1429 \
    --output_path data/roleplay_data/mixed_train_balanced.jsonl

# 2. Train (~1.3 hr on 2× 2080 Ti)
nohup conda run -n eagle-roleplay python -u scripts/train_draft.py \
    --target_model_path models/Qwen3-4B-jack-sparrow \
    --draft_model_path models/Qwen3-4B_eagle3 \
    --data_path data/roleplay_data/mixed_train_balanced.jsonl \
    --output_dir models/Qwen3-4B_eagle3_proposed_balanced \
    --epochs 8 --batch_size 1 --gradient_accumulation_steps 16 \
    --save_every 2 --target_device cuda:0 --draft_device cuda:1 \
    > logs/proposed_balanced.log 2>&1 &

# 3. Evaluate (~25 min)
CUDA_VISIBLE_DEVICES=0 conda run -n eagle-roleplay python -u \
    scripts/baseline2_eagle3_speculative.py \
    --base_model_path models/Qwen3-4B-jack-sparrow \
    --ea_model_path models/Qwen3-4B_eagle3_proposed_balanced \
    --data_path data/roleplay_data/jack_sparrow_test.jsonl \
    --output_dir results/proposed_balanced_fixed256 \
    --max_new_tokens 256 --min_new_tokens 256
```
