Data sources and how to populate the data/ folder
=================================================

This project does NOT redistribute any data. All datasets are pulled from
HuggingFace Hub on demand. Two raw datasets are needed; both are processed
into JSONL chat format under data/roleplay_data/ before training or eval.

Layout after a clean download + preprocessing:

    data/
    ├── readme_data.txt                     (this file)
    ├── rolebench/                          (raw RoleBench, ~hundreds of MB)
    │   ├── instructions-eng/
    │   ├── profiles-eng/
    │   │   └── desc.json
    │   ├── rolebench-eng/
    │   └── ... (Chinese splits, unused)
    └── roleplay_data/                      (processed splits, all JSONL)
        ├── jack_sparrow_train.jsonl        (1,429 samples)
        ├── jack_sparrow_test.jsonl         (349 samples)
        ├── jack_sparrow_meta.json          (character profile + stats)
        ├── ultrachat_test.jsonl            (349 OOD samples)
        ├── mixed_train_balanced.jsonl      (2,858 = 50 % JS + 50 % UC, used by Prop-B)
        └── mixed_train.jsonl               (1,429 JS + 5,000 UC, earlier 22 % char ratio)

----------------------------------------------------------------------
1. RoleBench (character dialogue, used for SFT + in-domain eval)
----------------------------------------------------------------------

Source : https://huggingface.co/datasets/ZenMoore/RoleBench
Paper  : Wang et al., RoleLLM (arXiv:2310.00746)
License: Apache 2.0

Download:

    huggingface-cli download ZenMoore/RoleBench \
        --repo-type dataset --local-dir data/rolebench

Convert RoleBench to chat-format JSONL for one character (Jack Sparrow used
in this project; any character listed in data/rolebench/profiles-eng/desc.json
can be substituted):

    python scripts/preprocess_rolebench.py \
        --character "Jack Sparrow" \
        --output_dir data/roleplay_data

This writes jack_sparrow_train.jsonl, jack_sparrow_test.jsonl and
jack_sparrow_meta.json into data/roleplay_data/.

----------------------------------------------------------------------
2. UltraChat 200K (general dialogue, used for the Prop-B mix + OOD eval)
----------------------------------------------------------------------

Source : https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
License: MIT

UltraChat is loaded on demand by the prep scripts via
`datasets.load_dataset(...)`; you do not need to manually download it. It
will land in the standard HuggingFace cache (~/.cache/huggingface/datasets
or $HF_DATASETS_CACHE).

Build the Prop-B training mix (50 % Jack Sparrow + 50 % UltraChat train_sft):

    python scripts/prepare_mixed_data.py \
        --jack_sparrow_path data/roleplay_data/jack_sparrow_train.jsonl \
        --output_path data/roleplay_data/mixed_train_balanced.jsonl \
        --balanced

Build the OOD test split from UltraChat test_sft (disjoint from train_sft):

    python scripts/prepare_uc_test.py \
        --output_path data/roleplay_data/ultrachat_test.jsonl

----------------------------------------------------------------------
3. Models (downloaded into ../models/, NOT into data/)
----------------------------------------------------------------------

Two HuggingFace models are required to reproduce; they live under the
sibling `models/` directory, not under `data/`:

    huggingface-cli download Qwen/Qwen3-4B \
        --local-dir ../models/Qwen3-4B          # base target

    huggingface-cli download AngelSlim/Qwen3-4B_eagle3 \
        --local-dir ../models/Qwen3-4B_eagle3   # off-the-shelf EAGLE-3 draft (B2')

The remaining three models — `Qwen3-4B-jack-sparrow` (SFT target),
`Qwen3-4B_eagle3_b3` (B3 draft), `Qwen3-4B_eagle3_proposed_balanced`
(Prop-B draft) — are produced by training scripts in this repository.
See the top-level README.md for the training commands.

----------------------------------------------------------------------
Disk footprint (approximate)
----------------------------------------------------------------------

  data/rolebench                 ~600 MB
  data/roleplay_data             ~5 MB total
  ~/.cache/huggingface/datasets  ~5 GB (UltraChat cached on first load)
  models/Qwen3-4B (fp16)         ~8 GB
  models/Qwen3-4B_eagle3         ~1.5 GB
  models/Qwen3-4B-jack-sparrow   ~8 GB (after merging LoRA)
  models/Qwen3-4B_eagle3_b3      ~1.5 GB
  models/Qwen3-4B_eagle3_proposed_balanced  ~1.5 GB
