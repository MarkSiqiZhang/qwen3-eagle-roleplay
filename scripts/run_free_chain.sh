#!/bin/bash
# Sequentially run remaining free-length evals on GPU 0
set -u

run_eval() {
  local name=$1
  local draft=$2
  echo "[chain] launching $name"
  CUDA_VISIBLE_DEVICES=0 conda run -n eagle-roleplay python -u \
    scripts/baseline2_eagle3_speculative.py \
    --base_model_path models/Qwen3-4B-jack-sparrow \
    --ea_model_path "$draft" \
    --data_path data/roleplay_data/jack_sparrow_test.jsonl \
    --output_dir "results/$name" \
    --max_new_tokens 512 \
    > "logs/free_$name.log" 2>&1
  echo "[chain] $name exit $?"
  sleep 10
}

run_eval proposed_balanced       models/Qwen3-4B_eagle3_proposed_balanced
run_eval proposed                models/Qwen3-4B_eagle3_proposed
run_eval proposed_balanced_20ep  models/Qwen3-4B_eagle3_proposed_balanced_20ep
echo "[chain] done"
