#!/bin/bash
# Wait for B2' (PID 1070648) to finish, then run B3, Proposed-B, Proposed-C sequentially on GPU 0.
set -u

B2P_PID=1070648
echo "[chain] waiting for B2' PID $B2P_PID to finish..."
while kill -0 $B2P_PID 2>/dev/null; do sleep 30; done
echo "[chain] B2' done, sleeping 10s for GPU cleanup"
sleep 10

run_eval() {
  local name=$1
  local draft=$2
  echo "[chain] launching $name on GPU 0"
  CUDA_VISIBLE_DEVICES=0 conda run -n eagle-roleplay python -u \
    scripts/baseline2_eagle3_speculative.py \
    --base_model_path models/Qwen3-4B-jack-sparrow \
    --ea_model_path "$draft" \
    --data_path data/roleplay_data/ultrachat_test.jsonl \
    --output_dir "results/cross_domain_$name" \
    --max_new_tokens 256 --min_new_tokens 256 \
    > "logs/cross_domain_$name.log" 2>&1
  echo "[chain] $name exit code $?"
  sleep 10
}

run_eval b3          models/Qwen3-4B_eagle3_b3
run_eval proposed_b  models/Qwen3-4B_eagle3_proposed_balanced
run_eval proposed_c  models/Qwen3-4B_eagle3_proposed_balanced_20ep
echo "[chain] all done"
