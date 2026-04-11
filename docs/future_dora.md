# Future Work: Try DoRA for Target SFT

## What is DoRA?

Weight-Decomposed Low-Rank Adaptation (DoRA) decomposes pre-trained weights into magnitude and direction, then only fine-tunes the direction via LoRA. Generally slightly better than vanilla LoRA at the same rank, with minimal extra cost.

Paper: https://arxiv.org/abs/2402.09353

## How to Switch

In `scripts/sft_target.py`, change the LoRA config:

```python
# Current: LoRA
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Change to: DoRA (just add use_dora=True)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    use_dora=True,  # <-- only change needed
)
```

Requires `peft >= 0.10.0` (already satisfied by requirements.txt).

## Other Things to Try

- Lower rank (r=8) to reduce overfitting risk on 1.4K samples
- Compare LoRA vs DoRA on acceptance rate, not just generation quality
