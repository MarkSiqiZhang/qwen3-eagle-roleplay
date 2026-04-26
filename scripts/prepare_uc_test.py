"""
Prepare a cross-domain test set from UltraChat 200K test_sft (disjoint from train_sft
used during Proposed training).

Output format matches data/roleplay_data/jack_sparrow_test.jsonl:
  {"messages": [system, user, assistant]}
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def token_length(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def convert(sample):
    msgs = sample["messages"]
    user_msg = None
    assistant_msg = None
    for msg in msgs:
        if msg["role"] == "user" and user_msg is None:
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg is not None and assistant_msg is None:
            assistant_msg = msg["content"]
            break
    if not user_msg or not assistant_msg:
        return None
    if len(assistant_msg.split()) < 10 or len(assistant_msg.split()) > 500:
        return None
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=349)
    parser.add_argument("--output_path", type=str,
                        default="data/roleplay_data/ultrachat_test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer_path", type=str,
                        default="models/Qwen3-4B-jack-sparrow")
    parser.add_argument("--max_token_length", type=int, default=1024)
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print("Loading UltraChat 200K test_sft...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    print(f"  {len(ds)} total test_sft samples")

    out = []
    dropped_long = 0
    dropped_bad = 0
    indices = list(range(len(ds)))
    random.shuffle(indices)
    for i in indices:
        sample = ds[i]
        converted = convert(sample)
        if converted is None:
            dropped_bad += 1
            continue
        if token_length(tokenizer, converted["messages"]) > args.max_token_length:
            dropped_long += 1
            continue
        out.append(converted)
        if len(out) >= args.num_samples:
            break

    print(f"  dropped {dropped_bad} bad, {dropped_long} > {args.max_token_length} tokens")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved {len(out)} UC test samples to {output_path}")


if __name__ == "__main__":
    main()
