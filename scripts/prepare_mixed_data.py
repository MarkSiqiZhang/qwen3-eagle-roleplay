"""
Prepare mixed training data for the Proposed method.

Downloads a subset of UltraChat 200K (general conversational data),
converts to our messages format, and mixes with Jack Sparrow training data.

Usage:
    python scripts/prepare_mixed_data.py \
        --character_data data/roleplay_data/jack_sparrow_train.jsonl \
        --general_samples 5000 \
        --output_path data/roleplay_data/mixed_train.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def load_character_data(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def token_length(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def convert_ultrachat_sample(sample):
    """Convert UltraChat 200K sample to our messages format.

    UltraChat format: {"messages": [{"role": "user", "content": ...},
                                     {"role": "assistant", "content": ...}, ...]}
    Our format: {"messages": [system, user, assistant]} (single-turn)
    """
    msgs = sample["messages"]
    # Take only the first turn (user + assistant)
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

    # Skip very short or very long responses
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
    parser = argparse.ArgumentParser(description="Prepare mixed training data")
    parser.add_argument("--character_data", type=str,
                        default="data/roleplay_data/jack_sparrow_train.jsonl")
    parser.add_argument("--general_samples", type=int, default=5000,
                        help="Number of general samples to include")
    parser.add_argument("--output_path", type=str,
                        default="data/roleplay_data/mixed_train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer_path", type=str,
                        default="models/Qwen3-4B-jack-sparrow",
                        help="Tokenizer for length filtering")
    parser.add_argument("--max_token_length", type=int, default=0,
                        help="Drop samples exceeding this token length (0 = no filter)")
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer = None
    if args.max_token_length > 0:
        print(f"Loading tokenizer from {args.tokenizer_path} for length filtering...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load character data
    print(f"Loading character data from {args.character_data}...")
    char_data = load_character_data(args.character_data)
    if tokenizer is not None:
        before = len(char_data)
        char_data = [s for s in char_data
                     if token_length(tokenizer, s["messages"]) <= args.max_token_length]
        print(f"  {len(char_data)} character samples (dropped {before - len(char_data)} over {args.max_token_length} tokens)")
    else:
        print(f"  {len(char_data)} character samples")

    # Download and process UltraChat 200K
    print("Downloading UltraChat 200K (train_sft split)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    print(f"  {len(ds)} total samples in UltraChat 200K train_sft")

    # Convert and filter (collect more than needed, then subsample)
    print("Converting UltraChat samples...")
    general_data = []
    over_length = 0
    target_pool = args.general_samples * 3 if tokenizer is not None else args.general_samples * 2
    for sample in ds:
        converted = convert_ultrachat_sample(sample)
        if converted is None:
            continue
        if tokenizer is not None:
            if token_length(tokenizer, converted["messages"]) > args.max_token_length:
                over_length += 1
                continue
        general_data.append(converted)
        if len(general_data) >= target_pool:
            break

    if tokenizer is not None:
        print(f"  {over_length} UltraChat samples dropped for > {args.max_token_length} tokens")

    # Subsample to target count
    if len(general_data) > args.general_samples:
        general_data = random.sample(general_data, args.general_samples)
    print(f"  {len(general_data)} general samples after filtering")

    # Mix
    mixed = char_data + general_data
    random.shuffle(mixed)

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in mixed:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nMixed dataset saved to {output_path}")
    print(f"  {len(char_data)} character + {len(general_data)} general = {len(mixed)} total")
    print(f"  Character ratio: {len(char_data)/len(mixed)*100:.1f}%")


if __name__ == "__main__":
    main()
