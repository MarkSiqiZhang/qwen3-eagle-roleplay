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


def load_character_data(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


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
    args = parser.parse_args()

    random.seed(args.seed)

    # Load character data
    print(f"Loading character data from {args.character_data}...")
    char_data = load_character_data(args.character_data)
    print(f"  {len(char_data)} character samples")

    # Download and process UltraChat 200K
    print("Downloading UltraChat 200K (train_sft split)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    print(f"  {len(ds)} total samples in UltraChat 200K train_sft")

    # Convert and filter
    print("Converting UltraChat samples...")
    general_data = []
    for sample in ds:
        converted = convert_ultrachat_sample(sample)
        if converted is not None:
            general_data.append(converted)
        if len(general_data) >= args.general_samples * 2:
            # Collect 2x needed, then subsample for diversity
            break

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
