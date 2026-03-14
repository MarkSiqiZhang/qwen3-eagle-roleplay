"""
Preprocess RoleBench data for a specific character into chat-format JSONL
suitable for SFT with Qwen3 series models' chat template.

Usage:
    python scripts/preprocess_rolebench.py \
        --character "<character_name>" \
        --output_dir "<output_dir>"
"""

import argparse
import json
import os
from pathlib import Path


ROLEBENCH_DIR = Path("data/rolebench/rolebench-eng")
PROFILES_PATH = Path("data/rolebench/profiles-eng/desc.json")


def load_character_profile(character: str) -> str:
    """Load the character description from profiles."""
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)
    if character not in profiles:
        available = [k for k in profiles if character.lower() in k.lower()]
        raise ValueError(
            f"Character '{character}' not found. Similar: {available}"
        )
    return profiles[character]


def load_samples(character: str, split_type: str, subset: str, split: str) -> list[dict]:
    """Load samples for a character from a specific split file."""
    path = ROLEBENCH_DIR / split_type / subset / f"{split}.jsonl"
    if not path.exists():
        return []
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("role") == character:
                samples.append(data)
    return samples


def format_chat(system_prompt: str, question: str, answer: str) -> dict:
    """Format a single QA pair into multi-turn chat format."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def build_system_prompt(character: str, profile: str) -> str:
    """Build the system prompt for the character."""
    return (
        f"You are {character}. Stay in character at all times and respond "
        f"as {character} would.\n\n"
        f"Character description: {profile}"
    )


def process_character(character: str, output_dir: Path):
    """Process all RoleBench data for a character into train/test JSONL."""
    profile = load_character_profile(character)
    system_prompt = build_system_prompt(character, profile)

    # Collect from instruction-generalization splits (used for our project)
    splits_config = {
        "train": [
            ("instruction-generalization", "general", "train"),
            ("instruction-generalization", "role_specific", "train"),
        ],
        "test": [
            ("instruction-generalization", "general", "test"),
            ("instruction-generalization", "role_specific", "test"),
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = character.lower().replace(" ", "_")

    stats = {}
    for out_split, sources in splits_config.items():
        conversations = []
        for split_type, subset, split in sources:
            samples = load_samples(character, split_type, subset, split)
            for sample in samples:
                generated = sample.get("generated", [])
                if not generated:
                    continue
                # Use the first (best) generated response
                answer = generated[0]
                conv = format_chat(system_prompt, sample["question"], answer)
                conversations.append(conv)

        out_path = output_dir / f"{safe_name}_{out_split}.jsonl"
        with open(out_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        stats[out_split] = len(conversations)
        print(f"  {out_split}: {len(conversations)} conversations -> {out_path}")

    # Also save the system prompt for reference
    meta = {
        "character": character,
        "profile": profile,
        "system_prompt": system_prompt,
        "stats": stats,
    }
    meta_path = output_dir / f"{safe_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, indent=2, fp=f, ensure_ascii=False)
    print(f"  metadata -> {meta_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess RoleBench for SFT")
    parser.add_argument(
        "--character", type=str, default="Jack Sparrow",
        help="Character name (must match RoleBench exactly)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/roleplay_data",
        help="Output directory for processed data",
    )
    args = parser.parse_args()

    print(f"Processing '{args.character}' from RoleBench...")
    stats = process_character(args.character, Path(args.output_dir))
    print(f"\nDone! Total: {sum(stats.values())} conversations")


if __name__ == "__main__":
    main()
