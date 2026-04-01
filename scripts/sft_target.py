"""
Target Model SFT: Fine-tune Qwen3-4B on character role-play data using LoRA.

Produces a persona-tuned target model for speculative decoding experiments.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/sft_target.py \
        --model_path models/Qwen3-4B \
        --data_path data/roleplay_data/jack_sparrow_train.jsonl \
        --output_dir models/Qwen3-4B-jack-sparrow-lora
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


def load_data(data_path):
    """Load JSONL chat data."""
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def tokenize_sample(sample, tokenizer, max_seq_length):
    """Tokenize a single conversation with assistant-only loss masking.

    The chat template produces:
        <|im_start|>system\n{content}<|im_end|>\n
        <|im_start|>user\n{content}<|im_end|>\n
        <|im_start|>assistant\n{content}<|im_end|>\n

    We mask everything before the assistant's content with -100 in labels.
    """
    messages = sample["messages"]

    # Tokenize full conversation (system + user + assistant)
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    full_ids = tokenizer(
        full_text, truncation=True, max_length=max_seq_length
    )

    # Tokenize prompt only (system + user) to find where assistant starts
    prompt_messages = messages[:2]  # system + user
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_length)

    input_ids = full_ids["input_ids"]
    attention_mask = full_ids["attention_mask"]

    # Labels: -100 for prompt tokens, actual ids for assistant tokens
    labels = [-100] * len(prompt_ids["input_ids"]) + input_ids[len(prompt_ids["input_ids"]):]

    # Ensure labels length matches input_ids
    labels = labels[:len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Target Model SFT with LoRA")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-4B")
    parser.add_argument("--data_path", type=str, default="data/roleplay_data/jack_sparrow_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/Qwen3-4B-jack-sparrow-lora")
    parser.add_argument("--merged_output_dir", type=str, default="models/Qwen3-4B-jack-sparrow")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_merge", action="store_true",
                        help="Skip merging LoRA into base model")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize data
    print(f"Loading data from {args.data_path}...")
    raw_samples = load_data(args.data_path)
    print(f"  {len(raw_samples)} samples loaded")

    tokenized = [
        tokenize_sample(s, tokenizer, args.max_seq_length)
        for s in raw_samples
    ]
    dataset = Dataset.from_list(tokenized)

    # Compute stats
    lengths = [len(t["input_ids"]) for t in tokenized]
    assistant_tokens = [sum(1 for l in t["labels"] if l != -100) for t in tokenized]
    print(f"  Avg seq length: {sum(lengths) / len(lengths):.0f} tokens")
    print(f"  Max seq length: {max(lengths)} tokens")
    print(f"  Avg assistant tokens: {sum(assistant_tokens) / len(assistant_tokens):.0f}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Merge LoRA into base model
    if not args.no_merge:
        print(f"Merging LoRA into base model -> {args.merged_output_dir}...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.merged_output_dir)
        tokenizer.save_pretrained(args.merged_output_dir)
        print(f"Merged model saved to {args.merged_output_dir}")

    print("Done!")


if __name__ == "__main__":
    main()
