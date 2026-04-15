"""
Train EAGLE 3 Draft Model.

Supports both B3 (character-only data) and Proposed (mixed data) by
changing --data_path. Starts from pretrained AngelSlim draft weights.

Usage:
    # B3: Train on Jack Sparrow data only
    CUDA_VISIBLE_DEVICES=0 python scripts/train_draft.py \
        --target_model_path models/Qwen3-4B-jack-sparrow \
        --draft_model_path models/Qwen3-4B_eagle3 \
        --data_path data/roleplay_data/jack_sparrow_train.jsonl \
        --output_dir models/Qwen3-4B_eagle3_b3
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eagle3_draft_model import Eagle3DraftModel, padding


# Hidden state layer indices for Qwen3-4B (36 layers)
# Pattern: [2, num_layers//2, num_layers-3]
HIDDEN_STATE_LAYERS = [2, 18, 33]


class DraftTrainDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_length):
        self.data = []
        for sample in samples:
            item = tokenize_sample(sample, tokenizer, max_seq_length)
            if item is not None:
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tokenize_sample(sample, tokenizer, max_seq_length):
    messages = sample["messages"]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)

    prompt_messages = messages[:2]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)

    input_ids = full_ids["input_ids"]
    prompt_len = len(prompt_ids["input_ids"])

    if len(input_ids) <= prompt_len:
        return None

    loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
    }


def collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    loss_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = 1
        loss_mask[i, :seq_len] = item["loss_mask"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}


def extract_hidden_states(target_model, input_ids, attention_mask):
    """Extract hidden states from target model layers [2, 18, 33] and logits."""
    with torch.no_grad():
        outputs = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hs = outputs.hidden_states  # tuple of 37 tensors (embedding + 36 layers)
        hidden_cat = torch.cat(
            [hs[HIDDEN_STATE_LAYERS[0]], hs[HIDDEN_STATE_LAYERS[1]], hs[HIDDEN_STATE_LAYERS[2]]],
            dim=-1,
        )
        target_logits = outputs.logits
    return hidden_cat, target_logits


def main():
    parser = argparse.ArgumentParser(description="Train EAGLE 3 Draft Model")
    parser.add_argument("--target_model_path", type=str, default="models/Qwen3-4B-jack-sparrow")
    parser.add_argument("--draft_model_path", type=str, default="models/Qwen3-4B_eagle3")
    parser.add_argument("--data_path", type=str, default="data/roleplay_data/jack_sparrow_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/Qwen3-4B_eagle3_b3")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_pred_steps", type=int, default=7)
    parser.add_argument("--loss_decay", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--target_device", type=str, default="cuda:0",
                        help="Device for frozen target model")
    parser.add_argument("--draft_device", type=str, default="cuda:1",
                        help="Device for trainable draft model")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    target_device = torch.device(args.target_device)
    draft_device = torch.device(args.draft_device)

    # Load tokenizer
    print(f"Loading tokenizer from {args.target_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # Load data
    print(f"Loading data from {args.data_path}...")
    raw_samples = []
    with open(args.data_path) as f:
        for line in f:
            raw_samples.append(json.loads(line))

    dataset = DraftTrainDataset(raw_samples, tokenizer, args.max_seq_length)
    print(f"  {len(dataset)} samples after tokenization (from {len(raw_samples)} raw)")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )

    # Load target model (frozen) on GPU 0
    print(f"Loading target model on {target_device}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path, torch_dtype=torch.float16, device_map=target_device,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    # Load draft model (trainable) on GPU 1
    print(f"Loading draft model on {draft_device}...")
    draft_model = Eagle3DraftModel.from_pretrained(
        args.draft_model_path, target_model,
    )
    draft_model.gradient_checkpointing = args.gradient_checkpointing
    draft_model = draft_model.to(draft_device).to(torch.float32)

    # Count parameters
    trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in draft_model.parameters())
    print(f"  Draft model: {trainable:,} trainable / {total:,} total parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in draft_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader) // args.gradient_accumulation_steps,
    )

    # Training loop
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"grad_accum={args.gradient_accumulation_steps}, lr={args.lr}")
    print(f"Steps per epoch: {len(dataloader)}, effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        draft_model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        optimizer.zero_grad()

        t_start = time.time()
        for step, batch in enumerate(dataloader):
            # Target model forward on target_device
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            loss_mask = batch["loss_mask"]

            hidden_cat, target_logits = extract_hidden_states(target_model, input_ids, attention_mask)

            # Shift left by 1 (predict next token), then move to draft_device
            target_logits_shifted = padding(target_logits, left=False).to(draft_device)
            input_ids_shifted = padding(input_ids, left=False).to(draft_device)
            hidden_cat = hidden_cat.to(draft_device)
            attention_mask = attention_mask.to(draft_device)
            loss_mask_expanded = loss_mask[..., None].to(draft_device)

            # Draft model forward on draft_device
            with torch.amp.autocast("cuda", dtype=torch.float16):
                total_loss, avg_acc, step_losses, step_accs = draft_model.train_forward(
                    hidden_states_cat=hidden_cat,
                    target_logits=target_logits_shifted,
                    input_ids=input_ids_shifted,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask_expanded,
                    num_steps=args.num_pred_steps,
                    loss_decay=args.loss_decay,
                )

            loss = total_loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item()
            epoch_acc += avg_acc
            num_batches += 1

        t_elapsed = time.time() - t_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
              f"LR: {lr:.2e} | Time: {t_elapsed:.1f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_draft_model(draft_model, args.draft_model_path, args.output_dir, epoch + 1)

    print(f"\nTraining complete. Final model saved to {args.output_dir}/")


def save_draft_model(draft_model, original_draft_path, output_dir, epoch):
    """Save draft model in the same format as the original."""
    from safetensors.torch import save_file

    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Collect weights (exclude frozen embed_tokens)
    state_dict = {}
    for name, param in draft_model.named_parameters():
        if "embed_tokens" not in name:
            state_dict[name] = param.data
    for name, buf in draft_model.named_buffers():
        if name in ("d2t", "t2d"):
            state_dict[name] = buf

    save_file(state_dict, os.path.join(save_dir, "model.safetensors"))

    # Copy config from original draft model
    src_config = os.path.join(original_draft_path, "config.json")
    if os.path.exists(src_config):
        shutil.copy2(src_config, os.path.join(save_dir, "config.json"))

    print(f"  Checkpoint saved (epoch {epoch}) -> {save_dir}/")


if __name__ == "__main__":
    main()
