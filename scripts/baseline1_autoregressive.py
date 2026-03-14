"""
Baseline 1: Standard Autoregressive Inference

Measures vanilla autoregressive generation speed of a target model
on role-play test prompts. Records TPS, TTFT, and saves outputs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/baseline1_autoregressive.py \
        --model_path models/Qwen3-4B \
        --data_path data/roleplay_data/jack_sparrow_test.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median, stdev

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


class TimingLogitsProcessor(LogitsProcessor):
    """Pass-through processor that records timestamps at each decode step."""

    def __init__(self):
        self.call_times = []

    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        self.call_times.append(time.perf_counter())
        return scores

    def reset(self):
        self.call_times = []


def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_test_data(data_path):
    samples = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            msgs = data["messages"]
            samples.append({
                "system": msgs[0]["content"],
                "user": msgs[1]["content"],
                "reference": msgs[2]["content"],
            })
    return samples


def prepare_input(tokenizer, system_content, user_content, device):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    return inputs


def generate_with_timing(model, inputs, timer, generation_kwargs):
    """Run generation and return timing metrics."""
    timer.reset()
    input_length = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    output_ids = model.generate(
        **inputs,
        logits_processor=[timer],
        **generation_kwargs,
    )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    num_generated = output_ids.shape[1] - input_length
    generated_ids = output_ids[0, input_length:]

    total_time = t_end - t_start

    if len(timer.call_times) > 0:
        ttft = timer.call_times[0] - t_start
        decode_time = t_end - timer.call_times[0]
        tps = num_generated / decode_time if decode_time > 0 else 0
    else:
        ttft = total_time
        decode_time = 0
        tps = 0

    return {
        "generated_ids": generated_ids,
        "num_input_tokens": input_length,
        "num_output_tokens": num_generated,
        "ttft_ms": ttft * 1000,
        "decode_time_ms": decode_time * 1000,
        "total_time_ms": total_time * 1000,
        "tps": tps,
    }


def run_warmup(model, tokenizer, device, generation_kwargs):
    """Run a few dummy generations to stabilize CUDA kernels."""
    print("Running warmup...")
    dummy_messages = [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Hello!"},
    ]
    prompt = tokenizer.apply_chat_template(
        dummy_messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    dummy_timer = TimingLogitsProcessor()

    for _ in range(3):
        model.generate(
            **inputs,
            logits_processor=[dummy_timer],
            **generation_kwargs,
        )
        dummy_timer.reset()
    print("Warmup complete.")


def main():
    parser = argparse.ArgumentParser(description="Baseline 1: Autoregressive Inference")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-4B")
    parser.add_argument("--data_path", type=str, default="data/roleplay_data/jack_sparrow_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/baseline1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    print(f"Model loaded on {args.device} (dtype=float16)")

    samples = load_test_data(args.data_path)
    print(f"Loaded {len(samples)} test samples")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    run_warmup(model, tokenizer, args.device, generation_kwargs)

    timer = TimingLogitsProcessor()
    results = []

    with torch.inference_mode():
        for idx, sample in enumerate(tqdm(samples, desc="Generating")):
            inputs = prepare_input(
                tokenizer, sample["system"], sample["user"], args.device
            )
            metrics = generate_with_timing(model, inputs, timer, generation_kwargs)

            generated_text = tokenizer.decode(
                metrics["generated_ids"], skip_special_tokens=True
            )

            result = {
                "sample_idx": idx,
                "user_prompt": sample["user"],
                "reference": sample["reference"],
                "generated": generated_text,
                "num_input_tokens": metrics["num_input_tokens"],
                "num_output_tokens": metrics["num_output_tokens"],
                "ttft_ms": round(metrics["ttft_ms"], 2),
                "decode_time_ms": round(metrics["decode_time_ms"], 2),
                "total_time_ms": round(metrics["total_time_ms"], 2),
                "tps": round(metrics["tps"], 2),
            }
            results.append(result)

    # Save per-sample results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs_path = output_dir / "generation_outputs.jsonl"
    with open(outputs_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute and save aggregate metrics
    tps_values = [r["tps"] for r in results if r["tps"] > 0]
    ttft_values = [r["ttft_ms"] for r in results]
    output_lengths = [r["num_output_tokens"] for r in results]
    total_times = [r["total_time_ms"] for r in results]

    summary = {
        "model": args.model_path,
        "dtype": "float16",
        "device": args.device,
        "num_samples": len(results),
        "max_new_tokens": args.max_new_tokens,
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": True,
        },
        "metrics": {
            "avg_tps": round(mean(tps_values), 2),
            "median_tps": round(median(tps_values), 2),
            "std_tps": round(stdev(tps_values), 2) if len(tps_values) > 1 else 0,
            "avg_ttft_ms": round(mean(ttft_values), 2),
            "median_ttft_ms": round(median(ttft_values), 2),
            "avg_output_length": round(mean(output_lengths), 2),
            "avg_total_time_ms": round(mean(total_times), 2),
            "total_generation_time_s": round(sum(total_times) / 1000, 2),
        },
    }

    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  Avg TPS: {summary['metrics']['avg_tps']}")
    print(f"  Median TPS: {summary['metrics']['median_tps']}")
    print(f"  Avg TTFT: {summary['metrics']['avg_ttft_ms']:.1f} ms")
    print(f"  Avg output length: {summary['metrics']['avg_output_length']:.1f} tokens")
    print(f"  Total time: {summary['metrics']['total_generation_time_s']:.1f} s")


if __name__ == "__main__":
    main()
