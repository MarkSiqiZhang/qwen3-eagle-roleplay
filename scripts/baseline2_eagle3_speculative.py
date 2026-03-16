"""
Baseline 2: EAGLE 3 Speculative Decoding with General Draft Model

Measures speculative decoding speed of Qwen3-4B + off-the-shelf EAGLE 3
draft model on role-play test prompts. Records TPS, acceptance length,
and speedup over autoregressive baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/baseline2_eagle3_speculative.py \
        --base_model_path models/Qwen3-4B \
        --ea_model_path models/Qwen3-4B_eagle3 \
        --data_path data/roleplay_data/jack_sparrow_test.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add EAGLE repo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EAGLE"))
from eagle.model.ea_model import EaModel


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


def prepare_input_ids(tokenizer, system_content, user_content, device):
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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return input_ids


def main():
    parser = argparse.ArgumentParser(description="Baseline 2: EAGLE 3 Speculative Decoding")
    parser.add_argument("--base_model_path", type=str, default="models/Qwen3-4B")
    parser.add_argument("--ea_model_path", type=str, default="models/Qwen3-4B_eagle3")
    parser.add_argument("--data_path", type=str, default="data/roleplay_data/jack_sparrow_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/baseline2")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--total_token", type=int, default=60,
                        help="Total draft tokens in the tree")
    parser.add_argument("--depth", type=int, default=7,
                        help="Max depth of the draft tree")
    parser.add_argument("--draft_top_k", type=int, default=10,
                        help="Top-k for draft tree expansion")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Loading EAGLE 3 model...")
    print(f"  Target: {args.base_model_path}")
    print(f"  Draft:  {args.ea_model_path}")

    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.draft_top_k,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
    print(f"Model loaded on {device} (dtype=float16)")

    samples = load_test_data(args.data_path)
    print(f"Loaded {len(samples)} test samples")

    # Warmup
    print("Running warmup...")
    dummy_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(device)
    for _ in range(3):
        model.eagenerate(
            dummy_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=32,
            max_length=args.max_length,
            log=False,
        )
    print("Warmup complete.")

    results = []
    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        input_ids = prepare_input_ids(
            tokenizer, sample["system"], sample["user"], device
        )
        input_len = input_ids.shape[1]

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        output_ids, new_token, steps = model.eagenerate(
            input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            log=True,
        )

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        total_time = t_end - t_start
        num_generated = new_token
        tps = num_generated / total_time if total_time > 0 else 0
        # acceptance length = tokens generated per verification step
        accept_length = num_generated / (steps + 1) if steps >= 0 else 0

        generated_text = tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        )

        result = {
            "sample_idx": idx,
            "user_prompt": sample["user"],
            "reference": sample["reference"],
            "generated": generated_text,
            "num_input_tokens": input_len,
            "num_output_tokens": num_generated,
            "num_steps": steps + 1,
            "avg_accept_length": round(accept_length, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "tps": round(tps, 2),
        }
        results.append(result)

    # Save per-sample results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs_path = output_dir / "generation_outputs.jsonl"
    with open(outputs_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute aggregate metrics
    tps_values = [r["tps"] for r in results if r["tps"] > 0]
    accept_lengths = [r["avg_accept_length"] for r in results if r["avg_accept_length"] > 0]
    output_lengths = [r["num_output_tokens"] for r in results]
    total_times = [r["total_time_ms"] for r in results]

    summary = {
        "model": args.base_model_path,
        "draft_model": args.ea_model_path,
        "method": "EAGLE 3 Speculative Decoding",
        "dtype": "float16",
        "device": args.device,
        "num_samples": len(results),
        "max_new_tokens": args.max_new_tokens,
        "eagle3_config": {
            "total_token": args.total_token,
            "depth": args.depth,
            "draft_top_k": args.draft_top_k,
        },
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "metrics": {
            "avg_tps": round(mean(tps_values), 2),
            "median_tps": round(median(tps_values), 2),
            "std_tps": round(stdev(tps_values), 2) if len(tps_values) > 1 else 0,
            "avg_accept_length": round(mean(accept_lengths), 2),
            "median_accept_length": round(median(accept_lengths), 2),
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
    print(f"  Avg acceptance length: {summary['metrics']['avg_accept_length']}")
    print(f"  Avg output length: {summary['metrics']['avg_output_length']:.1f} tokens")
    print(f"  Total time: {summary['metrics']['total_generation_time_s']:.1f} s")


if __name__ == "__main__":
    main()
