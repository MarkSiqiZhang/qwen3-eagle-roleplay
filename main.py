"""
Joint Alignment Speculative Decoding — entry point.

Default: single-prompt inference using the persona-tuned target with the
Prop-B Joint Alignment EAGLE-3 draft on one GPU. Streams the response to
stdout and reports the decode-only TPS + speculative acceptance length.

    python main.py "Captain, the crew is restless about rum. What do you say?"
    python main.py --domain general "Explain quicksort in three sentences."

For the three-column live web demo (requires 3 GPUs):

    python main.py --demo

Training, batch evaluation, and reference-similarity scoring live in
scripts/ — see the run guide in ReadMe.txt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def run_demo() -> None:
    from webdemo import main as webdemo_main
    webdemo_main()


def run_single(args: argparse.Namespace) -> None:
    from webdemo_models import EagleWorker, build_messages
    from webdemo_presets import JACK_SPARROW_SYSTEM, ULTRACHAT_SYSTEM

    target = REPO_ROOT / args.target_model
    draft = REPO_ROOT / args.draft_model
    for p in (target, draft):
        if not p.exists():
            raise SystemExit(f"model path not found: {p}")

    system = JACK_SPARROW_SYSTEM if args.domain == "jack-sparrow" else ULTRACHAT_SYSTEM
    messages = build_messages(system, args.prompt)

    print(f"Loading EAGLE-3 model on {args.device}...")
    print(f"  target: {target}")
    print(f"  draft : {draft}")
    worker = EagleWorker(
        base_model_path=str(target),
        ea_model_path=str(draft),
        device=args.device,
        label="main",
    )
    print("Warming up...")
    worker.warmup()

    print()
    print(f"=== domain: {args.domain} ===")
    print(f"USER: {args.prompt}")
    print("ASSISTANT: ", end="", flush=True)

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }

    last_text = ""
    final = None
    t0 = time.perf_counter()
    for ev in worker.stream(messages, gen_kwargs):
        delta = ev.text[len(last_text):]
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
        last_text = ev.text
        final = ev
    print()

    if final is not None:
        wall = time.perf_counter() - t0
        print()
        print(
            f"TPS (decode-only) : {final.tps:6.2f} tok/s   "
            f"accept len: {final.accept_len:.2f}   "
            f"wall: {wall:.2f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint Alignment Speculative Decoding — entry point",
    )
    parser.add_argument(
        "prompt", nargs="?",
        default="Captain, the crew is restless about the rum rations. What do you tell them?",
        help="user prompt for single-shot inference (ignored when --demo is set)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="launch the three-column Gradio web demo (requires 3 GPUs)",
    )
    parser.add_argument(
        "--domain", choices=["jack-sparrow", "general"], default="jack-sparrow",
        help="system prompt to use (default: jack-sparrow)",
    )
    parser.add_argument("--target_model", default="models/Qwen3-4B-jack-sparrow")
    parser.add_argument(
        "--draft_model", default="models/Qwen3-4B_eagle3_proposed_balanced",
        help="Prop-B Joint Alignment draft (50/50 mix, 8 epochs)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_single(args)


if __name__ == "__main__":
    main()
