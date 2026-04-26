"""
Side-by-side web demo of the Joint Alignment story (poster headline).

All three columns share the **same persona-tuned target** (Qwen3-4B-jack-sparrow)
and run EAGLE-3 speculative decoding. They differ **only in the draft model**:

  * Column A — B2'   off-the-shelf EAGLE-3 draft (no retraining).
  * Column B — B3    naive specialization: draft retrained on 100% Jack Sparrow.
  * Column C — Prop-B Joint Alignment (ours): draft retrained on 50/50
              Jack Sparrow + UltraChat mix.

Speculative decoding is mathematically lossless — output quality is identical
across all three columns. The draft is a pure speed dial. Switch the domain
preset to "General (out-of-domain)" to see B3's accept length collapse while
Prop-B holds — that is the failure mode the poster argues against.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2 \\
    /data/users/jupyter-siz227/.conda/envs/eagle-roleplay/bin/python \\
    scripts/webdemo.py --port 7860

One RTX 2080 Ti (11GB) per column. Each column loads the 4B target + a draft.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from webdemo_models import EagleWorker, WorkerEvent, build_messages  # noqa: E402
from webdemo_presets import DOMAINS  # noqa: E402


# Hard-coded benchmark summary from results/RESULTS.md (fixed-256 column).
# Shown beneath the live demo so viewers can anchor the live TPS / accept
# length in published averages. The autoregressive B1' row is included as a
# reference point for "no speculative decoding" (≈25 TPS), since none of the
# three live columns run autoregressively any more.
BENCHMARK_TABLE_MD = """
| Col | Config | Draft | JS Accept | UC Accept | UC Δ | JS TPS | UC TPS |
|---|---|---|---|---|---|---|---|
| — | B1' (no spec. dec., reference) | — autoregressive — | — | — | — | 25.31 | 23.13 |
| **A** | **B2'** off-the-shelf | `Qwen3-4B_eagle3` | 3.84 | 3.63 | −5.5% | 62.03 | 55.39 |
| **B** | **B3** persona-only (naive) | `Qwen3-4B_eagle3_b3` | **4.25** | 3.61 | **−15.1%** | **66.89** | 55.09 |
| **C** | **Prop-B Joint Alignment (ours)** | `Qwen3-4B_eagle3_proposed_balanced` | 4.01 | **3.86** | **−3.7%** | 58.75 | **58.86** |

All speculative rows use the same Qwen3-4B-jack-sparrow target and EAGLE-3 decoding (max_new_tokens=256, temperature=0.7, top_p=0.8, top_k=20). On out-of-domain (UltraChat) prompts, **B3 silently loses 15.1 % of its acceptance gain** (4.25 → 3.61) while **Prop-B holds (4.01 → 3.86, only −3.7 %)** — the headline result. See `results/RESULTS.md` for full tables.
"""


DEFAULT_GEN = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": 256,
}


def format_stats(tps: float, accept_len: float | None, done: bool) -> str:
    suffix = " ✓" if done else " …"
    if accept_len is None:
        return f"### **TPS:** {tps:5.2f} tok/s{suffix}"
    return f"### **TPS:** {tps:5.2f} tok/s  ·  **accept len:** {accept_len:4.2f}{suffix}"


def run_worker_thread(worker, q: queue.Queue, messages, gen_kwargs):
    try:
        for ev in worker.stream(messages, gen_kwargs):
            q.put(ev)
    except Exception as e:  # pragma: no cover
        import traceback
        traceback.print_exc()
        q.put(WorkerEvent(text=f"[error: {e}]", tps=0.0, accept_len=None, done=True))
    finally:
        q.put(None)  # sentinel


HISTORY_HEADERS = [
    "#", "Mode", "Domain",
    "A TPS", "A accept",
    "B TPS", "B accept",
    "C TPS", "C accept",
]


def _fmt_accept(v):
    return f"{v:.2f}" if v is not None else "—"


def _history_rows(history: list[dict]):
    """Convert internal history state into Dataframe rows."""
    return [
        [
            row["idx"],
            row["mode"],
            row["domain"],
            f"{row['a_tps']:.2f}", _fmt_accept(row["a_accept"]),
            f"{row['b_tps']:.2f}", _fmt_accept(row["b_accept"]),
            f"{row['c_tps']:.2f}", _fmt_accept(row["c_accept"]),
        ]
        for row in history
    ]


def make_handler(worker_a, worker_b, worker_c):

    workers = [worker_a, worker_b, worker_c]

    def _snapshot(states, history):
        return (
            states[0]["text"] or "_(waiting...)_",
            format_stats(states[0]["tps"], states[0]["accept_len"], states[0]["done"]),
            states[1]["text"] or "_(waiting...)_",
            format_stats(states[1]["tps"], states[1]["accept_len"], states[1]["done"]),
            states[2]["text"] or "_(waiting...)_",
            format_stats(states[2]["tps"], states[2]["accept_len"], states[2]["done"]),
            history,
            _history_rows(history),
        )

    def _drain_parallel(queues, states, history):
        sentinels = [False, False, False]
        last_yield = time.perf_counter()
        while not all(sentinels):
            changed = False
            for i in range(3):
                try:
                    while True:
                        item = queues[i].get_nowait()
                        if item is None:
                            sentinels[i] = True
                            continue
                        states[i]["text"] = item.text
                        states[i]["tps"] = item.tps
                        states[i]["accept_len"] = item.accept_len
                        states[i]["done"] = item.done
                        changed = True
                except queue.Empty:
                    pass
            now = time.perf_counter()
            if changed and (now - last_yield) > 0.066:
                yield _snapshot(states, history)
                last_yield = now
            else:
                time.sleep(0.03)

    def _run_parallel(messages, gen_kwargs, states, history):
        queues = [queue.Queue() for _ in range(3)]
        threads = [
            threading.Thread(
                target=run_worker_thread,
                args=(workers[i], queues[i], messages, gen_kwargs),
                daemon=True,
            )
            for i in range(3)
        ]
        for t in threads:
            t.start()
        yield _snapshot(states, history)
        yield from _drain_parallel(queues, states, history)
        for t in threads:
            t.join(timeout=5.0)
        yield _snapshot(states, history)

    def _run_sequential(messages, gen_kwargs, states, history):
        """Run the three workers one at a time for clean benchmark-matching TPS.

        Each worker runs in a thread so that the perf_counter() timing inside
        the worker is not inflated by Gradio yield overhead from the consumer
        side. Without this isolation, EAGLE's per-step yield in ea_generate
        round-trips through the Gradio event loop before the next step starts,
        adding tens of ms per step and dropping reported TPS by 2-3x vs the
        offline benchmark.
        """
        yield _snapshot(states, history)
        last_yield = time.perf_counter()
        for i in range(3):
            q: queue.Queue = queue.Queue()
            thread = threading.Thread(
                target=run_worker_thread,
                args=(workers[i], q, messages, gen_kwargs),
                daemon=True,
            )
            thread.start()
            while True:
                try:
                    item = q.get(timeout=0.05)
                except queue.Empty:
                    now = time.perf_counter()
                    if (now - last_yield) > 0.066:
                        yield _snapshot(states, history)
                        last_yield = now
                    continue
                if item is None:
                    break
                states[i]["text"] = item.text
                states[i]["tps"] = item.tps
                states[i]["accept_len"] = item.accept_len
                states[i]["done"] = item.done
                now = time.perf_counter()
                if (now - last_yield) > 0.066:
                    yield _snapshot(states, history)
                    last_yield = now
            thread.join(timeout=5.0)
            yield _snapshot(states, history)
            last_yield = time.perf_counter()

    def handle(system, user, domain, run_mode, temperature, top_p, top_k, max_new_tokens, history):
        user = (user or "").strip()
        history = list(history or [])
        if not user:
            yield (
                "_(type a user message and press Send)_", "", "", "", "", "",
                history, _history_rows(history),
            )
            return

        messages = build_messages(system, user)
        gen_kwargs = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
        }

        states = [
            {"text": "", "tps": 0.0, "accept_len": None, "done": False}
            for _ in range(3)
        ]

        if run_mode == "Sequential (accurate TPS)":
            yield from _run_sequential(messages, gen_kwargs, states, history)
        else:
            yield from _run_parallel(messages, gen_kwargs, states, history)

        # Append a row for this run.
        history.append({
            "idx": len(history) + 1,
            "mode": "seq" if run_mode.startswith("Sequential") else "par",
            "domain": "JS" if "Jack Sparrow" in domain else "OOD",
            "a_tps": states[0]["tps"],
            "a_accept": states[0]["accept_len"],
            "b_tps": states[1]["tps"],
            "b_accept": states[1]["accept_len"],
            "c_tps": states[2]["tps"],
            "c_accept": states[2]["accept_len"],
        })
        yield _snapshot(states, history)

    return handle


def build_ui(worker_a, worker_b, worker_c, prompt_cache):

    def on_domain_change(domain_name):
        system = DOMAINS[domain_name]["system"]
        prompts = prompt_cache[domain_name]
        default_prompt = prompts[0] if prompts else ""
        return (
            gr.update(value=system),
            gr.update(choices=prompts, value=default_prompt),
            default_prompt,
        )

    def on_preset_pick(prompt):
        return prompt

    with gr.Blocks(title="Joint Alignment — EAGLE-3 Draft Comparison") as demo:
        gr.Markdown(
            "# Joint Alignment — EAGLE-3 Draft Comparison\n"
            "All three columns share the **same persona-tuned target** "
            "(`Qwen3-4B-jack-sparrow`) and run EAGLE-3 speculative decoding. "
            "They differ **only in the draft model**:\n"
            "* **A — B2'** off-the-shelf draft *(baseline, never aligned to JS-FT target)*\n"
            "* **B — B3** persona-only draft *(naive specialization: 100 % Jack Sparrow, 20 ep)*\n"
            "* **C — Prop-B** Joint Alignment (ours) *(50 / 50 Jack Sparrow + UltraChat, 8 ep)*\n\n"
            "Speculative decoding is mathematically lossless ⇒ **all three outputs share the same distribution**; "
            "the draft is a pure speed dial. Switch the domain dropdown to **General (out-of-domain)** "
            "to see B3's accept length collapse while C holds — the failure mode the poster argues against."
        )

        domain_names = list(DOMAINS.keys())
        with gr.Row():
            domain_dd = gr.Dropdown(
                choices=domain_names, value=domain_names[0], label="Domain",
                scale=1,
            )
            preset_dd = gr.Dropdown(
                choices=prompt_cache[domain_names[0]],
                value=prompt_cache[domain_names[0]][0] if prompt_cache[domain_names[0]] else "",
                label="Preset prompt",
                scale=3,
            )

        system_tb = gr.Textbox(
            label="System prompt",
            value=DOMAINS[domain_names[0]]["system"],
            lines=4,
        )
        user_tb = gr.Textbox(
            label="User message",
            value=prompt_cache[domain_names[0]][0] if prompt_cache[domain_names[0]] else "",
            lines=3,
        )

        with gr.Row():
            run_mode = gr.Radio(
                choices=["Sequential (accurate TPS)", "Parallel (all three at once)"],
                value="Sequential (accurate TPS)",
                label="Run mode",
                info="Sequential gives benchmark-matching TPS (one column at a time). Parallel is visually livelier but each column slows by GIL/PCIe sharing.",
            )

        with gr.Accordion("Sampling settings", open=False):
            with gr.Row():
                temperature_s = gr.Slider(0.0, 1.5, value=DEFAULT_GEN["temperature"], step=0.05, label="temperature")
                top_p_s = gr.Slider(0.1, 1.0, value=DEFAULT_GEN["top_p"], step=0.05, label="top_p")
                top_k_s = gr.Slider(1, 100, value=DEFAULT_GEN["top_k"], step=1, label="top_k")
                max_new_s = gr.Slider(32, 512, value=DEFAULT_GEN["max_new_tokens"], step=32, label="max_new_tokens")

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", scale=2)
            clear_btn = gr.Button("Clear outputs", scale=1)

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown(
                    "### A. B2' — off-the-shelf draft\n"
                    "*draft:* `Qwen3-4B_eagle3` *(baseline)*"
                )
                stats_a = gr.Markdown("### **TPS:** —")
                text_a = gr.Markdown("_(waiting...)_", height=400)
            with gr.Column():
                gr.Markdown(
                    "### B. B3 — persona-only draft *(naive)*\n"
                    "*draft:* `Qwen3-4B_eagle3_b3` *(100 % Jack Sparrow, 20 ep)*"
                )
                stats_b = gr.Markdown("### **TPS:** —")
                text_b = gr.Markdown("_(waiting...)_", height=400)
            with gr.Column():
                gr.Markdown(
                    "### C. **Prop-B — Joint Alignment (ours)**\n"
                    "*draft:* `Qwen3-4B_eagle3_proposed_balanced` *(50 / 50 mix, 8 ep)*"
                )
                stats_c = gr.Markdown("### **TPS:** —")
                text_c = gr.Markdown("_(waiting...)_", height=400)

        gr.Markdown("---\n### This session's runs")
        history_state = gr.State([])
        history_df = gr.Dataframe(
            headers=HISTORY_HEADERS, value=[], interactive=False, wrap=False,
            label="Per-run stats — toggle JS / OOD presets and watch B's accept length drop while C holds",
        )
        with gr.Row():
            reset_history_btn = gr.Button("Reset history")

        gr.Markdown("---\n### Benchmark averages (from `results/RESULTS.md`, fixed-256)")
        gr.Markdown(BENCHMARK_TABLE_MD)

        domain_dd.change(
            on_domain_change,
            inputs=[domain_dd],
            outputs=[system_tb, preset_dd, user_tb],
        )
        preset_dd.change(on_preset_pick, inputs=[preset_dd], outputs=[user_tb])

        handler = make_handler(worker_a, worker_b, worker_c)
        send_btn.click(
            handler,
            inputs=[
                system_tb, user_tb, domain_dd, run_mode,
                temperature_s, top_p_s, top_k_s, max_new_s,
                history_state,
            ],
            outputs=[
                text_a, stats_a, text_b, stats_b, text_c, stats_c,
                history_state, history_df,
            ],
        )

        def clear_outputs():
            return (
                "_(waiting...)_", "### **TPS:** —",
                "_(waiting...)_", "### **TPS:** —",
                "_(waiting...)_", "### **TPS:** —",
            )

        clear_btn.click(
            clear_outputs,
            outputs=[text_a, stats_a, text_b, stats_b, text_c, stats_c],
        )
        reset_history_btn.click(
            lambda: ([], []),
            outputs=[history_state, history_df],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Joint Alignment EAGLE-3 draft comparison demo",
    )
    parser.add_argument(
        "--target_model", default="models/Qwen3-4B-jack-sparrow",
        help="shared persona-tuned target for all three columns",
    )
    parser.add_argument(
        "--draft_a", default="models/Qwen3-4B_eagle3",
        help="column A — B2' off-the-shelf EAGLE-3 draft",
    )
    parser.add_argument(
        "--draft_b", default="models/Qwen3-4B_eagle3_b3",
        help="column B — B3 persona-only EAGLE-3 draft (100%% Jack Sparrow)",
    )
    parser.add_argument(
        "--draft_c", default="models/Qwen3-4B_eagle3_proposed_balanced",
        help="column C — Prop-B Joint Alignment draft (50/50 mix, ours)",
    )
    parser.add_argument("--device_a", default="cuda:0")
    parser.add_argument("--device_b", default="cuda:1")
    parser.add_argument("--device_c", default="cuda:2")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--skip_warmup", action="store_true",
        help="Skip model warmup (faster startup but first generation will be slow)",
    )
    args = parser.parse_args()

    target = REPO_ROOT / args.target_model
    draft_a = REPO_ROOT / args.draft_a
    draft_b = REPO_ROOT / args.draft_b
    draft_c = REPO_ROOT / args.draft_c
    for p in (target, draft_a, draft_b, draft_c):
        if not p.exists():
            raise SystemExit(f"model path not found: {p}")

    print(f"CUDA visible devices: torch.cuda.device_count()={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
        print(f"  cuda:{i} {name} {mem}GiB")

    print("Loading worker A (target + B2' off-the-shelf draft)...")
    worker_a = EagleWorker(
        base_model_path=str(target),
        ea_model_path=str(draft_a),
        device=args.device_a,
        label="A",
    )
    print("Loading worker B (target + B3 persona-only draft)...")
    worker_b = EagleWorker(
        base_model_path=str(target),
        ea_model_path=str(draft_b),
        device=args.device_b,
        label="B",
    )
    print("Loading worker C (target + Prop-B Joint Alignment draft)...")
    worker_c = EagleWorker(
        base_model_path=str(target),
        ea_model_path=str(draft_c),
        device=args.device_c,
        label="C",
    )

    if not args.skip_warmup:
        print("Warming up worker A...")
        worker_a.warmup()
        print("Warming up worker B...")
        worker_b.warmup()
        print("Warming up worker C...")
        worker_c.warmup()

    # Pre-compute preset prompt lists (they read jsonl files).
    prompt_cache = {name: info["prompts"]() for name, info in DOMAINS.items()}

    demo = build_ui(worker_a, worker_b, worker_c, prompt_cache)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
