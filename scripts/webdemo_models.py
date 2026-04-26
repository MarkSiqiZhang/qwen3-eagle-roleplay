"""
Model workers for the side-by-side web demo.

Two thin wrappers that expose the same streaming interface:

    worker.stream(messages, gen_kwargs) -> generator of WorkerEvent

where WorkerEvent is a dataclass carrying the running decoded text, the
decode-only TPS (prefill excluded, matching the benchmark harness in
scripts/baseline2_eagle3_speculative.py), an optional acceptance length for
the speculative worker, and a 'done' flag.

  * PlainWorker: AutoModelForCausalLM + TextIteratorStreamer. Used for the
    two autoregressive columns (base Qwen3 and finetuned Jack-Sparrow).
  * EagleWorker: EaModel.ea_generate from the vendored EAGLE library at
    EAGLE/eagle/model/ea_model.py. Used for the speculative column.

Model loading mirrors scripts/baseline1_autoregressive.py:39 and
scripts/baseline2_eagle3_speculative.py:90 so the on-screen TPS values line
up with the numbers already reported in results/RESULTS.md.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "EAGLE"))
from eagle.model.ea_model import EaModel  # noqa: E402


@dataclass
class WorkerEvent:
    text: str
    tps: float
    accept_len: float | None
    done: bool


def build_messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class PlainWorker:
    """Standard autoregressive HF generate, streaming via TextIteratorStreamer."""

    def __init__(self, model_path: str, device: str, label: str):
        self.label = label
        self.device = device
        print(f"[{label}] loading plain model from {model_path} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        print(f"[{label}] ready")

    def warmup(self, n: int = 2):
        messages = build_messages("You are a pirate.", "Hello!")
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        for _ in range(n):
            with torch.inference_mode():
                self.model.generate(
                    **ids, max_new_tokens=16, do_sample=False,
                )
        torch.cuda.synchronize(self.device)

    def _prompt_ids(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def stream(self, messages, gen_kwargs, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        inputs = self._prompt_ids(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        kwargs = dict(inputs)
        kwargs.update(gen_kwargs)
        kwargs["streamer"] = streamer

        def _run():
            with torch.inference_mode():
                try:
                    self.model.generate(**kwargs)
                except Exception as e:
                    # Surface the error to the UI and close the streamer.
                    streamer.text_queue.put(f"\n[error: {e}]")
                    streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        text = ""
        chunk_count = 0
        decode_t0: float | None = None
        tps = 0.0

        for chunk in streamer:
            text += chunk
            chunk_count += 1
            if decode_t0 is None:
                # First chunk arrived => prefill done; start decode timer here.
                decode_t0 = time.perf_counter()
            else:
                elapsed = time.perf_counter() - decode_t0
                tps = (chunk_count - 1) / elapsed if elapsed > 0 else 0.0
            yield WorkerEvent(text=text, tps=tps, accept_len=None, done=False)

        thread.join()
        yield WorkerEvent(
            text=text,
            tps=tps if decode_t0 is not None else 0.0,
            accept_len=None,
            done=True,
        )


class EagleWorker:
    """EAGLE-3 speculative decoding, streaming via EaModel.ea_generate."""

    def __init__(
        self,
        base_model_path: str,
        ea_model_path: str,
        device: str,
        label: str,
        total_token: int = 60,
        depth: int = 7,
        draft_top_k: int = 10,
    ):
        self.label = label
        self.device = device
        print(
            f"[{label}] loading EAGLE model: "
            f"target={base_model_path} draft={ea_model_path} device={device}"
        )
        self.model = EaModel.from_pretrained(
            use_eagle3=True,
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=total_token,
            depth=depth,
            top_k=draft_top_k,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        self.tokenizer = self.model.get_tokenizer()
        # Resolve the exact device placement of the base model.
        self._target_device = self.model.base_model.model.layers[0].self_attn.q_proj.weight.device
        # Sized for prompt (<=512 tok) + max_new_tokens (<=512) + tree decoding
        # overhead. Smaller than benchmark default (2048) to leave headroom on
        # shared 2080 Ti.
        self._max_length = 1280
        print(f"[{label}] ready on {self._target_device}")

    def warmup(self, n: int = 2):
        ids = self.tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(self._target_device)
        for _ in range(n):
            for _out in self.model.ea_generate(
                ids, temperature=0.0, top_p=0.0, top_k=0, max_new_tokens=16,
                max_length=self._max_length,
            ):
                pass
        torch.cuda.synchronize(self._target_device)

    def _prompt_ids(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self._target_device)

    def stream(self, messages, gen_kwargs, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        input_ids = self._prompt_ids(messages)
        input_len = input_ids.shape[1]

        # Release any fragmented cache from prior runs — on 11GB cards this
        # meaningfully reduces the risk of OOM when third-party processes share
        # the GPU.
        torch.cuda.empty_cache()

        temperature = float(gen_kwargs.get("temperature", 0.7))
        top_p = float(gen_kwargs.get("top_p", 0.8))
        top_k = int(gen_kwargs.get("top_k", 20))
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 256))

        steps = 0
        decode_t0: float | None = None
        tokens_at_t0 = 0
        tps = 0.0
        accept_len = 0.0
        text = ""
        eos_id = self.tokenizer.eos_token_id

        for output_ids in self.model.ea_generate(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            max_length=self._max_length,
        ):
            steps += 1
            new_cu_len = output_ids.shape[1]
            new_tokens = max(0, new_cu_len - input_len)

            # decode everything we have after the prompt so far
            decode_ids = output_ids[0, input_len:].tolist()
            if eos_id in decode_ids:
                decode_ids = decode_ids[: decode_ids.index(eos_id) + 1]
            text = self.tokenizer.decode(
                decode_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            # Skip the first speculative step from the TPS measurement — it
            # includes prefill + kv-cache warmup and would spike the rate.
            if decode_t0 is None:
                decode_t0 = time.perf_counter()
                tokens_at_t0 = new_tokens
                tps = 0.0
            else:
                elapsed = time.perf_counter() - decode_t0
                decoded_since = new_tokens - tokens_at_t0
                tps = decoded_since / elapsed if elapsed > 0 else 0.0
            accept_len = new_tokens / steps if steps > 0 else 0.0

            yield WorkerEvent(text=text, tps=tps, accept_len=accept_len, done=False)

        yield WorkerEvent(text=text, tps=tps, accept_len=accept_len, done=True)
