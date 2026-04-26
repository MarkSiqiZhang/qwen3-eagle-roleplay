"""
Compute reference-similarity metrics (BLEU, ROUGE-1/2/L) between generated
responses and RoleBench ground-truth references.

For quality evaluation we need natural-length outputs. Configs that have a
free-length run use that file directly; configs that only have a fixed-256
run use it with trailing whitespace stripped (those generations reach natural
EOS and then pad with newlines to satisfy min_new_tokens).
"""

import argparse
import json
import re
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer


IN_DOMAIN_CONFIGS = [
    # name, path, strip_trailing_pad
    ("B1",           "results/baseline1/generation_outputs.jsonl",              False),
    ("B1'",          "results/baseline1_prime/generation_outputs.jsonl",        False),
    ("B2",           "results/baseline2/generation_outputs.jsonl",              False),
    ("B2'",          "results/baseline2_finetuned/generation_outputs.jsonl",    False),
    ("B3",           "results/baseline3/generation_outputs.jsonl",              False),
    ("Proposed-A",   "results/proposed/generation_outputs.jsonl",               False),
    ("Proposed-B",   "results/proposed_balanced/generation_outputs.jsonl",      False),
    ("Proposed-C",   "results/proposed_balanced_20ep/generation_outputs.jsonl", False),
]

CROSS_DOMAIN_CONFIGS = [
    ("B1-UC",        "results/cross_domain_b1/generation_outputs.jsonl",           False),
    ("B1'-UC",       "results/cross_domain_b1p/generation_outputs.jsonl",          False),
    ("B2'-UC",       "results/cross_domain_b2p/generation_outputs.jsonl",          True),
    ("B3-UC",        "results/cross_domain_b3/generation_outputs.jsonl",           True),
    ("Proposed-B-UC","results/cross_domain_proposed_b/generation_outputs.jsonl",   True),
    ("Proposed-C-UC","results/cross_domain_proposed_c/generation_outputs.jsonl",   True),
]


def clean(text, strip_pad):
    if strip_pad:
        # strip trailing whitespace / newline padding used to reach min_new_tokens
        text = text.rstrip()
        # also collapse any run of 5+ newlines in the middle (degenerate repeat pad)
        text = re.sub(r"\n{5,}", "\n\n", text)
    return text


def load_pairs(path, strip_pad):
    refs, hyps = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            refs.append(d["reference"])
            hyps.append(clean(d["generated"], strip_pad))
    return refs, hyps


def evaluate(refs, hyps):
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = 0.0, 0.0, 0.0
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rL += s["rougeL"].fmeasure
    n = len(refs)
    avg_chars = sum(len(h) for h in hyps) / n
    return {
        "n": n,
        "avg_chars": round(avg_chars, 1),
        "bleu": round(bleu, 2),
        "rouge1": round(100 * r1 / n, 2),
        "rouge2": round(100 * r2 / n, 2),
        "rougeL": round(100 * rL / n, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/reference_similarity.json")
    args = parser.parse_args()

    all_results = {}

    def run_block(title, configs):
        print(f"\n{title}")
        print(f"{'Config':<16} {'N':>5} {'Chars':>7} {'BLEU':>7} {'R-1':>7} {'R-2':>7} {'R-L':>7}")
        print("-" * 66)
        for name, path, strip_pad in configs:
            p = Path(path)
            if not p.exists():
                print(f"{name:<16} MISSING: {path}")
                continue
            refs, hyps = load_pairs(p, strip_pad)
            m = evaluate(refs, hyps)
            all_results[name] = {"path": path, "strip_pad": strip_pad, **m}
            print(f"{name:<16} {m['n']:>5} {m['avg_chars']:>7.1f} {m['bleu']:>7.2f} "
                  f"{m['rouge1']:>7.2f} {m['rouge2']:>7.2f} {m['rougeL']:>7.2f}")

    run_block("=== In-domain (Jack Sparrow test, RoleBench reference) ===", IN_DOMAIN_CONFIGS)
    run_block("=== Cross-domain (UltraChat test_sft, generic assistant reference) ===", CROSS_DOMAIN_CONFIGS)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
