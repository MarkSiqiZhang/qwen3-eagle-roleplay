"""
Generate the hero Pareto plot for the poster:
In-domain acceptance length (x) vs Out-of-domain acceptance length (y).

Proposed-B is highlighted as the jointly-optimal point.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


# (label, in-domain metrics path, cross-domain metrics path, color, marker, is_hero)
CONFIGS = [
    ("B2'  (general draft)",
     "results/baseline2_finetuned_fixed256/metrics_summary.json",
     "results/cross_domain_b2p/metrics_summary.json",
     "#7f7f7f", "o", False),
    ("B3  (JS-only draft)",
     "results/baseline3_fixed256/metrics_summary.json",
     "results/cross_domain_b3/metrics_summary.json",
     "#d62728", "s", False),
    ("Proposed-A  (22% char)",
     "results/proposed_fixed256/metrics_summary.json",
     None,
     "#2ca02c", "^", False),
    ("Proposed-B  (50%, 8ep)",
     "results/proposed_balanced_fixed256/metrics_summary.json",
     "results/cross_domain_proposed_b/metrics_summary.json",
     "#1f77b4", "*", True),
    ("Proposed-C  (50%, 20ep)",
     "results/proposed_balanced_20ep_fixed256/metrics_summary.json",
     "results/cross_domain_proposed_c/metrics_summary.json",
     "#9467bd", "D", False),
]


def load_accept(path):
    if path is None:
        return None
    with open(path) as f:
        d = json.load(f)
    return d["metrics"]["avg_accept_length"]


def main():
    fig, ax = plt.subplots(figsize=(8.5, 6))

    points = []
    for label, js_path, uc_path, color, marker, hero in CONFIGS:
        js = load_accept(js_path)
        uc = load_accept(uc_path)
        if js is None or uc is None:
            continue
        size = 340 if hero else 180
        edgecolor = "gold" if hero else "black"
        lw = 1.6 if hero else 0.7
        ax.scatter([js], [uc], s=size, c=color, marker=marker,
                   edgecolor=edgecolor, linewidth=lw, zorder=5, label=label)
        points.append((js, uc, label))
        # label annotation — offsets tuned so no label overlaps any point
        offset_x, offset_y = 0.04, 0.02
        ha = "left"
        if "B2'" in label:
            offset_x, offset_y = 0.04, 0.0
            ha = "left"
        if "B3" in label:
            offset_x, offset_y = 0.05, 0.0
        if "Proposed-C" in label:
            offset_x, offset_y = 0.05, 0.0
        if "Proposed-B" in label:
            offset_x, offset_y = 0.05, 0.0
        ax.annotate(label.split("  ")[0],
                    (js + offset_x, uc + offset_y),
                    fontsize=11, fontweight="bold" if hero else "normal",
                    ha=ha, va="center")

    # Pareto-frontier: sort by x desc, track running max y
    sorted_pts = sorted(points, key=lambda p: -p[0])
    frontier_x, frontier_y = [], []
    running_max_y = -float("inf")
    for x, y, _ in sorted_pts:
        if y >= running_max_y:
            frontier_x.append(x)
            frontier_y.append(y)
            running_max_y = y
    # sort frontier by x for drawing
    frontier_x, frontier_y = zip(*sorted(zip(frontier_x, frontier_y)))
    ax.plot(frontier_x, frontier_y, "--", color="#1f77b4", alpha=0.45,
            linewidth=1.8, zorder=1, label="Pareto frontier")

    ax.set_xlabel("In-domain acceptance length\n(Jack Sparrow test set)", fontsize=12)
    ax.set_ylabel("Out-of-domain acceptance length\n(UltraChat 200K test_sft)", fontsize=12)
    ax.set_title("Joint Alignment: specialization vs generalization trade-off\nfor speculative-decoding draft models",
                 fontsize=13, pad=14)

    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

    # wider x-range so B3's marker + label have room on the right
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.set_xlim(3.2, 4.6)
    ax.set_ylim(min(ys) - 0.08, max(ys) + 0.18)

    # annotation box in upper-left (no data there)
    ax.text(0.02, 0.97,
            "Upper-right = better\nProposed-B (ours) is the only\npoint that dominates the\ngeneral-draft baseline on\nboth axes.",
            transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd",
                      edgecolor="#856404", linewidth=1))

    # Legend OUTSIDE plot (below axis) so it never covers data points.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=10, framealpha=0.95,
              handletextpad=0.5, columnspacing=1.2)

    plt.tight_layout()
    out = Path("results/pareto_plot.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out} and {out.with_suffix('.pdf')}")

    # also print the table
    print("\nData in the plot:")
    for label, js, uc in [(l, x, y) for x, y, l in points]:
        print(f"  {label:<28} JS={js:.2f}  UC={uc:.2f}")


if __name__ == "__main__":
    main()
