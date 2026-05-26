"""latency — per-layer wall-time budget for the FutBotMX pipelines.

Consumes the two metrics JSONs (baseline + ucognet), produces:

  • latency_budget.png      — stacked bar chart per arm with mean / p95 split
  • latency_distribution.png — histogram of per-frame infer-ms for each arm

This is a paper-grade figure (Section "Latency budget per layer" in
PAPER.md): demonstrates the U-CogNet arm pays a SMALL cognitive overhead
on top of the same SAM 3 inference, with the per-frame distribution
characterising tail risk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from experiments.futbotmx.evaluation.palette import (
    apply_paper_style, watermark,
    INK, INK_SOFT, TXT, TXT_MUTED, TXT_DIM, LINE,
    MINT, AMBER, ROSE, GOLD, TEAL,
)


def _load(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _arr(per_frame: List[Dict], key: str) -> np.ndarray:
    return np.array([float(r.get(key, 0.0)) for r in per_frame], dtype=np.float32)


def _stats(x: np.ndarray) -> Dict:
    return {
        "mean":   float(np.mean(x)),
        "median": float(np.percentile(x, 50)),
        "p90":    float(np.percentile(x, 90)),
        "p95":    float(np.percentile(x, 95)),
        "p99":    float(np.percentile(x, 99)),
        "max":    float(np.max(x)),
    }


def plot_budget(baseline_pf: List[Dict], ucognet_pf: List[Dict],
                out: Path) -> Dict:
    import matplotlib.pyplot as plt

    b_inf = _arr(baseline_pf, "infer_ms")
    u_inf = _arr(ucognet_pf,  "infer_ms")
    u_cog = _arr(ucognet_pf,  "cognitive_ms")
    u_total = u_inf + u_cog

    b_stats = _stats(b_inf)
    u_stats_inf  = _stats(u_inf)
    u_stats_cog  = _stats(u_cog)
    u_stats_tot  = _stats(u_total)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

    arms = ["BASELINE", "U-CogNet"]
    mean_sam = [b_stats["mean"], u_stats_inf["mean"]]
    mean_cog = [0.0,             u_stats_cog["mean"]]
    p95_sam  = [b_stats["p95"],  u_stats_inf["p95"]]
    p95_cog  = [0.0,             u_stats_cog["p95"]]

    x = np.arange(len(arms))
    w = 0.36

    # mean bars
    bars_sam_mean = ax.bar(x - w/2 - 0.005, mean_sam, width=w,
                            color=AMBER, alpha=0.85,
                            edgecolor=LINE, linewidth=0.6,
                            label="SAM 3 inference (mean)")
    bars_cog_mean = ax.bar(x - w/2 - 0.005, mean_cog, width=w,
                            bottom=mean_sam, color=MINT, alpha=0.85,
                            edgecolor=LINE, linewidth=0.6,
                            label="Cognitive layer (mean)")
    # p95 bars
    bars_sam_p95 = ax.bar(x + w/2 + 0.005, p95_sam, width=w,
                           color=AMBER, alpha=0.40,
                           hatch="//", edgecolor=LINE, linewidth=0.6,
                           label="SAM 3 inference (p95)")
    bars_cog_p95 = ax.bar(x + w/2 + 0.005, p95_cog, width=w,
                           bottom=p95_sam, color=MINT, alpha=0.40,
                           hatch="//", edgecolor=LINE, linewidth=0.6,
                           label="Cognitive layer (p95)")

    # totals at top
    for i, (m, p) in enumerate(zip(
        [mean_sam[0] + mean_cog[0], mean_sam[1] + mean_cog[1]],
        [p95_sam[0]  + p95_cog[0],  p95_sam[1]  + p95_cog[1]])):
        ax.text(i - w/2 - 0.005, m + 25, f"{m:.0f} ms",
                ha="center", color=TXT, fontsize=10.5, weight="bold")
        ax.text(i + w/2 + 0.005, p + 25, f"{p:.0f} ms",
                ha="center", color=TXT_MUTED, fontsize=10.5)

    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("wall-time per frame (ms)")
    ax.set_title("Latency budget per arm  ·  mean (solid) vs p95 (hatched)",
                 loc="left", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.set_ylim(0, max(p95_sam[1] + p95_cog[1], p95_sam[0]) * 1.20)

    watermark(fig, note="latency budget")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "baseline": {
            "infer_ms": b_stats,
            "total_ms": b_stats,           # baseline has no cognitive layer
        },
        "ucognet": {
            "infer_ms":     u_stats_inf,
            "cognitive_ms": u_stats_cog,
            "total_ms":     u_stats_tot,
        },
        "overhead_pct_mean": round(100 * u_stats_cog["mean"]
                                    / max(u_stats_inf["mean"], 1e-6), 1),
        "overhead_pct_p95":  round(100 * u_stats_cog["p95"]
                                    / max(u_stats_inf["p95"], 1e-6), 1),
    }


def plot_distribution(baseline_pf: List[Dict], ucognet_pf: List[Dict],
                       out: Path) -> None:
    import matplotlib.pyplot as plt

    b_inf  = _arr(baseline_pf, "infer_ms")
    u_inf  = _arr(ucognet_pf,  "infer_ms")
    u_cog  = _arr(ucognet_pf,  "cognitive_ms")
    u_tot  = u_inf + u_cog

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ── left: histograms overlaid ──
    ax = axes[0]
    bins = np.linspace(0, max(b_inf.max(), u_tot.max()) * 1.05, 36)
    ax.hist(b_inf, bins=bins, color=AMBER, alpha=0.55,
            label=f"BASELINE  (μ={b_inf.mean():.0f} · p95={np.percentile(b_inf,95):.0f})",
            edgecolor=LINE, linewidth=0.5)
    ax.hist(u_tot, bins=bins, color=MINT, alpha=0.55,
            label=f"U-CogNet   (μ={u_tot.mean():.0f} · p95={np.percentile(u_tot,95):.0f})",
            edgecolor=LINE, linewidth=0.5)
    ax.set_xlabel("per-frame latency (ms)")
    ax.set_ylabel("frame count")
    ax.set_title("Per-frame latency distribution",
                 loc="left", pad=10)
    ax.legend(loc="upper right", frameon=True)

    # ── right: per-frame trace ──
    ax = axes[1]
    t = np.arange(len(b_inf))
    ax.plot(t, b_inf, color=AMBER, linewidth=1.2, alpha=0.85,
            label="BASELINE infer")
    ax.plot(t, u_inf, color=MINT, linewidth=1.2, alpha=0.85,
            label="U-CogNet infer")
    ax.plot(t, u_cog, color=ROSE, linewidth=1.0, alpha=0.85, linestyle=":",
            label="cognitive layer only")
    ax.set_xlabel("frame")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency trace · per frame", loc="left", pad=10)
    ax.legend(loc="upper right", frameon=True)

    watermark(fig, note="latency distribution")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-metrics", type=Path, required=True)
    ap.add_argument("--ucognet-metrics",  type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    for p in (args.baseline_metrics, args.ucognet_metrics):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr); return 1
    out_dir = args.out_dir or (_HERE.parents[0] / "paper" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()
    b = _load(args.baseline_metrics)["per_frame"]
    u = _load(args.ucognet_metrics)["per_frame"]

    stats = plot_budget(b, u, out_dir / "latency_budget.png")
    plot_distribution(b, u, out_dir / "latency_distribution.png")

    # also dump the stats JSON for the paper to consume
    stats_path = out_dir.parent / "latency_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[lat] budget       → {out_dir / 'latency_budget.png'}")
    print(f"[lat] distribution → {out_dir / 'latency_distribution.png'}")
    print(f"[lat] stats        → {stats_path}")
    print(f"[lat] cognitive overhead: "
          f"+{stats['overhead_pct_mean']}% mean  ·  "
          f"+{stats['overhead_pct_p95']}% p95")
    return 0


if __name__ == "__main__":
    sys.exit(main())
