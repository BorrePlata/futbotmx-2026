"""cognitive — paper-grade plots of the U-CogNet cognitive trajectory.

Consumes a `<stem>_ucognet*.metrics.json` and renders three figures:

  • cognitive_trajectory.png  — F, φ, recon, understanding side-by-side
  • surprise_events.png       — surprise z-score timeline with model-break
                                markers and refractory periods shaded
  • cognitive_summary.png     — KPI panel (one card per scalar summary)

The figures are FULLY DERIVED from the cognitive layer's own self-report
(no human GT) and are exactly what `paper/PAPER.md` consumes.
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
    MINT, MINT_SOFT, AMBER, AMBER_SOFT, ROSE, LILAC, TEAL, GOLD,
)


# ── data helpers ────────────────────────────────────────────────
def _load(metrics_json: Path) -> Dict:
    with open(metrics_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _series(per_frame: List[Dict], key: str) -> np.ndarray:
    return np.array([float(r.get(key, 0.0)) for r in per_frame], dtype=np.float32)


def _bool_series(per_frame: List[Dict], key: str) -> np.ndarray:
    return np.array([bool(r.get(key, False)) for r in per_frame], dtype=bool)


# ── plot 1: cognitive trajectory (4 stacked panels) ─────────────
def plot_trajectory(per_frame: List[Dict], out: Path, *,
                    arm_label: str = "U-CogNet") -> None:
    import matplotlib.pyplot as plt

    t = _series(per_frame, "timestamp_s")
    if t.max() == 0:
        t = np.arange(len(per_frame))
        x_label = "frame"
    else:
        x_label = "time (s)"

    F   = _series(per_frame, "free_energy")
    phi = _series(per_frame, "phi")
    rec = _series(per_frame, "recon_error")
    und = _series(per_frame, "understanding")
    sz  = _series(per_frame, "surprise_z")
    brk = _bool_series(per_frame, "is_surprised")
    warm = _bool_series(per_frame, "warming_up")

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True,
                              gridspec_kw={"hspace": 0.32})

    # ── panel 1: F (free energy) ──
    ax = axes[0]
    ax.plot(t, F, color=MINT, linewidth=1.8, label="free energy $F$")
    ax.fill_between(t, F.min(), F, color=MINT, alpha=0.10)
    ax.set_ylabel("$F$ (free energy)", color=MINT)
    ax.tick_params(axis="y", colors=MINT)
    ax.set_title(f"U-CogNet cognitive trajectory  ·  {arm_label}",
                 loc="left", pad=12)
    if warm.any():
        warm_end = np.where(~warm)[0][0] if (~warm).any() else len(t)
        ax.axvspan(t[0], t[max(warm_end - 1, 0)],
                   color=LINE, alpha=0.5, label="warmup")
    ax.legend(loc="upper right", frameon=True)

    # ── panel 2: phi (integrated information) ──
    ax = axes[1]
    ax.plot(t, phi, color=LILAC, linewidth=1.8, label=r"integrated info $\phi$")
    ax.fill_between(t, 0, phi, color=LILAC, alpha=0.10)
    ax.set_ylabel(r"$\phi$ (integrated info)", color=LILAC)
    ax.tick_params(axis="y", colors=LILAC)
    ax.legend(loc="upper right", frameon=True)

    # ── panel 3: reconstruction error + understanding ──
    ax = axes[2]
    ax.plot(t, rec, color=TEAL, linewidth=1.6, label="recon error")
    ax.fill_between(t, 0, rec, color=TEAL, alpha=0.10)
    ax.set_ylabel("recon error", color=TEAL)
    ax.tick_params(axis="y", colors=TEAL)
    ax2 = ax.twinx()
    ax2.plot(t, und * 100, color=GOLD, linewidth=1.4, linestyle=":",
             label="understanding %")
    ax2.set_ylabel("understanding (%)", color=GOLD)
    ax2.tick_params(axis="y", colors=GOLD)
    ax2.set_ylim(0, 105)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True)

    # ── panel 4: surprise z + model break markers ──
    ax = axes[3]
    ax.plot(t, sz, color=ROSE, linewidth=1.5, label="surprise $z$")
    # threshold band
    ax.axhline(2.0, color=ROSE, linestyle="--", linewidth=0.9, alpha=0.5,
               label="surprise threshold")
    ax.axhline(-2.0, color=ROSE, linestyle="--", linewidth=0.9, alpha=0.5)
    # model break markers
    if brk.any():
        ax.scatter(t[brk], sz[brk], color=ROSE, s=70, zorder=5,
                   edgecolors=TXT, linewidths=1.2,
                   label=f"model break ({int(brk.sum())} events)")
        for ti in t[brk]:
            ax.axvline(ti, color=ROSE, alpha=0.15, linewidth=1)
    if warm.any():
        warm_end = np.where(~warm)[0][0] if (~warm).any() else len(t)
        ax.axvspan(t[0], t[max(warm_end - 1, 0)], color=LINE, alpha=0.4)
    ax.set_ylabel("surprise $z$", color=ROSE)
    ax.tick_params(axis="y", colors=ROSE)
    ax.set_xlabel(x_label)
    ax.legend(loc="upper right", frameon=True)

    watermark(fig, arm=arm_label, note="cognitive trajectory")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ── plot 2: surprise events timeline (clean, paper-grade) ───────
def plot_surprise(per_frame: List[Dict], out: Path, *,
                  arm_label: str = "U-CogNet") -> None:
    import matplotlib.pyplot as plt

    t   = _series(per_frame, "timestamp_s")
    sz  = _series(per_frame, "surprise_z")
    brk = _bool_series(per_frame, "is_surprised")
    if t.max() == 0:
        t = np.arange(len(per_frame))

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))
    ax.fill_between(t, 0, sz, where=(sz >= 0), color=ROSE, alpha=0.25,
                     interpolate=True, label="surprise $z$ (positive)")
    ax.fill_between(t, 0, sz, where=(sz < 0), color=TEAL, alpha=0.20,
                     interpolate=True, label="surprise $z$ (negative)")
    ax.plot(t, sz, color=ROSE, linewidth=1.4)
    ax.axhline(2.0,  color=GOLD, linestyle="--", linewidth=0.9,
               label="threshold $|z| = 2$")
    ax.axhline(-2.0, color=GOLD, linestyle="--", linewidth=0.9)
    if brk.any():
        ax.scatter(t[brk], sz[brk], color=ROSE, s=120, zorder=5,
                   edgecolors=TXT, linewidths=1.3,
                   label=f"model break events ({int(brk.sum())})")
    ax.set_xlabel("time (s)" if t.max() != len(t) - 1 else "frame")
    ax.set_ylabel("surprise $z$-score")
    ax.set_title(f"RealtimeReasoner surprise · {arm_label}",
                 loc="left", pad=12)
    ax.legend(loc="upper right", frameon=True)
    watermark(fig, arm=arm_label, note="surprise events")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ── plot 3: cognitive summary KPI cards ─────────────────────────
def plot_kpi(summary_cog: Dict, out: Path, *,
              arm_label: str = "U-CogNet") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cards = [
        ("$F$ final",        f"{summary_cog.get('free_energy_final', 0):+.3f}",
         "free energy",      MINT),
        (r"$\phi$ final",    f"{summary_cog.get('phi_final', 0):.3f}",
         "integrated information", LILAC),
        ("recon error",      f"{summary_cog.get('recon_error_final', 0):.4f}",
         "↓ vs initial "
         f"{summary_cog.get('recon_error_initial', 0):.4f}", TEAL),
        ("understanding",    f"{summary_cog.get('understanding_final', 0)*100:.0f} %",
         "self-prediction quality", GOLD),
        ("model breaks",     f"{summary_cog.get('n_model_breaks', 0)}",
         f"max $z$ = {summary_cog.get('max_surprise_z', 0):+.2f}", ROSE),
        ("frames observed",  f"{summary_cog.get('frames_observed', 0)}",
         f"post-warmup {summary_cog.get('frames_post_warmup', 0)}", AMBER),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.5),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.30})
    for ax, (label, value, sub, colour) in zip(axes.ravel(), cards):
        ax.axis("off")
        # subtle card background
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.03, 0.05), 0.94, 0.90, boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=INK_SOFT, edgecolor=LINE, linewidth=1,
            transform=ax.transAxes))
        # left accent stripe
        ax.add_patch(mpatches.Rectangle(
            (0.03, 0.10), 0.012, 0.80,
            facecolor=colour, transform=ax.transAxes, edgecolor="none"))
        ax.text(0.10, 0.72, label, color=TXT_MUTED, fontsize=11,
                 transform=ax.transAxes)
        ax.text(0.10, 0.40, value, color=colour, fontsize=30,
                 weight="bold", transform=ax.transAxes)
        ax.text(0.10, 0.18, sub, color=TXT_DIM, fontsize=10,
                 transform=ax.transAxes)
    fig.suptitle(f"U-CogNet cognitive summary  ·  {arm_label}",
                  color=TXT, fontsize=15, weight="bold", y=0.99,
                  ha="left", x=0.04)
    watermark(fig, arm=arm_label, note="cognitive summary")
    fig.savefig(out, bbox_inches="tight", facecolor=INK)
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=Path, required=True,
                    help="*_ucognet*.metrics.json from the ucognet pipeline")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to paper/figures/")
    ap.add_argument("--arm-label", default="U-CogNet")
    args = ap.parse_args()

    if not args.metrics.exists():
        print(f"ERROR: metrics not found {args.metrics}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or (_HERE.parents[0] / "paper" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()
    data = _load(args.metrics)
    per_frame = data["per_frame"]
    summary   = data["summary"].get("cognitive", {})

    paths = {
        "trajectory": out_dir / "cognitive_trajectory.png",
        "surprise":   out_dir / "surprise_events.png",
        "kpi":        out_dir / "cognitive_summary.png",
    }
    plot_trajectory(per_frame, paths["trajectory"], arm_label=args.arm_label)
    plot_surprise(per_frame, paths["surprise"],     arm_label=args.arm_label)
    plot_kpi(summary, paths["kpi"],                   arm_label=args.arm_label)

    for name, p in paths.items():
        kb = p.stat().st_size / 1024
        print(f"[cog] {name:11s} → {p}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
