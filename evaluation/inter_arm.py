"""inter_arm — what the cognitive layer ADDS over the SAM-3 baseline.

Both arms run the same SAM 3 perception, so the per-frame DETECTIONS
agree by construction.  The interesting signal is what U-CogNet
PRODUCES BEYOND the detections: model-break events the baseline can't
flag, cognitive uncertainty quantification, and an autonomous spatial
read.

Output:
  paper/figures/inter_arm_addition.png  — visual diff: baseline vs. U-CogNet
                                            on the same frame-level timeline
  paper/inter_arm_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from experiments.futbotmx.evaluation.palette import (
    apply_paper_style, watermark,
    INK, INK_SOFT, TXT, TXT_MUTED, TXT_DIM, LINE,
    MINT, AMBER, ROSE, TEAL, LILAC, GOLD,
)


def _arr(per_frame: List[Dict], key: str) -> np.ndarray:
    return np.array([float(r.get(key, 0.0)) for r in per_frame], dtype=np.float32)


def _bool_arr(per_frame: List[Dict], key: str) -> np.ndarray:
    return np.array([bool(r.get(key, False)) for r in per_frame], dtype=bool)


def evaluate(baseline_metrics: Dict, ucognet_metrics: Dict) -> Dict:
    b_pf = baseline_metrics["per_frame"]
    u_pf = ucognet_metrics["per_frame"]

    # Detection agreement (counts should be identical since same SAM 3)
    classes = sorted(set(list(b_pf[0]["per_class_counts"].keys())
                          + list(u_pf[0]["per_class_counts"].keys())))
    agreement_per_class: Dict[str, float] = {}
    for cls in classes:
        b_counts = np.array([r["per_class_counts"].get(cls, 0) for r in b_pf])
        u_counts = np.array([r["per_class_counts"].get(cls, 0) for r in u_pf])
        n_frames = min(len(b_counts), len(u_counts))
        agree = float(np.mean(b_counts[:n_frames] == u_counts[:n_frames]))
        agreement_per_class[cls] = round(agree, 4)

    # What U-CogNet adds: cognitive flags
    n = min(len(b_pf), len(u_pf))
    breaks = _bool_arr(u_pf[:n], "is_surprised")
    warming = _bool_arr(u_pf[:n], "warming_up")
    spatial_reads = [r.get("spatial_read", "") for r in u_pf[:n]
                      if r.get("spatial_read")]
    n_breaks = int(breaks.sum())
    n_breaks_post_warmup = int((breaks & ~warming).sum())

    # Latency cost
    b_lat = _arr(b_pf, "infer_ms")[:n]
    u_lat = _arr(u_pf, "infer_ms")[:n] + _arr(u_pf, "cognitive_ms")[:n]

    return {
        "schema":          "futbotmx.inter_arm.v1",
        "frames_compared": n,
        "detection_agreement_per_class": agreement_per_class,
        "ucognet_adds": {
            "model_break_events":          n_breaks,
            "model_break_post_warmup":     n_breaks_post_warmup,
            "spatial_reads_emitted":       len(spatial_reads),
            "spatial_read_sample":         spatial_reads[:3],
            "free_energy_final":           float(u_pf[-1].get("free_energy", 0.0)),
            "phi_final":                   float(u_pf[-1].get("phi", 0.0)),
            "understanding_final":         float(u_pf[-1].get("understanding", 0.0)),
        },
        "latency_overhead": {
            "mean_ms":  round(float(np.mean(u_lat - b_lat)), 2),
            "p95_ms":   round(float(np.percentile(u_lat - b_lat, 95)), 2),
            "mean_pct": round(float(np.mean(u_lat - b_lat) / max(np.mean(b_lat), 1e-6) * 100), 2),
        },
    }


def plot_addition(baseline_metrics: Dict, ucognet_metrics: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    b_pf = baseline_metrics["per_frame"]
    u_pf = ucognet_metrics["per_frame"]
    n = min(len(b_pf), len(u_pf))
    t = _arr(u_pf[:n], "timestamp_s")
    if t.max() == 0:
        t = np.arange(n)

    breaks = _bool_arr(u_pf[:n], "is_surprised")
    sz     = _arr(u_pf[:n], "surprise_z")
    u_lat  = _arr(u_pf[:n], "infer_ms") + _arr(u_pf[:n], "cognitive_ms")
    b_lat  = _arr(b_pf[:n], "infer_ms")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                              gridspec_kw={"hspace": 0.30, "height_ratios": [1.4, 1]})

    # ── panel 1: what U-CogNet sees that baseline doesn't ──
    ax = axes[0]
    # baseline "blind" timeline — empty
    ax.fill_between(t, -0.05, 0.05, color=AMBER, alpha=0.20,
                     label="BASELINE (no cognitive signal)")
    ax.axhline(0, color=AMBER, linewidth=0.6, alpha=0.5)
    # U-CogNet surprise
    ax.plot(t, sz, color=MINT, linewidth=1.5, label="U-CogNet surprise $z$")
    ax.fill_between(t, 0, sz, where=(sz > 0), color=MINT, alpha=0.18, interpolate=True)
    if breaks.any():
        ax.scatter(t[breaks], sz[breaks], color=ROSE, s=110, zorder=5,
                    edgecolors=TXT, linewidths=1.3,
                    label=f"model break ({int(breaks.sum())}) — baseline-invisible")
    ax.set_ylabel("surprise $z$")
    ax.set_title("What U-CogNet sees · the baseline arm has no model of the match",
                 loc="left", pad=12)
    ax.legend(loc="upper right", frameon=True)

    # ── panel 2: latency overhead ──
    ax = axes[1]
    ax.plot(t, b_lat, color=AMBER, linewidth=1.3, label="BASELINE infer")
    ax.plot(t, u_lat, color=MINT, linewidth=1.3, label="U-CogNet total")
    ax.fill_between(t, b_lat, u_lat, color=MINT, alpha=0.15,
                     label="cognitive overhead")
    ax.set_xlabel("time (s)" if t.max() != n - 1 else "frame")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Cost of the cognitive layer", loc="left", pad=8)
    ax.legend(loc="upper right", frameon=True)

    watermark(fig, note="inter-arm: what U-CogNet adds")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-metrics", type=Path, required=True)
    ap.add_argument("--ucognet-metrics",  type=Path, required=True)
    args = ap.parse_args()

    for p in (args.baseline_metrics, args.ucognet_metrics):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr); return 1

    b = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    u = json.loads(args.ucognet_metrics.read_text(encoding="utf-8"))

    res = evaluate(b, u)
    out_dir = _HERE.parents[0] / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir.parent / "inter_arm_metrics.json"
    metrics_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    apply_paper_style()
    plot_addition(b, u, out_dir / "inter_arm_addition.png")

    print(f"[arm] detection agreement: {res['detection_agreement_per_class']}")
    print(f"[arm] U-CogNet adds {res['ucognet_adds']['model_break_events']} model breaks "
          f"({res['ucognet_adds']['model_break_post_warmup']} post-warmup)")
    print(f"[arm] latency overhead: +{res['latency_overhead']['mean_ms']:.0f} ms mean "
          f"(+{res['latency_overhead']['mean_pct']:.1f}%)")
    print(f"[arm] figure → {out_dir / 'inter_arm_addition.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
