"""calibration — agreement-calibration ECE without human ground truth.

For each detection, the "true positive rate" is the empirical fraction
of frames in a ±W window where the SAME detection persists.  Bin by
declared score → empirical persistence rate → reliability diagram +
ECE.  This is the Sohn et al. weakly-supervised calibration estimator.

Output:
  paper/figures/reliability_diagram.png
  paper/figures/calibration_summary.png
  paper/calibration_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from experiments.futbotmx.evaluation.palette import (
    apply_paper_style, watermark,
    INK, INK_SOFT, TXT, TXT_MUTED, TXT_DIM, LINE,
    MINT, AMBER, ROSE, TEAL, LILAC, GOLD,
)
from experiments.futbotmx.evaluation.pseudo_gt import iou


def _persistence_for(per_frame: List[Dict], i: int, cls: str,
                     bbox: Tuple[float, float, float, float],
                     window: int, iou_thresh: float) -> int:
    """How many neighbour frames (excluding centre) have a same-class bbox
    with IoU ≥ τ to `bbox`."""
    count = 0
    for di in range(-window, window + 1):
        if di == 0:
            continue
        j = i + di
        if j < 0 or j >= len(per_frame):
            continue
        cands = per_frame[j].get("detections", {}).get(cls, [])
        best = 0.0
        for d in cands:
            bb = d.get("bbox")
            if not bb:
                continue
            v = iou(bbox, tuple(bb))
            if v > best:
                best = v
        if best >= iou_thresh:
            count += 1
    return count


def evaluate(cache: Dict, *, window: int = 3, iou_thresh: float = 0.30,
             persistence_threshold: int = 3, n_bins: int = 10,
             ) -> Dict:
    """Reliability + ECE per class + overall.

    A detection is a "positive" iff its persistence ≥ `persistence_threshold`
    over a ±window neighbour count."""
    per_frame = cache["per_frame"]
    per_class_pairs: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    all_pairs: List[Tuple[float, bool]] = []

    n_window = 2 * window           # number of neighbours considered
    pos_thresh = persistence_threshold

    for i, fr in enumerate(per_frame):
        for cls, dets in fr.get("detections", {}).items():
            for d in dets:
                bb = d.get("bbox")
                if not bb:
                    continue
                s = float(d.get("score", 0.0))
                persistence = _persistence_for(per_frame, i, cls,
                                                tuple(bb), window, iou_thresh)
                is_positive = persistence >= pos_thresh
                per_class_pairs[cls].append((s, is_positive))
                all_pairs.append((s, is_positive))

    def _ece(pairs: List[Tuple[float, bool]], n_bins: int = n_bins
             ) -> Tuple[float, List[Dict]]:
        if not pairs:
            return 0.0, []
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins_data = []
        ece = 0.0
        N = len(pairs)
        for k in range(n_bins):
            lo, hi = edges[k], edges[k + 1]
            in_bin = [(s, p) for (s, p) in pairs if (lo <= s < hi if k < n_bins - 1
                                                     else lo <= s <= hi)]
            if not in_bin:
                bins_data.append({"lo": float(lo), "hi": float(hi),
                                  "n": 0, "mean_score": 0.0, "empirical_pos": 0.0})
                continue
            mean_s = float(np.mean([s for s, _ in in_bin]))
            emp_p  = float(np.mean([1 if p else 0 for _, p in in_bin]))
            gap    = abs(mean_s - emp_p)
            ece   += (len(in_bin) / N) * gap
            bins_data.append({"lo": float(lo), "hi": float(hi),
                              "n": len(in_bin), "mean_score": round(mean_s, 4),
                              "empirical_pos": round(emp_p, 4),
                              "gap": round(gap, 4)})
        return round(float(ece), 4), bins_data

    per_class_results: Dict[str, Dict] = {}
    for cls, pairs in per_class_pairs.items():
        ece, bins = _ece(pairs)
        per_class_results[cls] = {
            "n_detections":         len(pairs),
            "n_positive":           int(sum(1 for _, p in pairs if p)),
            "empirical_pos_rate":   round(float(np.mean(
                [1 if p else 0 for _, p in pairs])), 4),
            "ece":                  ece,
            "bins":                 bins,
        }
    overall_ece, overall_bins = _ece(all_pairs)
    return {
        "schema":              "futbotmx.calibration.v1",
        "window":              window,
        "iou_thresh":          iou_thresh,
        "persistence_threshold": persistence_threshold,
        "n_bins":              n_bins,
        "overall": {
            "n_detections":       len(all_pairs),
            "ece":                overall_ece,
            "bins":               overall_bins,
        },
        "per_class": per_class_results,
    }


def plot_reliability(report: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    palette = {"robot": MINT, "ball": GOLD, "field": TEAL,
                "hand": LILAC, "goal": ROSE}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4),
                              gridspec_kw={"width_ratios": [1, 1]})

    # ── left: overall reliability diagram (bin bars) ──
    ax = axes[0]
    bins = report["overall"]["bins"]
    centres = [(b["lo"] + b["hi"]) / 2 for b in bins]
    widths  = [b["hi"] - b["lo"] - 0.005 for b in bins]
    emp     = [b["empirical_pos"] for b in bins]
    counts  = [b["n"] for b in bins]
    max_n = max(counts) if counts else 1
    alphas  = [0.4 + 0.6 * (n / max(max_n, 1)) for n in counts]

    for c, w, e, a in zip(centres, widths, emp, alphas):
        ax.bar(c, e, width=w, color=MINT, alpha=a, edgecolor=LINE, linewidth=0.6)
    ax.plot([0, 1], [0, 1], color=TXT, linestyle="--", linewidth=1.0,
            label="perfect calibration")
    ece = report["overall"]["ece"]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("declared SAM 3 score (binned)")
    ax.set_ylabel("empirical persistence rate")
    ax.set_title(f"Overall reliability  ·  ECE = {ece:.4f}",
                 loc="left", pad=12)
    ax.legend(loc="upper left", frameon=True)

    # ── right: per-class ECE bar chart ──
    ax = axes[1]
    classes = sorted(report["per_class"].keys())
    eces = [report["per_class"][c]["ece"] for c in classes]
    cols = [palette.get(c, AMBER) for c in classes]
    bars = ax.barh(classes, eces, color=cols, edgecolor=LINE, linewidth=0.6,
                    alpha=0.88)
    for b, v, cls in zip(bars, eces, classes):
        n = report["per_class"][cls]["n_detections"]
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:.3f}  (n={n})", va="center", color=TXT, fontsize=10.5)
    ax.set_xlabel("Expected Calibration Error (ECE) — lower is better")
    ax.set_title("Per-class ECE", loc="left", pad=12)
    ax.set_xlim(0, max(eces + [0.01]) * 1.3)

    watermark(fig, note="agreement-calibration")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--iou-thresh", type=float, default=0.30)
    ap.add_argument("--persistence", type=int, default=3)
    args = ap.parse_args()

    if not args.cache.exists():
        print(f"ERROR: missing {args.cache}", file=sys.stderr); return 1

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    report = evaluate(cache, window=args.window,
                      iou_thresh=args.iou_thresh,
                      persistence_threshold=args.persistence)

    out_dir = _HERE.parents[0] / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir.parent / "calibration_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    apply_paper_style()
    plot_reliability(report, out_dir / "reliability_diagram.png")

    print(f"[cal] overall ECE = {report['overall']['ece']}")
    for cls, v in report["per_class"].items():
        print(f"  {cls:8s} ECE={v['ece']:.4f}  pos_rate={v['empirical_pos_rate']:.3f}  "
              f"n={v['n_detections']}")
    print(f"[cal] reliability → {out_dir / 'reliability_diagram.png'}")
    print(f"[cal] metrics     → {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
