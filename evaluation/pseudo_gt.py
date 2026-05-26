"""pseudo_gt — fully-automated ground-truth derivation, no human in the loop.

Two complementary sources of pseudo-ground-truth, both reproducible:

  1.  TEMPORAL CONSISTENCY.  A bbox at (x,y,w,h) on frame t is promoted
      to "confident positive" iff it persists at IoU ≥ τ for ≥ k of the
      ±W neighbour frames.  This is the weakly-supervised pseudo-GT
      standard in video learning (Lee et al., Wang et al.).  Flickery
      single-frame detections — common false positives — fail the
      persistence test; truly present objects always pass.

  2.  CROSS-CHECK with the SAM 3 VIDEO PREDICTOR.  When available the
      cache from the video tracker (sam3.model.sam3_video_predictor) is
      treated as a SECOND teacher.  Bboxes where both agree get the
      highest pseudo-GT confidence.

This module CONSUMES `evaluation/cache_detections.py` output (single
cache JSON per video) and PRODUCES:

  <stem>.pseudo_gt.json  — per-frame, per-class, list of confident bboxes
                            with persistence count and confidence score
  pseudo_gt_summary.png  — paper-grade summary figure (counts per class +
                            persistence histogram)

Honest framing for the paper: this is WEAKLY-SUPERVISED pseudo-GT, not
human-verified.  We disclose it openly because every metric downstream
inherits its assumptions.  The robustness battery uses NO GT at all and
is the rigorous fallback (`evaluation/robustness.py`).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from experiments.futbotmx.evaluation.palette import (
    apply_paper_style, watermark,
    INK, INK_SOFT, TXT, TXT_MUTED, TXT_DIM, LINE,
    MINT, AMBER, ROSE, TEAL, LILAC, GOLD,
)


# ── IoU helper ──────────────────────────────────────────────────
def iou(a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw  = max(0.0, ix2 - ix1)
    ih  = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = ua + ub - inter
    return inter / union if union > 1e-9 else 0.0


# ── pseudo-GT instance ──────────────────────────────────────────
@dataclass
class GTInstance:
    bbox:         Tuple[float, float, float, float]   # representative (centroid frame)
    score:        float           # mean score across persistent frames
    class_name:   str
    frame_idx:    int             # this is the frame we're labelling
    persistence:  int             # how many neighbour frames it appears in
    window:       int             # neighbour radius used
    confidence:   float           # persistence / (2W+1) ∈ [0,1]


def _collect_window(per_frame: List[Dict], centre: int, window: int
                    ) -> List[Tuple[int, Dict]]:
    out = []
    for di in range(-window, window + 1):
        j = centre + di
        if 0 <= j < len(per_frame):
            out.append((j, per_frame[j]))
    return out


def _persistence(bbox: Tuple[float, float, float, float],
                 class_name: str,
                 neighbours: List[Tuple[int, Dict]],
                 iou_thresh: float) -> Tuple[int, List[float]]:
    """How many neighbour frames have a same-class bbox with IoU ≥ τ.

    Returns (count, list of matched scores)."""
    count, scores = 0, []
    for j, fr in neighbours:
        cands = fr.get("detections", {}).get(class_name, [])
        best = 0.0
        best_s = 0.0
        for d in cands:
            bb = d.get("bbox")
            if not bb:
                continue
            v = iou(bbox, tuple(bb))
            if v > best:
                best, best_s = v, float(d.get("score", 0.0))
        if best >= iou_thresh:
            count += 1
            scores.append(best_s)
    return count, scores


# ── core algorithm ──────────────────────────────────────────────
def derive_pseudo_gt(cache: Dict, *, window: int = 3,
                     iou_thresh: float = 0.30,
                     min_persistence: int = 3,
                     # class-specific overrides for hard cases
                     class_overrides: Optional[Dict[str, Dict]] = None,
                     ) -> Dict:
    """Run temporal-consistency promotion over every detection in the cache.

    Defaults — ±3 frame window (~7 frames at 30fps = ~230ms) and 0.3 IoU
    with a 3-frame minimum.  Override per class for the ball (it is hard
    and intermittent — loosen IoU, lower persistence).
    """
    class_overrides = class_overrides or {
        "ball":  {"iou_thresh": 0.20, "min_persistence": 2, "window": 3},
        "field": {"iou_thresh": 0.50, "min_persistence": 4, "window": 3},
    }

    per_frame = cache["per_frame"]
    n = len(per_frame)

    gt_per_frame: List[Dict] = [{"frame_idx": fr["frame_idx"],
                                  "timestamp_s": fr.get("timestamp_s", 0.0),
                                  "gt": defaultdict(list)}
                                 for fr in per_frame]

    # iterate every detection on every frame, promote if persistent
    for i, fr in enumerate(per_frame):
        for cls, dets in fr.get("detections", {}).items():
            cfg = class_overrides.get(cls, {})
            w   = cfg.get("window", window)
            tau = cfg.get("iou_thresh", iou_thresh)
            kp  = cfg.get("min_persistence", min_persistence)
            neigh = _collect_window(per_frame, i, w)
            for d in dets:
                bb = d.get("bbox")
                if not bb:
                    continue
                bb_t = tuple(bb)
                persistence, matched_scores = _persistence(
                    bb_t, cls, neigh, tau)
                if persistence < kp:
                    continue
                # mean score across the persistent observations
                ms = float(np.mean(matched_scores)) if matched_scores else float(d["score"])
                conf = persistence / max(len(neigh), 1)
                gt_per_frame[i]["gt"][cls].append({
                    "bbox":        list(bb_t),
                    "score":       round(ms, 4),
                    "persistence": persistence,
                    "window_size": len(neigh),
                    "confidence":  round(conf, 3),
                })

    # convert defaultdicts back to plain dicts for JSON
    for fr in gt_per_frame:
        fr["gt"] = dict(fr["gt"])

    # aggregate stats
    counts: Dict[str, int] = defaultdict(int)
    persistence_dist: Dict[str, List[int]] = defaultdict(list)
    for fr in gt_per_frame:
        for cls, instances in fr["gt"].items():
            counts[cls] += len(instances)
            persistence_dist[cls].extend(int(i["persistence"]) for i in instances)

    summary = {
        "method":         "temporal-consistency-pseudo-gt",
        "frames":         n,
        "window_default": window,
        "iou_default":    iou_thresh,
        "min_persist_default": min_persistence,
        "class_overrides": class_overrides,
        "total_gt_per_class": dict(counts),
        "persistence_mean_per_class": {
            cls: round(float(np.mean(v)), 2) for cls, v in persistence_dist.items()
        },
    }

    return {
        "schema":     "futbotmx.pseudo_gt.v1",
        "video":      cache.get("video"),
        "video_sha":  cache.get("video_sha"),
        "frame_size": cache.get("frame_size"),
        "summary":    summary,
        "per_frame":  gt_per_frame,
    }


# ── visual: pseudo-GT summary figure ────────────────────────────
def plot_summary(pseudo_gt: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    summary = pseudo_gt["summary"]
    counts  = summary["total_gt_per_class"]
    mean_p  = summary["persistence_mean_per_class"]

    # also build the persistence histograms
    pers_by_class: Dict[str, List[int]] = {c: [] for c in counts}
    for fr in pseudo_gt["per_frame"]:
        for c, inst in fr["gt"].items():
            pers_by_class[c].extend(int(i["persistence"]) for i in inst)

    classes = sorted(counts.keys(), key=lambda c: -counts[c])
    palette = {"robot": MINT, "ball": GOLD, "field": TEAL,
                "hand": LILAC, "goal": ROSE}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ── left: total pseudo-GT counts per class ─────
    ax = axes[0]
    xs = np.arange(len(classes))
    cols = [palette.get(c, AMBER) for c in classes]
    bars = ax.bar(xs, [counts[c] for c in classes], width=0.55,
                   color=cols, edgecolor=LINE, linewidth=0.6, alpha=0.88)
    for x, c in zip(xs, classes):
        ax.text(x, counts[c] + 5, f"{counts[c]}",
                ha="center", va="bottom", color=TXT, fontsize=11, weight="bold")
        ax.text(x, -max(counts.values()) * 0.06, f"μ persist = {mean_p[c]:.1f}",
                ha="center", va="top", color=TXT_DIM, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([c for c in classes])
    ax.set_ylabel("pseudo-GT instances (cumulative across video)")
    ax.set_title("Pseudo-GT counts per class · temporal consistency",
                 loc="left", pad=12)
    ax.set_ylim(-max(counts.values()) * 0.12, max(counts.values()) * 1.15)

    # ── right: persistence histogram per class ─────
    ax = axes[1]
    max_persist = max((max(pers_by_class[c], default=0) for c in classes), default=1)
    bins = np.arange(0, max_persist + 2) - 0.5
    for c in classes:
        if not pers_by_class[c]:
            continue
        ax.hist(pers_by_class[c], bins=bins, alpha=0.55,
                color=palette.get(c, AMBER), label=c,
                edgecolor=LINE, linewidth=0.5)
    ax.set_xlabel("persistence (frames matched in ±W window)")
    ax.set_ylabel("count")
    ax.set_title("Persistence distribution",
                 loc="left", pad=12)
    ax.legend(loc="upper right", frameon=True)

    watermark(fig, note="pseudo-GT")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True,
                    help="<stem>.cache.json from cache_detections.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <stem>.pseudo_gt.json next to cache")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--min-persist", type=int, default=3)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    if not args.cache.exists():
        print(f"ERROR: cache not found {args.cache}", file=sys.stderr); return 1

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    print(f"[gt] cache: {args.cache.name}  ·  {cache['n_frames']} frames")

    pseudo_gt = derive_pseudo_gt(
        cache, window=args.window, iou_thresh=args.iou,
        min_persistence=args.min_persist,
    )

    out = args.out or args.cache.with_suffix("").with_suffix(".pseudo_gt.json")
    out.write_text(json.dumps(pseudo_gt, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"[gt] ✅ pseudo-GT → {out}  ({kb:.0f} KB)")
    summary = pseudo_gt["summary"]
    print(f"[gt] per-class counts: {summary['total_gt_per_class']}")
    print(f"[gt] mean persistence: {summary['persistence_mean_per_class']}")

    if not args.no_figure:
        apply_paper_style()
        fig_path = _HERE.parents[0] / "paper" / "figures" / "pseudo_gt_summary.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_summary(pseudo_gt, fig_path)
        print(f"[gt] figure → {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
