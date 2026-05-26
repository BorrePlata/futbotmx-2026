"""detection — IoU, precision/recall and mAP vs the temporal-consistency
pseudo-GT (no human labels).

Consumes (a) a detection cache and (b) the matching pseudo-GT.  Computes
per-class:

  • TP / FP / FN at multiple IoU thresholds (0.30 / 0.50 / 0.70)
  • precision, recall, F1
  • Average Precision (AP) via 11-point interpolation
  • per-frame detection counts overlaid with pseudo-GT counts

Produces:
  paper/figures/detection_pr_curves.png
  paper/figures/detection_per_class.png
  paper/figures/detection_count_overlay.png
  paper/detection_metrics.json

The honest framing in PAPER.md: these are metrics against a
weakly-supervised teacher.  The robustness battery and the cognitive
trajectory plots are the model-self-witness rigour that doesn't
inherit pseudo-GT assumptions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from experiments.futbotmx.evaluation.pseudo_gt import iou


# ── greedy IoU matching (per-frame, per-class) ──────────────────
def _match_one_class(preds: List[Dict], gts: List[Dict],
                     iou_thresh: float) -> Tuple[int, int, int, List[float]]:
    """Greedy 1-to-1 matching by descending score.  Returns
    (TP, FP, FN, list_of_matched_ious)."""
    if not preds and not gts:
        return 0, 0, 0, []
    if not gts:
        return 0, len(preds), 0, []
    if not preds:
        return 0, 0, len(gts), []

    preds_sorted = sorted(preds, key=lambda d: -float(d["score"]))
    gt_used = [False] * len(gts)
    tp, fp = 0, 0
    matched_ious: List[float] = []

    for p in preds_sorted:
        pbb = p.get("bbox")
        if not pbb:
            fp += 1
            continue
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if gt_used[j]:
                continue
            gbb = g.get("bbox")
            if not gbb:
                continue
            v = iou(tuple(pbb), tuple(gbb))
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp += 1
            gt_used[best_j] = True
            matched_ious.append(best_iou)
        else:
            fp += 1
    fn = sum(1 for u in gt_used if not u)
    return tp, fp, fn, matched_ious


def _ap_11point(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """PASCAL VOC-style 11-point AP interpolation."""
    if len(precisions) == 0:
        return 0.0
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        mask = recalls >= r
        p = float(precisions[mask].max()) if mask.any() else 0.0
        ap += p / 11.0
    return ap


def _pr_curve(preds_with_match: List[Tuple[float, bool]], n_gt: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by score desc, cumulate TP/FP → precision, recall arrays.

    `preds_with_match` is list of (score, is_TP).
    Returns (scores, precision, recall) parallel arrays.
    """
    if not preds_with_match or n_gt == 0:
        return np.array([]), np.array([]), np.array([])
    sorted_p = sorted(preds_with_match, key=lambda t: -t[0])
    scores = np.array([t[0] for t in sorted_p])
    is_tp  = np.array([1 if t[1] else 0 for t in sorted_p], dtype=np.int32)
    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(1 - is_tp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall    = tp_cum / max(n_gt, 1e-9)
    return scores, precision, recall


# ── full evaluation ─────────────────────────────────────────────
def evaluate(cache: Dict, pseudo_gt: Dict,
             iou_thresholds: Tuple[float, ...] = (0.30, 0.50, 0.70)
             ) -> Dict:
    pred_per_frame = cache["per_frame"]
    gt_per_frame   = pseudo_gt["per_frame"]
    if len(pred_per_frame) != len(gt_per_frame):
        print(f"WARN: cache has {len(pred_per_frame)} frames, "
              f"pseudo-GT has {len(gt_per_frame)}", file=sys.stderr)
    classes = sorted(set(
        list(pseudo_gt["summary"]["total_gt_per_class"].keys())
        + [c for fr in pred_per_frame for c in fr.get("detections", {}).keys()]
    ))

    out: Dict[str, Dict] = {
        "iou_thresholds": list(iou_thresholds),
        "per_class": {},
        "pr_curves": {},          # per-class @ iou=0.5 — for plotting
    }

    for cls in classes:
        per_iou: Dict[str, Dict] = {}
        pr_data_at_50: Optional[Tuple[List[Tuple[float, bool]], int]] = None
        for tau in iou_thresholds:
            tot_tp, tot_fp, tot_fn = 0, 0, 0
            ious_all: List[float] = []
            preds_with_match: List[Tuple[float, bool]] = []
            n_gt = 0
            for fi in range(min(len(pred_per_frame), len(gt_per_frame))):
                preds = pred_per_frame[fi].get("detections", {}).get(cls, [])
                gts   = gt_per_frame[fi].get("gt", {}).get(cls, [])
                tp, fp, fn, matched = _match_one_class(preds, gts, tau)
                tot_tp += tp; tot_fp += fp; tot_fn += fn
                ious_all.extend(matched)
                n_gt += len(gts)
                # PR curve accounting at iou=0.5
                if abs(tau - 0.50) < 1e-6:
                    # for each pred, decide TP/FP via the same greedy match
                    preds_sorted = sorted(preds, key=lambda d: -float(d["score"]))
                    gt_used = [False] * len(gts)
                    for p in preds_sorted:
                        pbb = p.get("bbox")
                        s = float(p.get("score", 0.0))
                        if not pbb:
                            preds_with_match.append((s, False))
                            continue
                        best_j, best_iou = -1, 0.0
                        for j, g in enumerate(gts):
                            if gt_used[j] or not g.get("bbox"):
                                continue
                            v = iou(tuple(pbb), tuple(g["bbox"]))
                            if v > best_iou:
                                best_iou, best_j = v, j
                        is_tp = (best_j >= 0 and best_iou >= 0.5)
                        if is_tp:
                            gt_used[best_j] = True
                        preds_with_match.append((s, is_tp))
            precision = tot_tp / max(tot_tp + tot_fp, 1e-9)
            recall    = tot_tp / max(tot_tp + tot_fn, 1e-9)
            f1        = 2 * precision * recall / max(precision + recall, 1e-9)
            mean_iou  = float(np.mean(ious_all)) if ious_all else 0.0
            per_iou[f"@{tau:.2f}"] = {
                "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "mean_iou_of_matches": round(mean_iou, 4),
                "n_gt": n_gt,
            }
            if abs(tau - 0.50) < 1e-6:
                pr_data_at_50 = (preds_with_match, n_gt)

        # AP@0.5 from PR curve
        ap50 = 0.0
        if pr_data_at_50 and pr_data_at_50[1] > 0:
            scores, prec, rec = _pr_curve(*pr_data_at_50)
            ap50 = _ap_11point(prec, rec)
            out["pr_curves"][cls] = {
                "scores":    scores.tolist(),
                "precision": prec.tolist(),
                "recall":    rec.tolist(),
                "ap@0.5":    round(float(ap50), 4),
            }
        out["per_class"][cls] = {
            "iou": per_iou,
            "ap@0.5": round(float(ap50), 4),
        }

    # mean AP (mAP) at 0.5 across classes with at least 1 GT
    aps = [v["ap@0.5"] for v in out["per_class"].values() if v["ap@0.5"] > 0]
    out["mAP@0.5"] = round(float(np.mean(aps)), 4) if aps else 0.0
    return out


# ── plots ───────────────────────────────────────────────────────
def plot_pr_curves(eval_res: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    palette = {"robot": MINT, "ball": GOLD, "field": TEAL,
                "hand": LILAC, "goal": ROSE}
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for cls, curve in eval_res.get("pr_curves", {}).items():
        prec = np.array(curve["precision"])
        rec  = np.array(curve["recall"])
        if len(prec) == 0:
            continue
        # sort by recall (monotone-ish) for cleaner plot
        order = np.argsort(rec)
        ax.plot(rec[order], prec[order],
                color=palette.get(cls, AMBER), linewidth=1.8,
                label=f"{cls}  AP={curve['ap@0.5']:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision-recall  ·  IoU=0.5  ·  mAP={eval_res['mAP@0.5']:.3f}",
                 loc="left", pad=12)
    ax.legend(loc="lower left", frameon=True)
    watermark(fig, note="detection vs pseudo-GT")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def plot_per_class(eval_res: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    classes = list(eval_res["per_class"].keys())
    palette = {"robot": MINT, "ball": GOLD, "field": TEAL,
                "hand": LILAC, "goal": ROSE}
    cols = [palette.get(c, AMBER) for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    metrics = [("precision", "Precision @ IoU=0.5"),
               ("recall",    "Recall @ IoU=0.5"),
               ("f1",        "F1 @ IoU=0.5")]
    for ax, (key, title) in zip(axes, metrics):
        vals = [eval_res["per_class"][c]["iou"]["@0.50"][key] for c in classes]
        bars = ax.bar(classes, vals, color=cols, edgecolor=LINE,
                       linewidth=0.6, alpha=0.88)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", color=TXT, fontsize=10.5,
                    weight="bold")
        ax.set_ylim(0, 1.10)
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel(key)
    watermark(fig, note="per-class detection")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def plot_count_overlay(cache: Dict, pseudo_gt: Dict, out: Path) -> None:
    """For each class, plot pred-count and pseudo-GT-count per frame."""
    import matplotlib.pyplot as plt

    classes = list(pseudo_gt["summary"]["total_gt_per_class"].keys())
    palette = {"robot": MINT, "ball": GOLD, "field": TEAL,
                "hand": LILAC, "goal": ROSE}

    n = len(classes)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.6 * n + 1.2), sharex=True,
                              gridspec_kw={"hspace": 0.30})
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        pc = [len(fr.get("detections", {}).get(cls, [])) for fr in cache["per_frame"]]
        gc = [len(fr.get("gt", {}).get(cls, [])) for fr in pseudo_gt["per_frame"]]
        x = np.arange(len(pc))
        colour = palette.get(cls, AMBER)
        ax.fill_between(x, 0, gc, color=colour, alpha=0.30, label="pseudo-GT")
        ax.plot(x, pc, color=colour, linewidth=1.5, label="prediction")
        ax.set_ylabel(cls, color=colour)
        ax.tick_params(axis="y", colors=colour)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    axes[-1].set_xlabel("frame")
    axes[0].set_title("Detection count overlay · prediction vs pseudo-GT",
                      loc="left", pad=12)
    watermark(fig, note="count overlay")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache",     type=Path, required=True)
    ap.add_argument("--pseudo-gt", type=Path, required=True)
    ap.add_argument("--out-dir",   type=Path, default=None)
    args = ap.parse_args()

    for p in (args.cache, args.pseudo_gt):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr); return 1

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    pgt   = json.loads(args.pseudo_gt.read_text(encoding="utf-8"))

    out_dir = args.out_dir or (_HERE.parents[0] / "paper" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    res = evaluate(cache, pgt)
    metrics_path = out_dir.parent / "detection_metrics.json"
    metrics_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[det] mAP@0.5 = {res['mAP@0.5']}")
    for cls, v in res["per_class"].items():
        m50 = v["iou"]["@0.50"]
        print(f"  {cls:8s} P={m50['precision']:.3f}  R={m50['recall']:.3f}  "
              f"F1={m50['f1']:.3f}  AP={v['ap@0.5']:.3f}  ({m50['n_gt']} GT)")

    plot_pr_curves(res, out_dir / "detection_pr_curves.png")
    plot_per_class(res, out_dir / "detection_per_class.png")
    plot_count_overlay(cache, pgt, out_dir / "detection_count_overlay.png")
    print(f"[det] figures → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
