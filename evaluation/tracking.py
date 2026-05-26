"""tracking — MOTA / MOTP / ID-switch via greedy IoU tracking, no GT.

A minimal Kalman-free tracker that uses ONLY the cached SAM 3 detections
(no extra inference).  IDs are propagated frame-to-frame by greedy IoU
matching with an age-out grace; pseudo-GT trajectories are derived by
running the SAME tracker over the temporal-consistency-filtered
pseudo-GT.  Standard MOT metrics then come out cleanly without ByteTrack
as a hard dependency.

Output:
  paper/figures/tracking_metrics.png   — per-class MOTA / MOTP / IDs-switch
  paper/tracking_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
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
from experiments.futbotmx.evaluation.pseudo_gt import iou


# ── tiny IoU tracker ────────────────────────────────────────────
@dataclass
class _Track:
    id:        int
    bbox:      Tuple[float, float, float, float]
    score:     float
    last_seen: int
    history:   List[Tuple[int, Tuple[float, float, float, float]]] = field(default_factory=list)


def greedy_iou_track(per_frame: List[Dict], cls: str, *,
                     iou_thresh: float = 0.30,
                     max_age: int = 5) -> Dict[int, List[Dict]]:
    """Track instances of `cls` across frames.

    Returns: {frame_idx: [{"id": int, "bbox": [...], "score": float}, ...]}.
    """
    tracks: List[_Track] = []
    next_id = 0
    frame_assignment: Dict[int, List[Dict]] = {}

    for fi, fr in enumerate(per_frame):
        dets = fr.get("detections", fr.get("gt", {})).get(cls, [])
        # cull stale tracks
        alive = [t for t in tracks if fi - t.last_seen <= max_age]
        # greedy match
        used = [False] * len(dets)
        assigned: List[Dict] = []
        # sort by track score (proxy: last score)
        for t in sorted(alive, key=lambda tt: -tt.score):
            best_j, best_iou = -1, 0.0
            for j, d in enumerate(dets):
                if used[j]:
                    continue
                bb = d.get("bbox")
                if not bb:
                    continue
                v = iou(t.bbox, tuple(bb))
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_thresh:
                d = dets[best_j]
                used[best_j] = True
                t.bbox  = tuple(d["bbox"])
                t.score = float(d.get("score", t.score))
                t.last_seen = fi
                t.history.append((fi, t.bbox))
                assigned.append({"id": t.id, "bbox": list(t.bbox),
                                  "score": t.score})
        # spawn new tracks for unmatched detections
        for j, d in enumerate(dets):
            if used[j]:
                continue
            bb = d.get("bbox")
            if not bb:
                continue
            new = _Track(id=next_id, bbox=tuple(bb),
                          score=float(d.get("score", 0.0)),
                          last_seen=fi, history=[(fi, tuple(bb))])
            next_id += 1
            tracks.append(new)
            assigned.append({"id": new.id, "bbox": list(new.bbox),
                              "score": new.score})
        frame_assignment[fi] = assigned

    return frame_assignment


# ── MOTA / MOTP computation ─────────────────────────────────────
def _compute_mot(pred_tracks: Dict[int, List[Dict]],
                 gt_tracks: Dict[int, List[Dict]],
                 iou_thresh: float = 0.30) -> Dict:
    """MOTA = 1 - (FN + FP + IDSW) / total_gt; MOTP = mean IoU of matches."""
    n_frames = max(max(pred_tracks.keys(), default=-1),
                   max(gt_tracks.keys(),   default=-1)) + 1
    fp = fn = idsw = 0
    ious: List[float] = []
    total_gt = 0
    last_match: Dict[int, int] = {}  # gt_id -> last matched pred_id

    for fi in range(n_frames):
        preds = pred_tracks.get(fi, [])
        gts   = gt_tracks.get(fi, [])
        total_gt += len(gts)
        if not gts:
            fp += len(preds); continue
        if not preds:
            fn += len(gts); continue

        # greedy 1-1 match by IoU descending
        pairs: List[Tuple[float, int, int]] = []
        for ip, p in enumerate(preds):
            for ig, g in enumerate(gts):
                v = iou(tuple(p["bbox"]), tuple(g["bbox"]))
                if v >= iou_thresh:
                    pairs.append((v, ip, ig))
        pairs.sort(reverse=True)
        used_p = set(); used_g = set()
        for v, ip, ig in pairs:
            if ip in used_p or ig in used_g:
                continue
            used_p.add(ip); used_g.add(ig)
            ious.append(v)
            gt_id   = gts[ig]["id"]
            pred_id = preds[ip]["id"]
            if gt_id in last_match and last_match[gt_id] != pred_id:
                idsw += 1
            last_match[gt_id] = pred_id

        fp += sum(1 for ip in range(len(preds)) if ip not in used_p)
        fn += sum(1 for ig in range(len(gts)) if ig not in used_g)

    motp = float(np.mean(ious)) if ious else 0.0
    mota = 1.0 - (fp + fn + idsw) / max(total_gt, 1e-9)
    return {
        "total_gt":     int(total_gt),
        "fp":           int(fp),
        "fn":           int(fn),
        "id_switches":  int(idsw),
        "mota":         round(float(mota), 4),
        "motp":         round(float(motp), 4),
        "n_matches":    len(ious),
    }


def evaluate(cache: Dict, pseudo_gt: Dict, *,
             iou_thresh: float = 0.30, max_age: int = 5) -> Dict:
    pred_pf = cache["per_frame"]
    gt_pf   = pseudo_gt["per_frame"]
    n = min(len(pred_pf), len(gt_pf))
    classes = sorted(set(pseudo_gt["summary"]["total_gt_per_class"].keys()))

    per_class: Dict[str, Dict] = {}
    for cls in classes:
        pred_tracks = greedy_iou_track(pred_pf[:n], cls,
                                        iou_thresh=iou_thresh, max_age=max_age)
        gt_tracks   = greedy_iou_track(gt_pf[:n], cls,
                                        iou_thresh=iou_thresh, max_age=max_age)
        per_class[cls] = _compute_mot(pred_tracks, gt_tracks,
                                       iou_thresh=iou_thresh)

    # overall
    tot = {"total_gt": 0, "fp": 0, "fn": 0, "id_switches": 0, "n_matches": 0}
    iou_acc: List[float] = []
    for v in per_class.values():
        tot["total_gt"]    += v["total_gt"]
        tot["fp"]          += v["fp"]
        tot["fn"]          += v["fn"]
        tot["id_switches"] += v["id_switches"]
        tot["n_matches"]   += v["n_matches"]
        iou_acc.append(v["motp"] * v["n_matches"])
    mota = 1.0 - (tot["fp"] + tot["fn"] + tot["id_switches"]) / max(tot["total_gt"], 1e-9)
    motp = float(sum(iou_acc) / max(tot["n_matches"], 1e-9))
    overall = {**tot, "mota": round(mota, 4), "motp": round(motp, 4)}

    return {
        "schema":       "futbotmx.tracking.v1",
        "iou_thresh":   iou_thresh,
        "max_age":      max_age,
        "overall":      overall,
        "per_class":    per_class,
    }


def plot_metrics(report: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    classes = list(report["per_class"].keys())
    palette = {"robot": MINT, "ball": GOLD, "field": TEAL,
                "hand": LILAC, "goal": ROSE}
    cols = [palette.get(c, AMBER) for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [("mota", "MOTA"),
               ("motp", "MOTP (mean IoU of matches)"),
               ("id_switches", "ID switches (lower is better)")]
    for ax, (key, title) in zip(axes, metrics):
        vals = [report["per_class"][c][key] for c in classes]
        bars = ax.bar(classes, vals, color=cols, edgecolor=LINE,
                       linewidth=0.6, alpha=0.88)
        ymax = max(vals + [0.05]) * 1.20 if max(vals) > 0 else 1
        for b, v in zip(bars, vals):
            fmt = f"{v:.3f}" if key != "id_switches" else f"{int(v)}"
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, fmt,
                    ha="center", va="bottom", color=TXT, fontsize=10.5,
                    weight="bold")
        if key in ("mota", "motp"):
            ax.set_ylim(0, 1.10)
        else:
            ax.set_ylim(0, max(vals + [1]) * 1.30)
        ax.set_title(title, loc="left", pad=10)
    fig.suptitle(f"Greedy-IoU tracking · MOTA={report['overall']['mota']:.3f}  "
                  f"MOTP={report['overall']['motp']:.3f}  "
                  f"IDSW={report['overall']['id_switches']}",
                  color=TXT, fontsize=13, weight="bold", y=1.02,
                  ha="left", x=0.04)
    watermark(fig, note="tracking · greedy IoU")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache",     type=Path, required=True)
    ap.add_argument("--pseudo-gt", type=Path, required=True)
    ap.add_argument("--iou-thresh", type=float, default=0.30)
    ap.add_argument("--max-age",   type=int,  default=5)
    args = ap.parse_args()

    for p in (args.cache, args.pseudo_gt):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr); return 1

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    pgt   = json.loads(args.pseudo_gt.read_text(encoding="utf-8"))
    res = evaluate(cache, pgt, iou_thresh=args.iou_thresh, max_age=args.max_age)

    out_dir = _HERE.parents[0] / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir.parent / "tracking_metrics.json"
    metrics_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    apply_paper_style()
    plot_metrics(res, out_dir / "tracking_metrics.png")

    o = res["overall"]
    print(f"[trk] overall  MOTA={o['mota']}  MOTP={o['motp']}  IDSW={o['id_switches']}")
    for cls, v in res["per_class"].items():
        print(f"  {cls:8s} MOTA={v['mota']:.3f}  MOTP={v['motp']:.3f}  "
              f"FP={v['fp']}  FN={v['fn']}  IDSW={v['id_switches']}")
    print(f"[trk] figure → {out_dir / 'tracking_metrics.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
