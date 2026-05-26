"""robustness — Hendrycks-Dietterich-style perturbation battery, zero GT.

22 conditions (7 perturbation types × 3 intensities + 1 clean baseline)
on N sampled frames.  Measures per-class self-consistency vs the clean
detection set — same scene, perturbed, must give nearly the same
detections.  No ground truth.

Output:
  paper/figures/robustness_accuracy.png
  paper/robustness_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from experiments.futbotmx.evaluation.pseudo_gt import iou


PERTURBATIONS = ["gaussian_noise", "gaussian_blur", "brightness_up",
                  "brightness_down", "contrast", "jpeg", "rotation"]


def _perturb(image_bgr: np.ndarray, kind: str, level: int) -> np.ndarray:
    import cv2
    if kind == "gaussian_noise":
        sigma = (8, 18, 30)[level]
        noisy = image_bgr.astype(np.float32) + np.random.normal(0, sigma, image_bgr.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "gaussian_blur":
        k = (3, 7, 13)[level]
        return cv2.GaussianBlur(image_bgr, (k, k), 0)
    if kind == "brightness_up":
        gain = (1.15, 1.30, 1.55)[level]
        return np.clip(image_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if kind == "brightness_down":
        gain = (0.85, 0.70, 0.50)[level]
        return np.clip(image_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if kind == "contrast":
        gain = (1.20, 1.45, 1.75)[level]
        mean = image_bgr.mean()
        return np.clip((image_bgr.astype(np.float32) - mean) * gain + mean,
                        0, 255).astype(np.uint8)
    if kind == "jpeg":
        q = (60, 30, 15)[level]
        ok, enc = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else image_bgr.copy()
    if kind == "rotation":
        deg = (3, 7, 12)[level]
        h, w = image_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(image_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(f"unknown perturbation: {kind}")


def _consistency(clean: Dict[str, List[Dict]],
                 pert: Dict[str, List[Dict]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls in set(list(clean.keys()) + list(pert.keys())):
        c = clean.get(cls, []); p = pert.get(cls, [])
        if not c and not p:
            out[cls] = 1.0; continue
        if not c or not p:
            out[cls] = 0.0; continue
        used = [False] * len(p)
        matched_ious: List[float] = []
        for cd in c:
            cbb = cd.get("bbox")
            if not cbb: continue
            best_j, best_iou = -1, 0.0
            for j, pd in enumerate(p):
                if used[j] or not pd.get("bbox"): continue
                v = iou(tuple(cbb), tuple(pd["bbox"]))
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                used[best_j] = True; matched_ious.append(best_iou)
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        count_ratio = min(len(p), len(c)) / max(max(len(p), len(c)), 1)
        out[cls] = round(float(np.sqrt(mean_iou * count_ratio)), 4)
    return out


def _load_dotenv(path: Path) -> None:
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def evaluate(video_path: Path, *, n_sample: int = 12,
             device: str = "cuda", max_side: int = 720) -> Dict:
    import cv2
    from experiments.futbotmx.pipelines.baseline_sam3 import (
        BaselineSam3Pipeline,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    scale = min(1.0, max_side / max(W, H))
    vw, vh = int(W * scale), int(H * scale)

    sample_idxs = np.linspace(2, max(nf - 3, 2), n_sample).astype(int)
    print(f"[rob] {video_path.name} ({nf} frames) → sampling {len(sample_idxs)}",
          file=sys.stderr)

    frames: List[np.ndarray] = []
    for fi in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok: continue
        if scale < 1.0:
            fr = cv2.resize(fr, (vw, vh), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()

    pipe = BaselineSam3Pipeline(device=device, max_side=max_side)
    pipe._ensure_model()

    # 1) clean reference
    clean = []
    t0 = time.time()
    for fr in frames:
        dets, _ = pipe.detect_frame(fr)
        clean.append({cls: [{"bbox": d["bbox"], "score": d["score"]}
                            for d in dets[cls] if d.get("bbox")]
                       for cls in dets})
    print(f"[rob] clean ref in {time.time()-t0:.1f}s", file=sys.stderr)

    # 2) per-condition
    results: Dict[str, Dict] = {}
    for kind in PERTURBATIONS:
        for level in range(3):
            cond = f"{kind}/L{level + 1}"
            t1 = time.time()
            per_class_acc: Dict[str, List[float]] = {}
            for i, fr in enumerate(frames):
                pert = _perturb(fr, kind, level)
                dets_p, _ = pipe.detect_frame(pert)
                pert_ser = {cls: [{"bbox": d["bbox"], "score": d["score"]}
                                   for d in dets_p[cls] if d.get("bbox")]
                              for cls in dets_p}
                cs = _consistency(clean[i], pert_ser)
                for cls, v in cs.items():
                    per_class_acc.setdefault(cls, []).append(v)
            results[cond] = {
                "perturbation": kind, "level": level + 1,
                "per_class": {cls: round(float(np.mean(vs)), 4)
                               for cls, vs in per_class_acc.items()},
                "overall":  round(float(np.mean(
                    [v for vs in per_class_acc.values() for v in vs])), 4),
                "elapsed_s": round(time.time() - t1, 1),
            }
            print(f"[rob] {cond:22s} overall={results[cond]['overall']:.3f}  "
                  f"({results[cond]['elapsed_s']:.1f}s)", file=sys.stderr)

    overall = float(np.mean([r["overall"] for r in results.values()]))
    return {
        "schema":              "futbotmx.robustness.v1",
        "video":               str(video_path),
        "n_sample_frames":     len(frames),
        "n_conditions":        len(results),
        "overall_consistency": round(overall, 4),
        "per_condition":       results,
    }


def plot_accuracy(report: Dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    palette = {"gaussian_noise": ROSE, "gaussian_blur": TEAL,
                "brightness_up": GOLD, "brightness_down": AMBER,
                "contrast": LILAC, "jpeg": MINT, "rotation": TXT_MUTED}

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.4))
    levels = [1, 2, 3]
    for kind in PERTURBATIONS:
        ys = [report["per_condition"][f"{kind}/L{lv}"]["overall"]
              for lv in levels]
        ax.plot(levels, ys, marker="o", linewidth=1.8,
                color=palette.get(kind, AMBER), markersize=8,
                markeredgecolor=LINE, label=kind)
    ax.axhline(report["overall_consistency"], color=TXT, linestyle="--",
                linewidth=1.0, alpha=0.7,
                label=f"battery mean = {report['overall_consistency']:.3f}")
    ax.set_xlabel("perturbation intensity (1 = mild, 3 = severe)")
    ax.set_ylabel("self-consistency vs clean (IoU x count agreement)")
    ax.set_ylim(0, 1.05); ax.set_xticks(levels)
    ax.set_title("Robustness battery | 22 conditions | zero ground-truth",
                  loc="left", pad=12)
    ax.legend(loc="lower left", frameon=True, ncol=2)
    watermark(fig, note="robustness / no GT")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--n-sample", type=int, default=12)
    ap.add_argument("--max-side", type=int, default=720)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    _load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    os.environ.setdefault("HF_HOME", "F:/U-CogNet-ToGo/sam3")

    if not args.video.exists():
        print(f"ERROR: missing {args.video}", file=sys.stderr); return 1
    out_dir = _HERE.parents[0] / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate(args.video, n_sample=args.n_sample,
                      device=args.device, max_side=args.max_side)
    metrics_path = out_dir.parent / "robustness_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    apply_paper_style()
    plot_accuracy(report, out_dir / "robustness_accuracy.png")
    print(f"[rob] OK overall = {report['overall_consistency']}")
    print(f"[rob] metrics -> {metrics_path}")
    print(f"[rob] figure  -> {out_dir / 'robustness_accuracy.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
