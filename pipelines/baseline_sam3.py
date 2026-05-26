"""baseline_sam3 — pure SAM 3 reference pipeline (no cognitive layer).

This is the BASELINE arm of the scientific comparison study.  It uses
SAM 3 alone — concept-prompted segmentation per frame, no integration,
no calibration, no abstention, no self-representation.  Whatever SAM 3
outputs is what the baseline emits.

What it does per frame:
  1.  Run SAM 3 with the FutBotMX prompt bank (field, robots, ball,
      goal, hand) → list of Detection.
  2.  Render an annotated frame (masks + bboxes + score labels).
  3.  Record raw metrics: per-class detection counts, mean score,
      latency.

Outputs:
  output/<video_stem>_baseline.mp4               — annotated video
  output/<video_stem>_baseline.metrics.json      — per-frame metrics
  output/<video_stem>_baseline.manifest.json     — Evidence Manifest

This module is the FLOOR that the U-CogNet arm (pipelines/ucognet_sam3.py)
must beat on every interesting metric (calibration, abstention,
robustness, OOD detection, surprise localisation, …) for the comparison
to mean anything.  Same SAM 3 weights, same prompts, same frames — only
the cognitive layer differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
_FUTBOTMX  = _REPO_ROOT / "experiments" / "futbotmx"
DEFAULT_OUT = _FUTBOTMX / "output"


# ── Prompt bank: open-vocabulary text prompts for robot soccer ──
DEFAULT_PROMPTS: Dict[str, str] = {
    "field":    "green soccer field, green carpet playing area",
    "robot":    "small wheeled soccer robot",
    "ball":     "small ball on the playing field",
    "goal":     "soccer goal, goalpost on the side of the field",
    "hand":     "human hand at the edge of the table",
}

# Stable BGR colours per class for the annotated video
CLASS_COLOURS_BGR: Dict[str, Tuple[int,int,int]] = {
    "field":  (110, 200, 110),
    "robot":  ( 60, 240, 100),
    "ball":   (  0, 165, 255),
    "goal":   (255, 130,  20),
    "hand":   (180, 180, 200),
}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── Per-frame record ────────────────────────────────────────────
@dataclass
class FrameMetric:
    frame_idx: int
    timestamp_s: float
    per_class_counts:   Dict[str, int]      # 'robot': 4, 'ball': 1, ...
    per_class_top_score: Dict[str, float]   # top-scoring instance per class
    per_class_mean_score: Dict[str, float]
    infer_ms: float
    render_ms: float


# ── Evidence Manifest ───────────────────────────────────────────
def _sha256_short(path: Path, nbytes: int = 1_000_000) -> str:
    """Cheap content hash — first MB only, enough for run identity."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def build_evidence_manifest(*, video: Path, output_video: Path,
                             metrics_path: Path, prompts: Dict[str, str],
                             args_dict: Dict, total_frames: int,
                             total_seconds: float) -> Dict:
    return {
        "schema":           "futbotmx.baseline.v1",
        "arm":              "baseline_sam3",
        "generated_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform":         platform.platform(),
        "python":           platform.python_version(),
        "video": {
            "path":         str(video),
            "sha256_first1mb": _sha256_short(video) if video.exists() else "",
            "n_frames":     total_frames,
            "duration_s":   round(total_seconds, 3),
        },
        "model": {
            "name":         "facebook/sam3",
            "revision":     "main",
            "hf_home":      os.environ.get("HF_HOME", ""),
            "device":       args_dict.get("device", "cuda"),
            "autocast":     "bfloat16",
        },
        "prompts":          prompts,
        "args":             args_dict,
        "outputs": {
            "video":        str(output_video),
            "metrics_json": str(metrics_path),
        },
        "reproduce": {
            "venv":         "F:/U-CogNet-ToGo/futbotmx_venv",
            "entry":        "experiments.futbotmx.pipelines.baseline_sam3",
        },
    }


# ── core pipeline ───────────────────────────────────────────────
class BaselineSam3Pipeline:
    def __init__(self, *, device: str = "cuda",
                 prompts: Optional[Dict[str, str]] = None,
                 score_min: float = 0.30, max_side: int = 960):
        self.device = device
        self.prompts = dict(prompts or DEFAULT_PROMPTS)
        self.score_min = score_min
        self.max_side = max_side
        self._model = None
        self._processor = None
        self._torch = None
        self._loaded_at: Optional[float] = None

    # ── model loading ───────────────────────────────────────────
    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        print(f"[baseline] loading SAM 3 on {self.device} …", file=sys.stderr)
        t0 = time.time()
        self._torch = torch
        self._model = build_sam3_image_model(device=self.device, load_from_HF=True)
        self._processor = Sam3Processor(self._model)
        self._loaded_at = time.time()
        print(f"[baseline] model ready in {self._loaded_at - t0:.1f}s",
              file=sys.stderr)

    # ── one-frame inference ─────────────────────────────────────
    def detect_frame(self, frame_bgr: np.ndarray
                     ) -> Tuple[Dict[str, List[Dict]], float]:
        """Return {class_name: [{mask, bbox, score}, …]}, infer_ms.

        Mask is a bool numpy array; bbox is [x1,y1,x2,y2] in pixels of
        the resized frame; score is a Python float.
        """
        import cv2
        from PIL import Image
        torch = self._torch
        self._ensure_model()
        torch = self._torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        per_class: Dict[str, List[Dict]] = {c: [] for c in self.prompts}
        t0 = time.time()
        with torch.inference_mode(), \
             torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            state = self._processor.set_image(pil)
            for class_name, text in self.prompts.items():
                out = self._processor.set_text_prompt(state=state, prompt=text)
                masks  = out.get("masks")
                boxes  = out.get("boxes")
                scores = out.get("scores")
                if masks is None or len(masks) == 0:
                    continue
                scores_np = scores.detach().float().cpu().numpy()
                order = np.argsort(-scores_np)
                for i in order:
                    s = float(scores_np[i])
                    if s < self.score_min:
                        break
                    m = masks[i].detach().float().cpu().numpy()
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m[0]
                    m = (np.squeeze(m) > 0.5)
                    b = boxes[i].detach().float().cpu().tolist() \
                        if boxes is not None else None
                    per_class[class_name].append({"mask": m, "bbox": b, "score": s})
        infer_ms = (time.time() - t0) * 1000
        return per_class, infer_ms

    # ── CLEAN frame rendering (aesthetic mode) ──────────────────
    def render_clean(self, frame_bgr: np.ndarray,
                     per_class: Dict[str, List[Dict]]) -> np.ndarray:
        """Pristine annotation — only soft masks and thin bboxes.  NO
        text/HUD on the video.  All telemetry goes to the sidebar."""
        import cv2
        out = frame_bgr.copy()
        # masks first (soft 0.32 alpha so the pitch stays readable)
        for class_name, dets in per_class.items():
            colour = CLASS_COLOURS_BGR.get(class_name, (200, 200, 200))
            for d in dets:
                mask = d["mask"]
                if mask.shape != out.shape[:2]:
                    continue
                colour_layer = np.zeros_like(out)
                colour_layer[mask] = colour
                out = np.where(mask[..., None],
                               (0.68 * out + 0.32 * colour_layer).astype(np.uint8),
                               out)
        # thin bboxes (no labels)
        for class_name, dets in per_class.items():
            colour = CLASS_COLOURS_BGR.get(class_name, (200, 200, 200))
            for d in dets:
                if d["bbox"] is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
        return out

    # ── annotated frame rendering (HUD mode — legacy) ───────────
    def render_frame(self, frame_bgr: np.ndarray,
                     per_class: Dict[str, List[Dict]],
                     hud_lines: Optional[Sequence[str]] = None) -> np.ndarray:
        import cv2
        out = frame_bgr.copy()
        # masks first, then bboxes on top
        for class_name, dets in per_class.items():
            colour = CLASS_COLOURS_BGR.get(class_name, (200, 200, 200))
            for d in dets:
                mask = d["mask"]
                if mask.shape != out.shape[:2]:
                    continue
                colour_layer = np.zeros_like(out)
                colour_layer[mask] = colour
                out = np.where(mask[..., None],
                               (0.55 * out + 0.45 * colour_layer).astype(np.uint8),
                               out)
        for class_name, dets in per_class.items():
            colour = CLASS_COLOURS_BGR.get(class_name, (200, 200, 200))
            for d in dets:
                if d["bbox"] is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
                tag = f"{class_name} {d['score']:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x1, max(0, y1 - th - 6)),
                              (x1 + tw + 6, y1), colour, -1)
                cv2.putText(out, tag, (x1 + 3, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

        # HUD watermark — bottom-left, identifies the arm
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, h - 28), (220, h), (0, 0, 0), -1)
        cv2.putText(out, "BASELINE | SAM 3 only",
                    (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (110, 220, 255), 1, cv2.LINE_AA)

        if hud_lines:
            self._draw_hud(out, hud_lines)
        return out

    def _draw_hud(self, img: np.ndarray, lines: Sequence[str]) -> None:
        import cv2
        pad, lh = 8, 20
        box_h = pad * 2 + lh * len(lines)
        box_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                    for l in lines) + 2 * pad
        ov = img.copy()
        cv2.rectangle(ov, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
        for i, ln in enumerate(lines):
            cv2.putText(img, ln,
                        (10 + pad, 10 + pad + (i + 1) * lh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 240, 255), 1,
                        cv2.LINE_AA)

    # ── run on a whole video ────────────────────────────────────
    def run_video(self, video_path: Path, out_dir: Path,
                  max_frames: Optional[int] = None,
                  output_fps: Optional[float] = None,
                  aesthetic: bool = False,
                  panel_w: int = 1280, panel_h: int = 720) -> Dict:
        import cv2
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = video_path.stem
        suffix = "baseline_aesthetic" if aesthetic else "baseline"
        out_video_path   = out_dir / f"{stem}_{suffix}.mp4"
        out_metrics_path = out_dir / f"{stem}_{suffix}.metrics.json"
        out_manifest_path = out_dir / f"{stem}_{suffix}.manifest.json"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 cannot open {video_path}")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nf      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        scale   = min(1.0, self.max_side / max(W, H))
        vid_W, vid_H = int(W * scale), int(H * scale)
        fps_out = output_fps or fps_src

        # Aesthetic mode composes a (panel_w, panel_h) canvas; HUD mode
        # writes the raw vid_W x vid_H frame
        if aesthetic:
            out_W, out_H = panel_w, panel_h
            from experiments.futbotmx.viz.aesthetic import (
                AestheticCompositor, SidebarState,
            )
            self._aesthetic = AestheticCompositor(panel_w=panel_w,
                                                    panel_h=panel_h)
        else:
            out_W, out_H = vid_W, vid_H

        print(f"[baseline] video: {video_path.name}  {W}x{H} @ {fps_src:.1f}fps  "
              f"({nf} frames) → vid {vid_W}x{vid_H}{' · panel ' + str(panel_w) + 'x' + str(panel_h) if aesthetic else ''}",
              file=sys.stderr)

        # mp4v is universally readable + bundled with cv2 on Windows
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc,
                                  fps_out, (out_W, out_H))
        if not writer.isOpened():
            raise RuntimeError(f"cv2 VideoWriter failed: {out_video_path}")

        self._ensure_model()
        per_frame: List[Dict] = []
        n_written = 0
        t_start = time.time()
        try:
            for idx in range(nf):
                ok, frame = cap.read()
                if not ok:
                    break
                if scale < 1.0:
                    frame = cv2.resize(frame, (vid_W, vid_H),
                                        interpolation=cv2.INTER_AREA)

                per_class, infer_ms = self.detect_frame(frame)
                t_render = time.time()
                if aesthetic:
                    clean = self.render_clean(frame, per_class)
                    state = SidebarState(
                        arm_label="BASELINE   |   SAM 3 only",
                        arm_colour=(115, 200, 250),       # amber
                        frame_idx=idx, total_frames=nf,
                        infer_ms=infer_ms, cognitive_ms=0.0,
                        per_class={c: {"count": len(d),
                                        "top": max((dd["score"] for dd in d), default=0.0)}
                                    for c, d in per_class.items() if d},
                    )
                    vis = self._aesthetic.render(clean, state)
                else:
                    hud = [
                        f"frame {idx+1}/{nf}  ({(idx+1)/max(nf,1)*100:.0f}%)",
                        f"infer {infer_ms:.0f} ms",
                    ]
                    for c, dets in per_class.items():
                        if dets:
                            top = max(d["score"] for d in dets)
                            hud.append(f"  {c}: {len(dets)} (top {top:.2f})")
                    vis = self.render_frame(frame, per_class, hud_lines=hud)
                render_ms = (time.time() - t_render) * 1000
                writer.write(vis)
                n_written += 1

                counts = {c: len(d) for c, d in per_class.items()}
                tops   = {c: max((dd["score"] for dd in d), default=0.0)
                          for c, d in per_class.items()}
                means  = {c: (float(np.mean([dd["score"] for dd in d])) if d else 0.0)
                          for c, d in per_class.items()}
                per_frame.append(asdict(FrameMetric(
                    frame_idx=idx, timestamp_s=round(idx / fps_src, 4),
                    per_class_counts=counts, per_class_top_score=tops,
                    per_class_mean_score=means,
                    infer_ms=round(infer_ms, 2),
                    render_ms=round(render_ms, 2),
                )))

                if max_frames and n_written >= max_frames:
                    break
        finally:
            cap.release()
            writer.release()

        total_s = time.time() - t_start
        agg = _aggregate_metrics(per_frame)
        agg["video_path"]      = str(video_path)
        agg["output_video"]    = str(out_video_path)
        agg["frames_written"]  = n_written
        agg["wall_seconds"]    = round(total_s, 2)
        agg["wall_fps"]        = round(n_written / max(total_s, 1e-3), 2)
        out_metrics_path.write_text(json.dumps(
            {"summary": agg, "per_frame": per_frame},
            indent=2, ensure_ascii=False
        ), encoding="utf-8")

        manifest = build_evidence_manifest(
            video=video_path, output_video=out_video_path,
            metrics_path=out_metrics_path, prompts=self.prompts,
            args_dict={"device": self.device, "score_min": self.score_min,
                       "max_side": self.max_side,
                       "max_frames": max_frames},
            total_frames=n_written, total_seconds=total_s,
        )
        out_manifest_path.write_text(json.dumps(manifest, indent=2),
                                      encoding="utf-8")

        print(f"\n[baseline] ✅ written {n_written} frames in {total_s:.1f}s "
              f"({n_written/max(total_s,1e-3):.2f} fps)", file=sys.stderr)
        print(f"[baseline]   video    : {out_video_path}", file=sys.stderr)
        print(f"[baseline]   metrics  : {out_metrics_path}", file=sys.stderr)
        print(f"[baseline]   manifest : {out_manifest_path}", file=sys.stderr)
        return agg


def _aggregate_metrics(per_frame: List[Dict]) -> Dict:
    if not per_frame:
        return {}
    classes = sorted({c for r in per_frame for c in r["per_class_counts"]})
    out: Dict[str, Dict] = {"per_class": {}}
    for c in classes:
        counts = [r["per_class_counts"].get(c, 0) for r in per_frame]
        tops   = [r["per_class_top_score"].get(c, 0.0) for r in per_frame]
        means  = [r["per_class_mean_score"].get(c, 0.0) for r in per_frame]
        nonzero_tops = [t for t in tops if t > 0]
        out["per_class"][c] = {
            "n_frames_with_detection": int(sum(1 for c_ in counts if c_ > 0)),
            "total_detections":         int(sum(counts)),
            "mean_count_per_frame":     round(float(np.mean(counts)), 2),
            "max_count_in_a_frame":     int(max(counts)),
            "mean_top_score":           round(float(np.mean(nonzero_tops)), 3) if nonzero_tops else 0.0,
            "mean_score_overall":       round(float(np.mean(means)), 3),
        }
    infer = [r["infer_ms"] for r in per_frame]
    out["latency_ms"] = {
        "infer_mean":  round(float(np.mean(infer)), 2),
        "infer_p50":   round(float(np.percentile(infer, 50)), 2),
        "infer_p95":   round(float(np.percentile(infer, 95)), 2),
    }
    return out


# ── CLI ─────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline SAM 3 pipeline (no cognitive layer)")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--score-min", type=float, default=0.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aesthetic", action="store_true",
                    help="Paper-grade composition (clean video + sidebar)")
    ap.add_argument("--panel-w", type=int, default=1280)
    ap.add_argument("--panel-h", type=int, default=720)
    ap.add_argument("--open", action="store_true",
                    help="Open the annotated MP4 when done")
    args = ap.parse_args()

    _load_dotenv(_REPO_ROOT / ".env")
    os.environ.setdefault("HF_HOME", "F:/U-CogNet-ToGo/sam3")

    pipe = BaselineSam3Pipeline(
        device=args.device, score_min=args.score_min, max_side=args.max_side,
    )
    summary = pipe.run_video(args.video, args.out_dir, max_frames=args.max_frames,
                              aesthetic=args.aesthetic,
                              panel_w=args.panel_w, panel_h=args.panel_h)

    print("\n[baseline] summary →")
    print(json.dumps(summary, indent=2))

    suffix = "baseline_aesthetic" if args.aesthetic else "baseline"
    out_video = args.out_dir / f"{args.video.stem}_{suffix}.mp4"
    if args.open and out_video.exists():
        if sys.platform == "win32":
            os.startfile(out_video)
        else:
            subprocess.run(["xdg-open", str(out_video)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
