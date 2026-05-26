"""cache_detections — run SAM 3 once on a video, persist every detection.

Every downstream evaluator (`pseudo_gt`, `detection`, `tracking`,
`calibration`, `inter_arm`, `robustness`) reads from the SAME cache so
we never pay the SAM-3 inference cost twice.  Cache format is a single
JSON file with one entry per frame containing all per-class instances:

    {
      "video":      "F:/.../IMG_9914.MOV",
      "video_sha":  "...",                  # first-MB hash for identity
      "n_frames":   133,
      "fps":        30.0,
      "frame_size": [720, 405],             # after --max-side resize
      "prompts":    {"robot": "...", "ball": "...", ...},
      "score_min":  0.30,
      "device":     "cuda",
      "per_frame": [
        {
          "frame_idx": 0,
          "timestamp_s": 0.000,
          "infer_ms": 940.5,
          "detections": {
            "robot": [{"bbox": [x1,y1,x2,y2], "score": 0.94}, ...],
            "ball":  [...], ...
          }
        }, ...
      ]
    }

Mask pixels are NOT cached (they're large and recomputable from masks
on demand).  Bboxes + scores are what every evaluator actually consumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
_FUTBOTMX  = _REPO_ROOT / "experiments" / "futbotmx"

# Make pipeline imports work
sys.path.insert(0, str(_REPO_ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _sha256_short(path: Path, nbytes: int = 1_000_000) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video",  type=Path, required=True)
    ap.add_argument("--out",    type=Path, default=None,
                    help="cache JSON path (default: output/<stem>.cache.json)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-side",   type=int, default=720)
    ap.add_argument("--score-min",  type=float, default=0.30)
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    _load_dotenv(_REPO_ROOT / ".env")
    os.environ.setdefault("HF_HOME", "F:/U-CogNet-ToGo/sam3")

    if not args.video.exists():
        print(f"ERROR: video not found {args.video}", file=sys.stderr)
        return 1

    out = args.out or (_FUTBOTMX / "output" / f"{args.video.stem}.cache.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    from experiments.futbotmx.pipelines.baseline_sam3 import (
        BaselineSam3Pipeline, DEFAULT_PROMPTS,
    )

    pipe = BaselineSam3Pipeline(
        device=args.device, score_min=args.score_min, max_side=args.max_side,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cv2 cannot open {args.video}", file=sys.stderr); return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    scale = min(1.0, args.max_side / max(W, H))
    vw, vh = int(W * scale), int(H * scale)
    print(f"[cache] {args.video.name}  {W}x{H} @ {fps:.1f}fps  "
          f"({nf} frames) → {vw}x{vh}", file=sys.stderr)

    per_frame: List[Dict] = []
    t0 = time.time()
    pipe._ensure_model()
    target = min(nf, args.max_frames) if args.max_frames else nf
    try:
        for idx in range(target):
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (vw, vh),
                                    interpolation=cv2.INTER_AREA)
            per_class, infer_ms = pipe.detect_frame(frame)
            dets_serialised = {
                cls: [
                    {"bbox": [float(x) for x in d["bbox"]] if d["bbox"] else None,
                     "score": float(d["score"])}
                    for d in detections
                ]
                for cls, detections in per_class.items()
                if detections
            }
            per_frame.append({
                "frame_idx":   idx,
                "timestamp_s": round(idx / fps, 4),
                "infer_ms":    round(float(infer_ms), 2),
                "detections":  dets_serialised,
            })
            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                fps_w   = (idx + 1) / elapsed
                eta     = (target - idx - 1) / fps_w
                print(f"[cache]   {idx + 1}/{target}  ({fps_w:.2f} fps · ETA {eta:.0f}s)",
                      file=sys.stderr)
    finally:
        cap.release()

    elapsed = time.time() - t0
    cache = {
        "schema":     "futbotmx.cache.v1",
        "video":      str(args.video),
        "video_sha":  _sha256_short(args.video),
        "n_frames":   len(per_frame),
        "fps":        round(float(fps), 3),
        "frame_size": [vw, vh],
        "prompts":    DEFAULT_PROMPTS,
        "score_min":  args.score_min,
        "device":     args.device,
        "wall_seconds": round(elapsed, 2),
        "per_frame":  per_frame,
    }
    out.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"[cache] ✅ {len(per_frame)} frames in {elapsed:.1f}s  → {out}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
