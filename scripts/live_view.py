"""live_view — REAL-TIME SAM 3 segmentation overlay on a video.

Opens an OpenCV window that streams the video while SAM 3 segments each
frame.  You SEE what the model sees as it processes.

Keyboard controls:
  q  → quit
  s  → save current frame as PNG (annotated + original side by side)
  +  → skip more frames per inference   (faster playback, less GPU work)
  -  → skip fewer frames per inference  (slower playback, smoother)
  p  → pause / resume

Usage:
  & F:/U-CogNet-ToGo/futbotmx_venv/Scripts/python.exe `
    -m experiments.futbotmx.scripts.live_view --video F:/U-CogNet-ToGo/sam3/IMG_9914.MOV
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# Robot-soccer prompt palette — high contrast, BGR (cv2 native)
PROMPT_COLOURS = [
    ("orange ball, small round ball",       (  0, 165, 255)),   # orange
    ("robot, small wheeled robot",          ( 60, 240, 100)),   # green
    ("blue robot",                          (255, 130,  20)),   # blue (BGR)
    ("red robot",                           ( 50,  50, 240)),   # red
    ("soccer field, green carpet",          (200, 200, 200)),   # grey
]


def overlay(frame_bgr: np.ndarray, mask: np.ndarray,
            colour_bgr: Tuple[int,int,int], alpha: float = 0.45) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = np.squeeze(mask).astype(bool)
    if mask.shape != frame_bgr.shape[:2]:
        return frame_bgr
    out = frame_bgr.copy()
    colour_layer = np.zeros_like(out)
    colour_layer[mask] = colour_bgr
    out = np.where(mask[..., None],
                   ((1 - alpha) * out + alpha * colour_layer).astype(np.uint8),
                   out)
    return out


def draw_bbox(img_bgr: np.ndarray, bbox, label: str,
              colour_bgr: Tuple[int,int,int], score: float) -> np.ndarray:
    import cv2
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), colour_bgr, 2)
    tag = f"{label}  {score:.2f}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)),
                  (x1 + tw + 6, y1), colour_bgr, -1)
    cv2.putText(img_bgr, tag, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                lineType=cv2.LINE_AA)
    return img_bgr


def draw_hud(img_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    import cv2
    h, w = img_bgr.shape[:2]
    pad = 8
    line_h = 22
    box_h = pad * 2 + line_h * len(lines)
    box_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                for l in lines) + 2 * pad
    overlay_img = img_bgr.copy()
    cv2.rectangle(overlay_img, (10, 10), (10 + box_w, 10 + box_h),
                  (0, 0, 0), -1)
    img_bgr = cv2.addWeighted(overlay_img, 0.55, img_bgr, 0.45, 0)
    for i, ln in enumerate(lines):
        cv2.putText(img_bgr, ln,
                    (10 + pad, 10 + pad + (i + 1) * line_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 240, 255), 1,
                    lineType=cv2.LINE_AA)
    return img_bgr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--prompts", type=str, default=None,
                    help="Comma-separated prompts (default: robot-soccer set)")
    ap.add_argument("--max-side", type=int, default=960,
                    help="Resize so longer side ≤ this (speed)")
    ap.add_argument("--skip", type=int, default=2,
                    help="Run SAM 3 every N frames (1 = every frame)")
    ap.add_argument("--score-min", type=float, default=0.35)
    ap.add_argument("--top-k", type=int, default=4,
                    help="Show only top-K masks per prompt")
    args = ap.parse_args()

    _load_dotenv(_REPO_ROOT / ".env")
    os.environ.setdefault("HF_HOME", "F:/U-CogNet-ToGo/sam3")

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    import cv2
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"[live] loading SAM 3 …")
    t0 = time.time()
    model = build_sam3_image_model(device="cuda", load_from_HF=True)
    processor = Sam3Processor(model)
    print(f"[live] model ready in {time.time()-t0:.1f}s")

    if args.prompts:
        prompts = [(p.strip(), PROMPT_COLOURS[i % len(PROMPT_COLOURS)][1])
                   for i, p in enumerate(args.prompts.split(",")) if p.strip()]
    else:
        prompts = PROMPT_COLOURS

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        return 1
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"[live] video: {args.video.name}  {W}x{H} @ {fps_src:.1f}fps  "
          f"({nf} frames)")
    print("[live] keys: q=quit · s=snapshot · +/- skip · p=pause")

    win = "FutBotMX · SAM 3 live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(W, 1280), min(H, 720))

    out_dir = _REPO_ROOT / "experiments" / "futbotmx" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    skip = max(1, int(args.skip))
    paused = False
    frame_idx = 0
    smoothed_fps = 0.0
    last_annotated = None
    t_loop = time.time()

    try:
        while True:
            if not paused:
                # advance `skip-1` frames cheaply, then read one
                for _ in range(skip - 1):
                    cap.grab()
                ok, frame = cap.read()
                if not ok:
                    print("[live] end of video")
                    break
                frame_idx += skip

                # resize for speed
                h, w = frame.shape[:2]
                scale = args.max_side / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                t_infer = time.time()
                annotated = frame.copy()
                per_prompt_counts = []
                with torch.inference_mode(), \
                     torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = processor.set_image(pil)
                    for (prompt, colour) in prompts:
                        out = processor.set_text_prompt(
                            state=state, prompt=prompt)
                        masks  = out.get("masks")
                        boxes  = out.get("boxes")
                        scores = out.get("scores")
                        if masks is None or len(masks) == 0:
                            per_prompt_counts.append((prompt, 0))
                            continue
                        scores_np = scores.detach().cpu().numpy()
                        order = np.argsort(-scores_np)[:args.top_k]
                        kept = 0
                        for i in order:
                            s = float(scores_np[i])
                            if s < args.score_min:
                                continue
                            m = masks[i].detach().cpu().numpy()
                            b = boxes[i].detach().cpu().tolist() \
                                if boxes is not None else None
                            annotated = overlay(annotated, m, colour)
                            if b is not None:
                                annotated = draw_bbox(annotated, b, prompt[:18],
                                                       colour, s)
                            kept += 1
                        per_prompt_counts.append((prompt, kept))
                infer_ms = (time.time() - t_infer) * 1000

                dt = time.time() - t_loop
                t_loop = time.time()
                inst_fps = 1.0 / max(dt, 1e-6)
                smoothed_fps = 0.85 * smoothed_fps + 0.15 * inst_fps if smoothed_fps else inst_fps

                hud = [
                    f"frame {frame_idx}/{nf}  ({frame_idx/max(nf,1)*100:.0f}%)",
                    f"infer {infer_ms:.0f} ms   display {smoothed_fps:.1f} fps   skip x{skip}",
                ]
                for (prompt, k) in per_prompt_counts:
                    hud.append(f"  · '{prompt[:30]}' → {k}")
                annotated = draw_hud(annotated, hud)
                last_annotated = annotated

            if last_annotated is not None:
                cv2.imshow(win, last_annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and last_annotated is not None:
                snap = out_dir / f"live_{args.video.stem}_f{frame_idx:06d}.png"
                cv2.imwrite(str(snap), last_annotated)
                print(f"[live] snapshot → {snap}")
            elif key == ord("+") or key == ord("="):
                skip = min(skip + 1, 30)
                print(f"[live] skip = {skip}")
            elif key == ord("-"):
                skip = max(1, skip - 1)
                print(f"[live] skip = {skip}")
            elif key == ord("p"):
                paused = not paused
                print(f"[live] {'paused' if paused else 'resumed'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
