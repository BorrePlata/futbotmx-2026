"""visualize_inference — RUN SAM 3 + SHOW what it segments.

Renders a side-by-side grid (original | mask overlay | bbox label) per
prompt and saves to PNG.  Auto-opens the file when done.

Usage:
  & F:/U-CogNet-ToGo/futbotmx_venv/Scripts/python.exe `
    -m experiments.futbotmx.scripts.visualize_inference                    # default demo
  & F:/U-CogNet-ToGo/futbotmx_venv/Scripts/python.exe `
    -m experiments.futbotmx.scripts.visualize_inference --image path.jpg --prompts "ball,robot,field"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = _REPO_ROOT / "experiments" / "futbotmx" / "output"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# Distinct, high-contrast colours for up to 8 prompts (RGB)
PALETTE = [
    (255,  90,  90),   # red
    ( 90, 170, 255),   # blue
    (255, 200,  60),   # yellow
    ( 90, 240, 170),   # mint
    (200, 130, 255),   # purple
    (255, 145,  60),   # orange
    (130, 240, 240),   # cyan
    (255, 110, 200),   # pink
]


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 colour, alpha: float = 0.50) -> np.ndarray:
    out = image_rgb.copy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = mask.astype(bool)
    if mask.shape != image_rgb.shape[:2]:
        # squeeze any extra dim
        mask = np.squeeze(mask)
    overlay = out.copy()
    overlay[mask] = colour
    return ((1 - alpha) * out + alpha * overlay).astype(np.uint8)


def draw_bbox(image_rgb: np.ndarray, bbox, label: str, colour,
              score: float = 0.0) -> np.ndarray:
    import cv2
    img = image_rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), colour[::-1], 3)  # BGR for cv2
    tag = f"{label}  {score:.2f}" if score else label
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), colour[::-1], -1)
    cv2.putText(img, tag, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                lineType=cv2.LINE_AA)
    return img


def open_file(path: Path) -> None:
    """Open the rendered PNG in the OS default viewer."""
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", str(path)])
        else:
            import subprocess
            subprocess.run(["xdg-open", str(path)])
    except Exception as e:
        print(f"[viz] could not auto-open: {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", type=Path, default=None,
                    help="Path to an image; defaults to SAM 3's truck.jpg sample")
    ap.add_argument("--prompts", type=str,
                    default="truck,wheel,window,headlight",
                    help="Comma-separated text prompts")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-open", action="store_true",
                    help="Skip auto-opening the result PNG")
    args = ap.parse_args()

    _load_dotenv(_REPO_ROOT / ".env")
    os.environ.setdefault("HF_HOME", "F:/U-CogNet-ToGo/sam3")

    import torch
    from PIL import Image
    import matplotlib.pyplot as plt

    image_path = args.image or Path("F:/U-CogNet-ToGo/sam3_src/assets/images/truck.jpg")
    if not image_path.exists():
        print(f"[viz] ERROR: image not found at {image_path}", file=sys.stderr)
        return 1

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] loading SAM 3 …")
    t0 = time.time()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device="cuda", load_from_HF=True)
    processor = Sam3Processor(model)
    print(f"[viz] model ready in {time.time()-t0:.1f}s")

    print(f"[viz] image: {image_path}")
    print(f"[viz] prompts: {prompts}")

    image = Image.open(image_path).convert("RGB")
    img_rgb = np.array(image)

    # Run all prompts in one session (state cached on the image).
    # For each prompt we keep ALL instances above `score_min`, so multi-
    # instance classes (robots, players) get rendered properly.
    score_min = 0.20
    results = []  # list[(prompt, [(mask, bbox, score), ...])]
    with torch.inference_mode(), \
         torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for prompt in prompts:
            t1 = time.time()
            out = processor.set_text_prompt(state=state, prompt=prompt)
            elapsed = time.time() - t1
            masks  = out.get("masks")
            boxes  = out.get("boxes")
            scores = out.get("scores")
            n = len(masks) if masks is not None else 0
            print(f"[viz]   '{prompt}' → {n} masks  ({elapsed*1000:.0f} ms)")
            instances = []
            if n > 0:
                # SAM 3 returns scores in bfloat16; numpy can't ingest it
                # directly, so promote to float32 before .cpu().numpy().
                scores_np = scores.detach().float().cpu().numpy()
                for i in np.argsort(-scores_np):
                    s = float(scores_np[i])
                    if s < score_min:
                        break
                    m = masks[i].detach().float().cpu().numpy()
                    b = boxes[i].detach().float().cpu().tolist() if boxes is not None else None
                    instances.append((m, b, s))
            results.append((prompt, instances))

    # ── render side-by-side grid: original + per-prompt overlay ──
    n_panels = 1 + len(prompts)
    cols = min(3, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.array(axes).reshape(rows, cols)

    # Panel 0 — original
    ax = axes[0, 0]
    ax.imshow(img_rgb)
    ax.set_title("ORIGINAL", fontsize=13, fontweight="bold")
    ax.axis("off")

    # Per-prompt panels — render ALL instances above threshold so
    # multi-instance classes (robots, players, balls) are all visible.
    for idx, (prompt, instances) in enumerate(results, start=1):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        colour = PALETTE[(idx - 1) % len(PALETTE)]
        if not instances:
            ax.imshow(img_rgb)
            ax.set_title(f"'{prompt}' — no mask", fontsize=12, color="red")
        else:
            vis = img_rgb.copy()
            for (mask, _, _) in instances:
                vis = overlay_mask(vis, mask, colour, alpha=0.45)
            for (_, bbox, score) in instances:
                if bbox is not None:
                    vis = draw_bbox(vis, bbox, f"{prompt[:18]}", colour, score)
            top_s = max(s for _, _, s in instances)
            ax.imshow(vis)
            ax.set_title(f"'{prompt}'  {len(instances)} inst.  top={top_s:.2f}",
                         fontsize=12, fontweight="bold")
        ax.axis("off")

    # Hide unused panels
    total_axes = rows * cols
    for k in range(n_panels, total_axes):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"SAM 3 · text-prompt segmentation · {image_path.name}",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.tight_layout()

    stem = image_path.stem
    out_path = args.out_dir / f"sam3_viz_{stem}_{int(time.time())}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    size_kb = out_path.stat().st_size / 1024
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[viz] saved {out_path}  ({size_kb:.0f} KB · peak VRAM {vram:.2f} GB)")
    if not args.no_open:
        print(f"[viz] opening in default image viewer …")
        open_file(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
