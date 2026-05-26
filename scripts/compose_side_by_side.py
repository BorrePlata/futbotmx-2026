"""compose_side_by_side — render a paper-grade BASELINE | U-COGNET MP4.

Takes the two arm outputs (`<stem>_baseline.mp4` and `<stem>_ucognet.mp4`)
and concatenates them horizontally with title strips so the comparison
is unambiguous.  This is what the demo video + paper figure use.

Usage:
  python -m experiments.futbotmx.scripts.compose_side_by_side `
    --baseline output/IMG_9914_baseline.mp4 `
    --ucognet  output/IMG_9914_ucognet.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


TITLE_H = 56
TITLE_BG = (16, 22, 34)


def _title_strip(width: int, text: str, accent_bgr) -> np.ndarray:
    import cv2
    strip = np.full((TITLE_H, width, 3), TITLE_BG, dtype=np.uint8)
    cv2.line(strip, (0, TITLE_H - 1), (width, TITLE_H - 1), accent_bgr, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(strip, text, ((width - tw) // 2, TITLE_H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 245, 255), 2,
                cv2.LINE_AA)
    # left accent stripe
    cv2.rectangle(strip, (0, 0), (8, TITLE_H), accent_bgr, -1)
    return strip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--ucognet",  type=Path, required=True)
    ap.add_argument("--out",      type=Path, default=None)
    ap.add_argument("--open",     action="store_true")
    args = ap.parse_args()

    import cv2
    for p in (args.baseline, args.ucognet):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    cap_b = cv2.VideoCapture(str(args.baseline))
    cap_u = cv2.VideoCapture(str(args.ucognet))
    fps   = cap_b.get(cv2.CAP_PROP_FPS) or 30.0
    nf_b  = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    nf_u  = int(cap_u.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    nf    = min(nf_b, nf_u)
    W     = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H     = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out_path = args.out or args.baseline.parent / f"{args.baseline.stem.replace('_baseline','')}_sxs.mp4"
    out_W = 2 * W + 8                  # 8 px divider
    out_H = TITLE_H + H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))

    title_b = _title_strip(W, "BASELINE   |   SAM 3 only",
                            accent_bgr=(110, 220, 255))   # amber-ish BGR
    title_u = _title_strip(W, "U-CogNet   |   SAM 3 + cognitive stack",
                            accent_bgr=(100, 240, 170))   # mint BGR
    divider = np.full((out_H, 8, 3), (10, 14, 22), dtype=np.uint8)
    title_row = np.hstack([title_b, np.full((TITLE_H, 8, 3), TITLE_BG, dtype=np.uint8), title_u])

    print(f"[sxs] {nf} frames · {out_W}x{out_H} @ {fps:.1f}fps → {out_path}")
    for i in range(nf):
        okb, fb = cap_b.read()
        oku, fu = cap_u.read()
        if not (okb and oku):
            break
        if (fb.shape[0], fb.shape[1]) != (H, W):
            fb = cv2.resize(fb, (W, H))
        if (fu.shape[0], fu.shape[1]) != (H, W):
            fu = cv2.resize(fu, (W, H))
        body = np.hstack([fb, np.full((H, 8, 3), (10, 14, 22), dtype=np.uint8), fu])
        composite = np.vstack([title_row, body])
        writer.write(composite)
    cap_b.release(); cap_u.release(); writer.release()

    size_mb = out_path.stat().st_size / 1e6
    print(f"[sxs] ✅ {out_path}  ({size_mb:.1f} MB)")
    if args.open:
        import os
        if sys.platform == "win32":
            os.startfile(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
