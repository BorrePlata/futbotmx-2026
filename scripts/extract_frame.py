"""extract_frame — pull a single frame out of a video for prompt prototyping.

Usage:
  python -m experiments.futbotmx.scripts.extract_frame --video PATH [--at SECONDS] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--at",    type=float, default=2.0, help="seconds into the video")
    ap.add_argument("--out",   type=Path, default=None,
                    help="output PNG (default: <video_stem>_t<sec>.png in output/)")
    args = ap.parse_args()

    import cv2

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cv2 cannot open {args.video}", file=sys.stderr)
        return 1
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dur    = nf / fps if fps else 0
    print(f"[frame] {args.video.name}  {w}x{h}  {fps:.1f} fps  {nf} frames  "
          f"({dur:.1f}s)")

    target = max(0, min(int(args.at * fps), nf - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"ERROR: read failed at frame {target}", file=sys.stderr)
        return 1

    out = args.out
    if out is None:
        out_dir = Path(__file__).resolve().parents[1] / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{args.video.stem}_t{int(args.at*1000):05d}ms.png"

    cv2.imwrite(str(out), frame)
    kb = out.stat().st_size / 1024
    print(f"[frame] saved {out}  ({kb:.0f} KB · t={target/fps:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
