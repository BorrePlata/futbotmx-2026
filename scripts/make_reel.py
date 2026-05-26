"""make_reel — generate a 1080x1920 (9:16) Instagram Reel for FutBotMX.

Builds a 35-second portrait video from the existing aesthetic outputs:

  Section            Duration   Source
  ─────────────────  ─────────  ─────────────────────────────────────────
  Title card           3.0 s    title screen with project name
  Hook caption         3.0 s    "AI that knows when it's wrong"
  Baseline playback    7.0 s    IMG_9914_baseline_aesthetic.mp4 (2x slow)
  Transition           1.0 s    fade + caption
  U-CogNet playback    7.0 s    IMG_9914_ucognet_aesthetic.mp4 (2x slow)
  Side-by-side dual    8.0 s    IMG_9914_FINAL_sxs_v2.mp4 (2x slow)
  Stats card           3.0 s    headline numbers
  CTA + repo URL       3.0 s    https://github.com/BorrePlata/futbotmx-2026

Total: ~35 seconds at 30 fps · respects Instagram safe zones · uses
the U-CogNet research palette so it's consistent with the deck and the
side-by-side video.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


# ── Reel canvas geometry ───────────────────────────────────────
W, H        = 1080, 1920
FPS         = 30
SAFE_TOP    = 220        # IG action chrome (camera + profile)
SAFE_BOT    = 380        # IG caption/like/share area
VIDEO_AREA_H = H - SAFE_TOP - SAFE_BOT
VIDEO_AREA_TOP = SAFE_TOP
VIDEO_AREA_W   = W - 60  # 30 px margins each side
VIDEO_AREA_X   = 30

# ── palette (BGR, mirrors viz/aesthetic.py) ───────────────────
INK         = ( 16,  22,  34)
INK_SOFT    = ( 28,  36,  52)
INK_HARD    = (  8,  12,  18)
TXT         = (245, 248, 252)
TXT_MUTED   = (160, 175, 195)
TXT_DIM     = (105, 120, 140)
MINT        = (170, 235, 110)
MINT_SOFT   = ( 90, 175,  90)
AMBER       = (115, 200, 250)
ROSE        = ( 95, 105, 240)
GOLD        = (115, 200, 250)


def _put(img, text, pos, size=1.0, colour=TXT, weight=2, font=None):
    import cv2
    f = font if font is not None else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, f, size, colour, weight, cv2.LINE_AA)


def _tw(text, size=1.0, weight=2):
    import cv2
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, weight)
    return w


def _center_text(img, text, y, size=1.0, colour=TXT, weight=2):
    """Draw text horizontally centred at row y."""
    x = (W - _tw(text, size, weight)) // 2
    _put(img, text, (x, y), size=size, colour=colour, weight=weight)


def _blank_frame() -> np.ndarray:
    return np.full((H, W, 3), INK, dtype=np.uint8)


def _decorate_chrome(frame: np.ndarray, section_label: str = "",
                      accent: Tuple[int, int, int] = MINT) -> None:
    """Common chrome: top brand bar + bottom status."""
    import cv2

    # Top brand strip
    cv2.rectangle(frame, (0, 0), (W, 120), INK_SOFT, -1)
    cv2.rectangle(frame, (0, 120 - 4), (W, 120), accent, -1)
    cv2.circle(frame, (44, 60), 12, MINT, -1)
    cv2.circle(frame, (44, 60), 12, MINT_SOFT, 2)
    _put(frame, "U-CogNet  |  FutBotMX 2026",
         (74, 70), size=0.78, colour=TXT, weight=2)

    if section_label:
        sw = _tw(section_label, 0.55, 2) + 24
        cv2.rectangle(frame, (W - sw - 30, 18),
                      (W - 30, 18 + 36), accent, -1)
        _put(frame, section_label,
             (W - sw - 30 + 12, 42),
             size=0.55, colour=INK_HARD, weight=2)

    # Bottom strip — small attribution
    cv2.rectangle(frame, (0, H - 70), (W, H), INK_SOFT, -1)
    cv2.rectangle(frame, (0, H - 70), (W, H - 70 + 3), accent, -1)
    _put(frame, "github.com/BorrePlata/futbotmx-2026",
         (40, H - 30), size=0.65, colour=TXT_MUTED, weight=2)


def _embed_video_pane(frame: np.ndarray, video_frame_bgr: np.ndarray,
                      x: int, y: int, w: int, h: int) -> None:
    """Letterboxed embed of a (possibly different-aspect) video frame."""
    import cv2
    cv2.rectangle(frame, (x, y), (x + w, y + h), INK_HARD, -1)
    vh, vw = video_frame_bgr.shape[:2]
    s = min(w / vw, h / vh)
    nw, nh = int(vw * s), int(vh * s)
    ox = x + (w - nw) // 2
    oy = y + (h - nh) // 2
    resized = cv2.resize(video_frame_bgr, (nw, nh),
                          interpolation=cv2.INTER_AREA)
    frame[oy:oy + nh, ox:ox + nw] = resized
    # subtle border
    cv2.rectangle(frame, (ox - 1, oy - 1), (ox + nw, oy + nh),
                  (38, 48, 66), 1)


# ── card builders ─────────────────────────────────────────────
def card_title(writer, duration: float) -> None:
    import cv2
    n = int(duration * FPS)
    for i in range(n):
        frame = _blank_frame()
        # subtle radial glow
        cy = H // 2
        overlay = frame.copy()
        cv2.circle(overlay, (W // 2, cy), 700, INK_SOFT, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
        # mint dot logo
        cv2.circle(frame, (W // 2, cy - 280), 26, MINT, -1)
        cv2.circle(frame, (W // 2, cy - 280), 26, MINT_SOFT, 3)
        _center_text(frame, "U-CogNet", cy - 80,
                     size=3.0, colour=TXT, weight=4)
        _center_text(frame, "x  SAM 3", cy + 30,
                     size=1.8, colour=TXT_MUTED, weight=3)
        _center_text(frame, "FutBotMX  2026", cy + 200,
                     size=1.2, colour=MINT, weight=3)
        _center_text(frame, "Computer Vision  |  Profesional",
                     cy + 280, size=0.9, colour=TXT_DIM, weight=2)
        # progress
        prog = (i + 1) / max(n, 1)
        cv2.rectangle(frame, (W // 2 - 200, cy + 420),
                      (W // 2 + 200, cy + 426), INK_SOFT, -1)
        cv2.rectangle(frame, (W // 2 - 200, cy + 420),
                      (W // 2 - 200 + int(400 * prog), cy + 426), MINT, -1)
        writer.write(frame)


def card_hook(writer, duration: float) -> None:
    n = int(duration * FPS)
    for i in range(n):
        frame = _blank_frame()
        _decorate_chrome(frame, section_label="HOOK")
        cy = H // 2
        _center_text(frame, "What if your AI", cy - 200,
                     size=1.8, colour=TXT, weight=3)
        _center_text(frame, "did not just SEE", cy - 100,
                     size=1.8, colour=TXT, weight=3)
        _center_text(frame, "but UNDERSTOOD?", cy + 30,
                     size=2.4, colour=MINT, weight=4)
        _center_text(frame, "and knew when it was WRONG?", cy + 160,
                     size=1.3, colour=ROSE, weight=3)
        writer.write(frame)


def card_video(writer, video_path: Path, duration: float, *,
                section_label: str, caption_top: str, caption_bottom: str,
                accent: Tuple[int, int, int], slow: float = 2.0) -> None:
    """Play video looped and slowed to fill `duration` seconds."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # cache all frames once (small clip)
    src_frames: List[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        src_frames.append(f)
    cap.release()
    if not src_frames:
        raise RuntimeError(f"no frames in {video_path}")

    n_target = int(duration * FPS)
    # Each src frame shown `slow` times, then loop
    for i in range(n_target):
        src_idx = int((i / slow) % len(src_frames))
        src = src_frames[src_idx]

        frame = _blank_frame()
        _decorate_chrome(frame, section_label=section_label, accent=accent)
        # caption top
        if caption_top:
            _center_text(frame, caption_top, 180,
                         size=0.9, colour=TXT, weight=2)
        # video pane
        _embed_video_pane(frame, src,
                           x=VIDEO_AREA_X, y=VIDEO_AREA_TOP + 60,
                           w=VIDEO_AREA_W, h=VIDEO_AREA_H - 140)
        # caption bottom
        if caption_bottom:
            _center_text(frame, caption_bottom, H - 130,
                         size=0.85, colour=TXT_MUTED, weight=2)
        writer.write(frame)


def card_transition(writer, duration: float) -> None:
    n = int(duration * FPS)
    for i in range(n):
        frame = _blank_frame()
        prog = (i + 1) / max(n, 1)
        # animated arrow
        cy = H // 2
        _center_text(frame, "Now watch what happens",
                     cy - 80, size=1.4, colour=TXT_MUTED, weight=2)
        _center_text(frame, "when we add a cognitive layer",
                     cy, size=1.4, colour=TXT, weight=3)
        # mint arrow
        ax = int(W * 0.5 - 60 + 120 * prog)
        ay = cy + 120
        import cv2
        cv2.arrowedLine(frame, (ax - 80, ay), (ax + 60, ay),
                         MINT, 6, line_type=cv2.LINE_AA, tipLength=0.30)
        writer.write(frame)


def card_stats(writer, duration: float) -> None:
    import cv2
    n = int(duration * FPS)
    stats = [
        ("mAP @ 0.5",         "0.984",  MINT,  "vs temporal-consistency pseudo-GT"),
        ("Robot recall",      "100 %",  MINT,  "every frame, every robot"),
        ("MOTA  /  MOTP",     "0.950 / 1.000", AMBER, "tracking quality"),
        ("Model breaks",      "7",      ROSE,  "events the baseline can't see"),
        ("Cognitive overhead", "+6.4 %", GOLD,  "wall-time vs baseline"),
    ]
    for i in range(n):
        frame = _blank_frame()
        _decorate_chrome(frame, section_label="RESULTS", accent=MINT)
        _center_text(frame, "What the numbers say",
                     230, size=1.2, colour=TXT, weight=3)
        y0 = 360
        for k, (label, value, colour, sub) in enumerate(stats):
            y = y0 + k * 200
            # mini card
            cv2.rectangle(frame, (60, y), (W - 60, y + 160),
                          INK_SOFT, -1)
            cv2.rectangle(frame, (60, y), (66, y + 160),
                          colour, -1)
            _put(frame, label, (96, y + 50),
                 size=0.95, colour=TXT_MUTED, weight=2)
            _put(frame, value, (96, y + 115),
                 size=1.8, colour=colour, weight=4)
            sub_w = _tw(sub, 0.6, 2)
            _put(frame, sub, (W - 60 - sub_w - 18, y + 115),
                 size=0.6, colour=TXT_DIM, weight=2)
        writer.write(frame)


def card_cta(writer, duration: float) -> None:
    import cv2
    n = int(duration * FPS)
    for i in range(n):
        frame = _blank_frame()
        _decorate_chrome(frame, section_label="OPEN SCIENCE", accent=MINT)
        cy = H // 2
        cv2.circle(frame, (W // 2, cy - 280), 36, MINT, -1)
        cv2.circle(frame, (W // 2, cy - 280), 36, MINT_SOFT, 4)
        _center_text(frame, "Open source.", cy - 140,
                     size=1.6, colour=TXT, weight=3)
        _center_text(frame, "Apache 2.0.  Reproducible.",
                     cy - 60, size=1.1, colour=TXT_MUTED, weight=2)
        _center_text(frame, "github.com/BorrePlata", cy + 80,
                     size=1.0, colour=MINT, weight=3)
        _center_text(frame, "/futbotmx-2026", cy + 140,
                     size=1.5, colour=MINT, weight=4)
        _center_text(frame, "paper  +  code  +  figures  +  demo video",
                     cy + 240, size=0.7, colour=TXT_DIM, weight=2)
        _center_text(frame, "Brainstream | U-CogNet research platform",
                     cy + 380, size=0.65, colour=TXT_DIM, weight=2)
        writer.write(frame)


# ── driver ────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline",  type=Path,
                    default=_ROOT / "output" / "IMG_9914_baseline_aesthetic.mp4")
    ap.add_argument("--ucognet",   type=Path,
                    default=_ROOT / "output" / "IMG_9914_ucognet_aesthetic.mp4")
    ap.add_argument("--sxs",       type=Path,
                    default=_ROOT / "output" / "IMG_9914_FINAL_sxs_v2.mp4")
    ap.add_argument("--out",       type=Path,
                    default=_ROOT / "paper" / "reel.mp4")
    args = ap.parse_args()

    for p in (args.baseline, args.ucognet, args.sxs):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr); return 1

    import cv2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, FPS, (W, H))
    if not writer.isOpened():
        print(f"ERROR: cv2.VideoWriter failed for {args.out}", file=sys.stderr)
        return 1

    print(f"[reel] writing 1080x1920@{FPS}fps to {args.out}")
    print("[reel] 1/7  title card        (3.0s)")
    card_title(writer, 3.0)
    print("[reel] 2/7  hook caption      (3.0s)")
    card_hook(writer, 3.0)
    print("[reel] 3/7  baseline playback (7.0s)")
    card_video(writer, args.baseline, 7.0,
               section_label="BASELINE",
               caption_top="SAM 3 alone sees what's there ...",
               caption_bottom="... but has no model of the match",
               accent=AMBER, slow=2.0)
    print("[reel] 4/7  transition        (1.0s)")
    card_transition(writer, 1.0)
    print("[reel] 5/7  U-CogNet playback (7.0s)")
    card_video(writer, args.ucognet, 7.0,
               section_label="U-CogNet",
               caption_top="Same SAM 3 + a free-energy cognitive layer",
               caption_bottom="Auto-flags surprise + spatial-correlation read",
               accent=MINT, slow=2.0)
    print("[reel] 6/7  side-by-side      (8.0s)")
    card_video(writer, args.sxs, 8.0,
               section_label="COMPARISON",
               caption_top="100 % detection agreement",
               caption_bottom="U-CogNet adds 7 model-break events the baseline can't see",
               accent=MINT, slow=2.0)
    print("[reel] 7/8  stats card        (3.0s)")
    card_stats(writer, 3.0)
    print("[reel] 8/8  CTA card          (3.0s)")
    card_cta(writer, 3.0)

    writer.release()
    size_mb = args.out.stat().st_size / 1e6
    print(f"[reel] OK  {args.out}  ({size_mb:.1f} MB, ~35s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
