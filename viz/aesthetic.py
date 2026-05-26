"""aesthetic — paper-grade compositor for FutBotMX outputs.

Builds a beautiful composed FRAME that keeps the video PRISTINE in a
left panel and lays all telemetry in a clean sidebar.  No overlay
covers the action.  Design intent:

  • U-CogNet research palette (deep navy bg, mint + amber accents)
  • Generous whitespace, no harsh borders
  • Sparkline trajectories for F / φ / surprise rolling window
  • MODEL-BREAK callout as a discreet pill, not a flashing border
  • Class detection table with stable per-class colour chips

The compositor exposes one entry point:

    AestheticCompositor(canvas_w, canvas_h).render(
        video_frame_bgr, sidebar_state)

where `sidebar_state` is the SidebarState dataclass.  Use the same
compositor for the single-arm video and for the side-by-side
comparison (just stack two horizontally).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# ── U-CogNet research palette (BGR for cv2) ─────────────────────
INK         = ( 16,  22,  34)         # deep navy bg
INK_SOFT    = ( 28,  36,  52)         # secondary bg
INK_HARD    = (  8,  12,  18)         # ultra-dark for the video frame letterbox
LINE        = ( 70,  82, 105)         # divider lines
LINE_SOFT   = ( 38,  48,  66)
TXT         = (245, 248, 252)         # primary text
TXT_MUTED   = (160, 175, 195)
TXT_DIM     = (105, 120, 140)

MINT        = (170, 235, 110)         # primary accent
MINT_SOFT   = ( 90, 175,  90)
AMBER       = (115, 200, 250)         # secondary accent (baseline arm)
AMBER_SOFT  = ( 90, 150, 195)
ROSE        = ( 95, 105, 240)         # warning (model break)
ROSE_SOFT   = ( 70,  85, 195)
TEAL        = (210, 195, 110)
LILAC       = (215, 155, 220)

# Per-class palette (matches baseline_sam3 CLASS_COLOURS_BGR for consistency)
CLASS_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "field":  (110, 200, 110),
    "robot":  (170, 235, 110),     # = MINT
    "ball":   (  0, 165, 255),
    "goal":   (255, 130,  20),
    "hand":   (180, 180, 200),
}


# ── sidebar state ───────────────────────────────────────────────
@dataclass
class SidebarState:
    """Everything the sidebar shows for one frame.  Keep it explicit."""
    arm_label:           str             # "BASELINE  | SAM 3 only" / "U-CogNet |…"
    arm_colour:          Tuple[int, int, int]   # accent colour for the arm

    frame_idx:           int             # current frame index (0-based)
    total_frames:        int             # total frames in source video
    infer_ms:            float           # SAM 3 inference latency
    cognitive_ms:        float = 0.0     # 0 when baseline arm

    # Per-class counts + top scores → rendered as a small table
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # entries: { 'robot': {'count': 3, 'top': 0.94}, ... }

    # Cognitive telemetry (only filled for U-CogNet arm)
    free_energy:         Optional[float] = None
    phi:                 Optional[float] = None
    recon_error:         Optional[float] = None
    understanding:       Optional[float] = None
    surprise_z:          Optional[float] = None
    warming_up:          bool = False
    model_break:         bool = False
    spatial_read:        str = ""

    # Rolling trajectories for the sparklines (caller owns the deques)
    F_trace:             Optional[List[float]] = None
    phi_trace:           Optional[List[float]] = None
    surprise_trace:      Optional[List[float]] = None


# ── compositor ──────────────────────────────────────────────────
class AestheticCompositor:
    """Compose a single 'panel' = title strip + video pane + sidebar +
    status strip.  Stack two of these horizontally for the side-by-side
    comparison."""

    def __init__(self, panel_w: int = 1280, panel_h: int = 720,
                 sidebar_w: int = 380, title_h: int = 60, status_h: int = 36,
                 pad: int = 18):
        self.W       = panel_w
        self.H       = panel_h
        self.SBW     = sidebar_w
        self.TH      = title_h
        self.STH     = status_h
        self.PAD     = pad
        self.VID_W   = panel_w - sidebar_w - 3 * pad
        self.VID_H   = panel_h - title_h - status_h - 2 * pad
        self.VID_X   = pad
        self.VID_Y   = title_h + pad
        self.SB_X    = panel_w - sidebar_w - pad
        self.SB_Y    = title_h + pad
        self.SB_H    = self.VID_H

    # ── helpers ─────────────────────────────────────────────────
    @staticmethod
    def _put_text(img, text, pos, size=0.5, colour=TXT, weight=1):
        import cv2
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size,
                    colour, weight, cv2.LINE_AA)

    @staticmethod
    def _text_w(text, size=0.5, weight=1):
        import cv2
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, weight)
        return w

    def _draw_title_bar(self, canvas, state: SidebarState):
        import cv2
        cv2.rectangle(canvas, (0, 0), (self.W, self.TH), INK_SOFT, -1)
        # Accent stripe — 4 px tall at the bottom of the title bar
        cv2.rectangle(canvas, (0, self.TH - 4), (self.W, self.TH),
                      state.arm_colour, -1)
        # Brand mark — small mint dot, then label
        cv2.circle(canvas, (self.PAD + 6, self.TH | 2), 7, MINT, -1)
        cv2.circle(canvas, (self.PAD + 6, self.TH | 2), 7, MINT_SOFT, 1)
        self._put_text(canvas, "U-CogNet |FutBotMX 2026",
                       (self.PAD + 22, self.TH | 2 + 5), size=0.55,
                       colour=TXT, weight=1)
        self._put_text(canvas, state.arm_label,
                       (self.W - self._text_w(state.arm_label, 0.55) - self.PAD,
                        self.TH | 2 + 5), size=0.55,
                       colour=state.arm_colour, weight=1)

    def _draw_status_bar(self, canvas, state: SidebarState):
        import cv2
        y0 = self.H - self.STH
        cv2.rectangle(canvas, (0, y0), (self.W, self.H), INK_SOFT, -1)
        cv2.rectangle(canvas, (0, y0), (self.W, y0 + 1), LINE_SOFT, -1)

        prog = (state.frame_idx + 1) / max(state.total_frames, 1)
        bar_x0, bar_w = self.PAD, self.W - 2 * self.PAD - 380
        bar_h = 6
        bar_y = y0 + (self.STH - bar_h) | 2
        cv2.rectangle(canvas, (bar_x0, bar_y),
                      (bar_x0 + bar_w, bar_y + bar_h), LINE_SOFT, -1)
        cv2.rectangle(canvas, (bar_x0, bar_y),
                      (bar_x0 + int(bar_w * prog), bar_y + bar_h),
                      state.arm_colour, -1)
        self._put_text(canvas, f"frame {state.frame_idx + 1:>4}/{state.total_frames}",
                       (bar_x0 + bar_w + 14, y0 + 22),
                       size=0.45, colour=TXT_MUTED)
        if state.cognitive_ms:
            lat = f"SAM 3 {state.infer_ms:>4.0f} ms  +  cog {state.cognitive_ms:>3.0f} ms"
        else:
            lat = f"SAM 3 {state.infer_ms:>4.0f} ms"
        self._put_text(canvas, lat,
                       (self.W - self._text_w(lat, 0.45) - self.PAD,
                        y0 + 22), size=0.45, colour=TXT_MUTED)

    def _draw_video(self, canvas, video_bgr):
        """Letterboxed video centred in the video pane.  Pristine."""
        import cv2
        vh, vw = video_bgr.shape[:2]
        # fit into (self.VID_W, self.VID_H) preserving aspect
        s = min(self.VID_W / vw, self.VID_H / vh)
        nw, nh = int(vw * s), int(vh * s)
        ox = self.VID_X + (self.VID_W - nw) | 2
        oy = self.VID_Y + (self.VID_H - nh) | 2
        # letterbox bg
        cv2.rectangle(canvas, (self.VID_X, self.VID_Y),
                      (self.VID_X + self.VID_W, self.VID_Y + self.VID_H),
                      INK_HARD, -1)
        if (nh, nw) != video_bgr.shape[:2]:
            video_bgr = cv2.resize(video_bgr, (nw, nh),
                                    interpolation=cv2.INTER_AREA)
        canvas[oy:oy + nh, ox:ox + nw] = video_bgr
        # 1-px subtle border
        cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + nw, oy + nh),
                      LINE_SOFT, 1)
        return (ox, oy, nw, nh)

    def _draw_break_pill(self, canvas, vid_box):
        """Discreet bottom-left pill when surprise fires.  No flashing
        border — paper-grade restraint."""
        import cv2
        ox, oy, nw, nh = vid_box
        text = "  MODEL BREAK  "
        tw = self._text_w(text, 0.5, 1) + 14
        px = ox + 12
        py = oy + nh - 36
        cv2.rectangle(canvas, (px, py), (px + tw, py + 26), ROSE, -1)
        cv2.rectangle(canvas, (px, py), (px + 4, py + 26), TXT, -1)
        self._put_text(canvas, text, (px + 8, py + 18),
                       size=0.5, colour=TXT, weight=1)

    # ── sidebar pieces ──────────────────────────────────────────
    def _section_header(self, canvas, x, y, label, accent):
        import cv2
        cv2.rectangle(canvas, (x, y), (x + 3, y + 14), accent, -1)
        self._put_text(canvas, label.upper(),
                       (x + 12, y + 12), size=0.45,
                       colour=TXT_MUTED, weight=1)

    def _kv_row(self, canvas, x, y, key, value, *,
                value_colour=TXT, key_colour=TXT_DIM, value_size=0.55,
                width=None):
        """key on the left dim, value on the right bright + monospaced look."""
        import cv2
        self._put_text(canvas, key, (x, y), size=0.46, colour=key_colour)
        w = width if width is not None else (self.SBW - 2 * self.PAD - 18)
        vstr = value
        self._put_text(canvas, vstr,
                       (x + w - self._text_w(vstr, value_size, 1), y),
                       size=value_size, colour=value_colour, weight=1)

    def _per_class_table(self, canvas, x, y, per_class: Dict[str, Dict[str, float]]):
        """Rows like:  ● robot      3   top 0.95"""
        import cv2
        for i, (cls, v) in enumerate(per_class.items()):
            row_y = y + i * 22
            colour = CLASS_PALETTE.get(cls, (200, 200, 200))
            cv2.circle(canvas, (x + 4, row_y - 4), 5, colour, -1)
            self._put_text(canvas, cls,
                           (x + 16, row_y), size=0.5, colour=TXT)
            count_str = f"{int(v.get('count', 0)):>2}"
            self._put_text(canvas, count_str,
                           (x + 130, row_y), size=0.5, colour=TXT, weight=1)
            top = v.get('top', 0.0)
            if top > 0:
                top_str = f"top {top:.2f}"
                self._put_text(canvas, top_str,
                               (x + 180, row_y), size=0.45, colour=TXT_MUTED)
        return y + len(per_class) * 22

    def _sparkline(self, canvas, x, y, w, h, values: List[float],
                   colour, label: str, value_str: str,
                   warming: bool = False, fill_alpha: float = 0.18):
        """Sparkline with axis-less mini-chart of recent rolling values."""
        import cv2
        # box background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), INK_SOFT, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), LINE_SOFT, 1)
        # label + current value
        self._put_text(canvas, label.upper(),
                       (x + 6, y + 12), size=0.38, colour=TXT_MUTED)
        self._put_text(canvas, value_str,
                       (x + w - self._text_w(value_str, 0.45, 1) - 6, y + 12),
                       size=0.45, colour=colour, weight=1)
        # plot
        if not values or len(values) < 2:
            return
        vals = np.array(values, dtype=np.float32)
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = max(vmax - vmin, 1e-6)
        n = len(vals)
        ix0, iy0 = x + 6, y + 20
        ix1, iy1 = x + w - 6, y + h - 8
        xs = np.linspace(ix0, ix1, n)
        ys = iy1 - (vals - vmin) / rng * (iy1 - iy0)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        # fill under curve
        poly = np.concatenate([pts, [[ix1, iy1], [ix0, iy1]]]).astype(np.int32)
        ov = canvas.copy()
        cv2.fillPoly(ov, [poly], colour)
        cv2.addWeighted(ov, fill_alpha, canvas, 1 - fill_alpha, 0, dst=canvas)
        cv2.polylines(canvas, [pts], False, colour, 1, cv2.LINE_AA)
        # last-point dot
        cv2.circle(canvas, tuple(pts[-1]), 3, colour, -1)
        if warming:
            self._put_text(canvas, "warmup",
                           (x + w - self._text_w("warmup", 0.38, 1) - 6,
                            y + h - 4), size=0.38, colour=TXT_DIM)

    # ── full sidebar ────────────────────────────────────────────
    def _draw_sidebar(self, canvas, state: SidebarState):
        import cv2
        x0 = self.SB_X
        y  = self.SB_Y

        # sidebar bg
        cv2.rectangle(canvas, (x0 - 4, y - 4),
                      (x0 + self.SBW + 4, y + self.SB_H + 4),
                      INK_SOFT, -1)
        cv2.rectangle(canvas, (x0 - 4, y - 4),
                      (x0 + self.SBW + 4, y + self.SB_H + 4),
                      LINE_SOFT, 1)

        inner_x = x0 + self.PAD
        inner_w = self.SBW - 2 * self.PAD
        cur_y   = y + self.PAD + 12

        # ── PERCEPTION (SAM 3) ──────────────────────────────────
        self._section_header(canvas, inner_x, cur_y - 12,
                              "Perception |SAM 3", AMBER)
        cur_y += 16
        if state.per_class:
            cur_y = self._per_class_table(canvas, inner_x, cur_y + 8,
                                           state.per_class) + 12

        # ── COGNITION (U-CogNet) ────────────────────────────────
        if state.free_energy is not None:
            cur_y += 6
            self._section_header(canvas, inner_x, cur_y,
                                  "Cognition |U-CogNet", MINT)
            cur_y += 28

            # primary metrics — clean key/value rows
            self._kv_row(canvas, inner_x, cur_y, "free energy F",
                          f"{state.free_energy:+.3f}",
                          value_colour=MINT)
            cur_y += 22
            self._kv_row(canvas, inner_x, cur_y, "phi |integrated info",
                          f"{state.phi:.3f}",
                          value_colour=LILAC)
            cur_y += 22
            self._kv_row(canvas, inner_x, cur_y, "recon error",
                          f"{state.recon_error:.4f}",
                          value_colour=TEAL)
            cur_y += 22
            self._kv_row(canvas, inner_x, cur_y, "understanding",
                          f"{(state.understanding or 0)*100:.0f} %",
                          value_colour=TXT)
            cur_y += 22
            sz_colour = ROSE if state.model_break else TXT
            self._kv_row(canvas, inner_x, cur_y, "surprise z-score",
                          f"{state.surprise_z:+.2f}",
                          value_colour=sz_colour)
            cur_y += 32

            # Sparklines panel
            if state.F_trace and state.phi_trace and state.surprise_trace:
                spw = inner_w
                sph = 56
                gap = 8
                self._sparkline(canvas, inner_x, cur_y, spw, sph,
                                 state.F_trace, MINT, "F (free energy)",
                                 f"{state.free_energy:+.3f}",
                                 warming=state.warming_up)
                cur_y += sph + gap
                self._sparkline(canvas, inner_x, cur_y, spw, sph,
                                 state.phi_trace, LILAC, "phi",
                                 f"{state.phi:.3f}",
                                 warming=state.warming_up)
                cur_y += sph + gap
                self._sparkline(canvas, inner_x, cur_y, spw, sph,
                                 state.surprise_trace, ROSE, "surprise z",
                                 f"{state.surprise_z:+.2f}",
                                 warming=state.warming_up)
                cur_y += sph + 4

            # spatial read sticky strip at the bottom of the sidebar
            if state.spatial_read:
                strip_y = y + self.SB_H - 50
                cv2.rectangle(canvas, (inner_x, strip_y),
                              (inner_x + inner_w, strip_y + 38),
                              INK_HARD, -1)
                cv2.rectangle(canvas, (inner_x, strip_y),
                              (inner_x + 3, strip_y + 38), ROSE, -1)
                self._put_text(canvas, "SPATIAL READ",
                                (inner_x + 10, strip_y + 14), size=0.36,
                                colour=TXT_DIM)
                self._put_text(canvas, state.spatial_read[:60],
                                (inner_x + 10, strip_y + 30), size=0.45,
                                colour=TXT)

    # ── full panel ──────────────────────────────────────────────
    def render(self, video_bgr: np.ndarray, state: SidebarState) -> np.ndarray:
        """Returns a (H, W, 3) BGR composed frame."""
        canvas = np.full((self.H, self.W, 3), INK, dtype=np.uint8)
        self._draw_title_bar(canvas, state)
        vid_box = self._draw_video(canvas, video_bgr)
        if state.model_break:
            self._draw_break_pill(canvas, vid_box)
        self._draw_sidebar(canvas, state)
        self._draw_status_bar(canvas, state)
        return canvas


# ── trace buffer helper ─────────────────────────────────────────
class TraceBuffers:
    """Rolling-window storage for the sparkline traces.  The pipeline
    pushes values per frame; the renderer reads the deques as lists."""

    def __init__(self, window: int = 120):
        self.window = window
        self.F:        Deque[float] = deque(maxlen=window)
        self.phi:      Deque[float] = deque(maxlen=window)
        self.surprise: Deque[float] = deque(maxlen=window)

    def push(self, F: float, phi: float, surprise: float) -> None:
        self.F.append(float(F))
        self.phi.append(float(phi))
        self.surprise.append(float(surprise))

    def as_lists(self) -> Tuple[List[float], List[float], List[float]]:
        return list(self.F), list(self.phi), list(self.surprise)
