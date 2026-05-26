"""palette — paper-grade plot styling for FutBotMX evaluation figures.

Single source of truth for the matplotlib styling used by every
`evaluation/*.py` script.  Mirrors the U-CogNet research deck palette
so figures shipped in PAPER.md are visually consistent with the
side-by-side video, the live demo and the Vet Microscopy AI deck.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ── canonical colours (HEX, matplotlib-friendly) ───────────────
INK         = "#0a0e16"     # deep navy bg
INK_SOFT    = "#141a26"
INK_HARD    = "#06080c"
TXT         = "#f4f7fc"
TXT_MUTED   = "#a4b0c0"
TXT_DIM     = "#6c7990"
LINE        = "#3a4256"
LINE_SOFT   = "#262d3e"

MINT        = "#6eebaa"     # primary accent (U-CogNet arm)
MINT_SOFT   = "#5aaf80"
AMBER       = "#fac873"     # secondary accent (baseline arm)
AMBER_SOFT  = "#c89e5e"
ROSE        = "#f0696b"     # surprise / model break
LILAC       = "#dc9bd7"     # phi
TEAL        = "#73c0d2"     # recon
GOLD        = "#e8c54e"     # highlight

# Per-class colours (match aesthetic.py CLASS_PALETTE, RGB-converted)
CLASS_COLORS = {
    "field":  "#6ec86e",
    "robot":  "#6eebaa",     # = MINT
    "ball":   "#ffa500",
    "goal":   "#1482ff",
    "hand":   "#c8c8b4",
}


def apply_paper_style():
    """Set matplotlib rcParams once per script for paper-grade output."""
    rcParams.update({
        # canvas
        "figure.facecolor":     INK,
        "savefig.facecolor":    INK,
        "axes.facecolor":       INK_SOFT,
        # text
        "text.color":           TXT,
        "axes.labelcolor":      TXT,
        "axes.titlecolor":      TXT,
        "xtick.color":          TXT_MUTED,
        "ytick.color":          TXT_MUTED,
        "legend.labelcolor":    TXT,
        # frame
        "axes.edgecolor":       LINE,
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "grid.color":           LINE_SOFT,
        "grid.alpha":           0.5,
        "grid.linestyle":       "-",
        "grid.linewidth":       0.4,
        # legend
        "legend.facecolor":     INK_HARD,
        "legend.edgecolor":     LINE,
        "legend.framealpha":    0.85,
        "legend.fontsize":      9.5,
        # fonts (fall back gracefully if Inter/Sora not installed)
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Inter", "Helvetica", "DejaVu Sans"],
        "font.size":            10.5,
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       11,
        "xtick.labelsize":      9.5,
        "ytick.labelsize":      9.5,
        # spine
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        # misc
        "savefig.dpi":          150,
        "figure.dpi":           110,
    })


def watermark(fig, *, arm: str = "", note: str = ""):
    """Tiny attribution stripe at the bottom of every figure."""
    text = "U-CogNet | FutBotMX 2026"
    if arm:
        text += f"  ·  {arm}"
    if note:
        text += f"  ·  {note}"
    fig.text(0.02, 0.005, text, color=TXT_DIM, fontsize=8.5, ha="left",
             va="bottom", family="monospace")
