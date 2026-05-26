"""match — robot-soccer cognition wrapper for the U-CogNet arm.

Three pieces:
  • `encode_raw_observation` — splats per-class foot-points into a
        (GRID_H, GRID_W, 4) occupancy grid (team A / team B / other /
        ball channels) and flattens it for the SingularityEngine.
  • `occupancy_grid` — the same grid in (H, W, 3) shape used by the
        spatial-correlation read when surprise fires.
  • `MatchCognition` — owns the SingularityEngine + RealtimeReasoner and
        emits one `CognitionTick` per frame with all metrics in flat schema.

Honest scope (carried into PAPER.md):
  - The cognitive layer reads STRUCTURE in the occupancy map, not class
    identity — team labels are positional in v0.1.
  - `correlate_surprise` reports *where* the perception delta is largest;
    it is an honest correlation, NOT a tactical claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .singularity import SingularityEngine
from .reasoner    import RealtimeReasoner


# Occupancy grid resolution — a perceptual parameter, not a tactical
# choice.  24x16 (W x H) keeps the bottleneck small enough for the
# Singularity engine's hashed projection while preserving enough spatial
# detail for the robot soccer playing area.
GRID_W, GRID_H = 24, 16
_CHANNELS = 4         # 0=team A, 1=team B, 2=other, 3=ball


# ── splatting primitive ────────────────────────────────────────
def _splat(grid_ch: np.ndarray, x_norm: float, y_norm: float,
           weight: float = 1.0) -> None:
    """Soft 3x3 Gaussian-ish bump for one point into one grid channel."""
    h, w = grid_ch.shape
    cx = x_norm * (w - 1)
    cy = y_norm * (h - 1)
    ix, iy = int(np.floor(cx)), int(np.floor(cy))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            gx, gy = ix + dx, iy + dy
            if 0 <= gx < w and 0 <= gy < h:
                d2 = (gx - cx) ** 2 + (gy - cy) ** 2
                grid_ch[gy, gx] += weight * float(np.exp(-d2 / 1.2))


# ── encoders ───────────────────────────────────────────────────
def encode_raw_observation(frame_shape: Tuple[int, int],
                            team_a_pts: np.ndarray,
                            team_b_pts: np.ndarray,
                            other_pts: np.ndarray,
                            ball_xy: Optional[Tuple[float, float]],
                            pitch_coverage: float = -1.0,
                            recenter: bool = False) -> np.ndarray:
    """Flatten the (GRID_H, GRID_W, 4) occupancy + 2 scalars to a vector.

    `recenter=True` maps coordinates relative to the players' collective
    centroid — useful with moving cameras; we keep absolute mapping for
    the fixed robot-soccer overhead shot."""
    h, w = frame_shape
    grid = np.zeros((GRID_H, GRID_W, _CHANNELS), dtype=np.float64)

    if recenter:
        all_pts = [p for p in (team_a_pts, team_b_pts, other_pts) if len(p)]
        if ball_xy is not None:
            all_pts.append(np.array([ball_xy]))
        if all_pts:
            stk = np.vstack(all_pts)
            c = stk.mean(axis=0)
            s = stk.std(axis=0) * 2.0 + 1e-3
        else:
            c, s = np.array([w / 2, h / 2]), np.array([w / 2, h / 2])

        def _map(x, y):
            return (float(np.clip(0.5 + (x - c[0]) / (4 * s[0]), 0, 1)),
                    float(np.clip(0.5 + (y - c[1]) / (4 * s[1]), 0, 1)))
    else:
        def _map(x, y):
            return (float(np.clip(x / max(w - 1, 1), 0, 1)),
                    float(np.clip(y / max(h - 1, 1), 0, 1)))

    def _add(pts: np.ndarray, ch: int, weight: float) -> None:
        for x, y in np.atleast_2d(pts):
            if w > 1 and h > 1:
                xn, yn = _map(x, y)
                _splat(grid[:, :, ch], xn, yn, weight)

    if len(team_a_pts):
        _add(team_a_pts, 0, 1.0)
    if len(team_b_pts):
        _add(team_b_pts, 1, 1.0)
    if len(other_pts):
        _add(other_pts, 2, 1.0)
    if ball_xy is not None:
        _add(np.array([ball_xy]), 3, 2.0)

    obs = grid.reshape(-1)
    field_scalar = pitch_coverage if pitch_coverage >= 0 else 0.0
    return np.concatenate([obs, [field_scalar, 1.0 if ball_xy else 0.0]])


def occupancy_grid(team_a_pts: np.ndarray, team_b_pts: np.ndarray,
                   ball_xy: Optional[Tuple[float, float]],
                   frame_shape: Tuple[int, int]) -> np.ndarray:
    """(H, W, 3) occupancy used by the spatial-correlation read.  Drops
    the 'other' channel (refs / hands don't drive tactical structure)."""
    h, w = frame_shape
    grid = np.zeros((GRID_H, GRID_W, 3), dtype=np.float64)
    for ch, pts in ((0, team_a_pts), (1, team_b_pts)):
        for x, y in np.atleast_2d(pts) if len(pts) else []:
            _splat(grid[:, :, ch],
                   float(np.clip(x / max(w - 1, 1), 0, 1)),
                   float(np.clip(y / max(h - 1, 1), 0, 1)))
    if ball_xy is not None:
        _splat(grid[:, :, 2],
               float(np.clip(ball_xy[0] / max(w - 1, 1), 0, 1)),
               float(np.clip(ball_xy[1] / max(h - 1, 1), 0, 1)), 2.0)
    return grid


# ── honest spatial-correlation read ────────────────────────────
def correlate_surprise(prev_grid: np.ndarray, grid: np.ndarray
                       ) -> Tuple[str, float]:
    """When surprise fires, report WHERE on the field the occupancy map
    changed most.  Honest correlation, not a tactical claim.

    Returns (human-readable string, confidence in [0, 1])."""
    delta = grid - prev_grid
    abs_delta = np.abs(delta)
    if abs_delta.sum() < 1e-6:
        return "", 0.0

    # Per-channel which region moved most
    H, W, C = delta.shape
    # Coarsen to a 3x3 spatial grid for the human-readable description
    rows = H // 3 if H >= 3 else 1
    cols = W // 3 if W >= 3 else 1
    coarse = np.zeros((3, 3, C))
    for r in range(3):
        for c in range(3):
            r0, r1 = r * rows, (r + 1) * rows if r < 2 else H
            c0, c1 = c * cols, (c + 1) * cols if c < 2 else W
            coarse[r, c] = abs_delta[r0:r1, c0:c1].sum(axis=(0, 1))

    # Which channel changed most + where
    total_per_ch = coarse.sum(axis=(0, 1))
    ch = int(np.argmax(total_per_ch))
    ch_name = ["Team A", "Team B", "ball"][ch] if C >= 3 else "scene"

    cell_intensity = coarse[:, :, ch]
    r, c = np.unravel_index(np.argmax(cell_intensity), cell_intensity.shape)
    horiz = ["left", "centre", "right"][c]
    vert  = ["top", "middle", "bottom"][r]
    confidence = float(cell_intensity[r, c] /
                       (cell_intensity.sum() + 1e-9))
    text = f"shift concentrated on the {vert}-{horiz}, dominated by {ch_name}"
    return text, round(confidence, 3)


# ── per-frame record ──────────────────────────────────────────
@dataclass
class CognitionTick:
    frame_idx:     int
    recon_error:   float
    free_energy:   float
    phi:           float
    coherence:     float
    understanding: float
    surprise_z:    float
    is_surprised:  bool
    warming_up:    bool
    spatial_read:  str = ""
    read_conf:     float = 0.0

    def to_dict(self) -> Dict:
        return {
            "frame_idx":     self.frame_idx,
            "recon_error":   round(self.recon_error, 6),
            "free_energy":   round(self.free_energy, 5),
            "phi":           round(self.phi, 4),
            "coherence":     round(self.coherence, 4),
            "understanding": round(self.understanding, 4),
            "surprise_z":    round(self.surprise_z, 2),
            "is_surprised":  self.is_surprised,
            "warming_up":    self.warming_up,
            "spatial_read":  self.spatial_read,
            "read_conf":     round(self.read_conf, 2),
        }


# ── public cognition wrapper ──────────────────────────────────
class MatchCognition:
    """Owns the SingularityEngine (slow track, builds the model) and the
    RealtimeReasoner (fast track, catches model breaks).  One frame in,
    one CognitionTick out."""

    def __init__(self, window_size: int = 64,
                 surprise_threshold: float = 2.0,
                 refractory: int = 8,
                 team_names: Optional[Dict[int, str]] = None):
        self.engine = SingularityEngine(ephemeral=True)
        self.window_size = window_size
        self.surprise_threshold = surprise_threshold
        self.refractory = refractory
        self.team_names = team_names or {}
        self.reasoner: Optional[RealtimeReasoner] = None
        self._n = 0
        self._recon_peak = 1e-9
        self._last_break = -10 ** 9
        self._prev_grid: Optional[np.ndarray] = None
        self.trace: List[CognitionTick] = []

    def observe(self, raw_obs: np.ndarray, frame_idx: int,
                grid_abc: Optional[np.ndarray] = None) -> CognitionTick:
        self._n += 1
        raw_obs = np.asarray(raw_obs, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(raw_obs))
        obs_n = raw_obs / norm if norm > 0 else raw_obs

        # Slow track: free-energy minimisation
        out = self.engine.observe_vector(raw_obs, kind="match")
        F     = float(out["free_energy"])
        recon = float(out.get("recon_error", F))
        phi   = float(out["phi"])
        coh   = float(out["coherence"])
        self._recon_peak = max(self._recon_peak, recon)
        understanding = float(np.clip(1.0 - recon / self._recon_peak, 0.0, 1.0))

        # Fast track: surprise on the raw perception
        if self.reasoner is None:
            self.reasoner = RealtimeReasoner(
                state_dim=obs_n.size,
                window_size=self.window_size,
                surprise_threshold=self.surprise_threshold,
                reduce_dim=64,
            )
        tick = self.reasoner.push(obs_n, t=float(frame_idx))
        warming = self._n <= self.window_size
        surprise_z = float(tick.surprise_z)
        raw_surprised = bool(tick.is_surprised) and not warming
        # Refractory: collapse sustained surprise into discrete events
        is_surprised = (raw_surprised and
                        (frame_idx - self._last_break) >= self.refractory)
        if is_surprised:
            self._last_break = frame_idx

        # Spatial correlation read on surprise
        spatial_read, read_conf = "", 0.0
        if is_surprised and grid_abc is not None and self._prev_grid is not None:
            spatial_read, read_conf = correlate_surprise(self._prev_grid, grid_abc)
            if 0 in self.team_names:
                spatial_read = spatial_read.replace("Team A", self.team_names[0])
            if 1 in self.team_names:
                spatial_read = spatial_read.replace("Team B", self.team_names[1])
        if grid_abc is not None:
            self._prev_grid = grid_abc.copy()

        ct = CognitionTick(
            frame_idx=frame_idx, recon_error=recon, free_energy=F,
            phi=phi, coherence=coh,
            understanding=understanding, surprise_z=surprise_z,
            is_surprised=is_surprised, warming_up=warming,
            spatial_read=spatial_read, read_conf=read_conf,
        )
        self.trace.append(ct)
        return ct

    def summary(self) -> Dict:
        if not self.trace:
            return {"frames": 0}
        post = [t for t in self.trace if not t.warming_up]
        rec = [t.recon_error for t in self.trace]
        blk = max(1, len(rec) // 10)
        surps = [t.surprise_z for t in post]
        return {
            "frames_observed":      len(self.trace),
            "frames_post_warmup":   len(post),
            "recon_error_initial":  round(float(np.mean(rec[:blk])), 6),
            "recon_error_final":    round(float(np.mean(rec[-blk:])), 6),
            "recon_error_min":      round(float(min(rec)), 6),
            "understanding_final":  round(self.trace[-1].understanding, 4),
            "free_energy_final":    round(self.trace[-1].free_energy, 5),
            "phi_final":            round(self.trace[-1].phi, 4),
            "mean_surprise_z":      round(float(np.mean(surps)), 2) if surps else 0.0,
            "max_surprise_z":       round(float(np.max(surps)), 2) if surps else 0.0,
            "n_model_breaks":       int(sum(t.is_surprised for t in post)),
            "interpretation": (
                "recon_error_initial -> final = how well the engine predicts "
                "what it sees as it watches (lower = better, drift-free); "
                "n_model_breaks counts frames where the match-model broke."
            ),
        }
