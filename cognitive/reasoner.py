"""reasoner — streaming geometric surprise via Mahalanobis distance.

Maintains a rolling buffer of observations and emits a surprise z-score
per new observation: distance from the running mean rescaled by the
shrunk inverse covariance.  When the score exceeds `surprise_threshold`
the tick is flagged `is_surprised`; the run-length of surprise events
is used downstream to detect 'model breaks' (sustained surprise =
the cognitive model failed).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Sequence

import numpy as np


@dataclass
class ReasoningTick:
    t:             float
    z_norm:        float           # L2 norm of the projected observation
    surprise_z:    float           # current z-score (sigma units)
    is_surprised:  bool            # > surprise_threshold
    n_pushes:      int             # cumulative pushes so far


def _shrunk_covariance(X: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage toward mu*I — keeps the metric well-
    conditioned on small rolling windows."""
    cov = np.cov(X, rowvar=False, ddof=1)
    mu_diag = float(np.mean(np.diag(cov)))
    target = mu_diag * np.eye(cov.shape[0], dtype=cov.dtype)
    return (1.0 - alpha) * cov + alpha * target


class RealtimeReasoner:
    """Streaming geometric reasoner — fast track of the two-speed cognitive
    system (Singularity = slow understanding, Reasoner = fast surprise)."""

    def __init__(self, state_dim: int,
                 window_size: int = 64,
                 surprise_threshold: float = 2.0,
                 reduce_dim: Optional[int] = None,
                 shrink_alpha: float = 0.10,
                 regularize: float = 1e-3,
                 seed: int = 42):
        self.state_dim          = int(state_dim)
        self.window_size        = max(8, int(window_size))
        self.surprise_threshold = float(surprise_threshold)
        self._shrink            = float(np.clip(shrink_alpha, 0.0, 1.0))
        self._reg               = float(regularize)

        rng = np.random.RandomState(int(seed))
        if reduce_dim is not None and 0 < reduce_dim < state_dim:
            self._proj = (rng.randn(state_dim, reduce_dim).astype(np.float64)
                          / np.sqrt(state_dim))
            self._eff_dim = int(reduce_dim)
        else:
            self._proj = None
            self._eff_dim = self.state_dim

        self._buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self._n = 0

    def _project(self, v: np.ndarray) -> np.ndarray:
        return v @ self._proj if self._proj is not None else v

    def push(self, z: np.ndarray, t: float = 0.0) -> ReasoningTick:
        """Push one observation, return the tick with its surprise z-score."""
        self._n += 1
        z = np.asarray(z, dtype=np.float64).ravel()
        if z.shape[0] != self.state_dim:
            # tolerate small shape drifts via truncation/padding
            if z.shape[0] > self.state_dim:
                z = z[:self.state_dim]
            else:
                pad = np.zeros(self.state_dim - z.shape[0], dtype=np.float64)
                z = np.concatenate([z, pad])
        zp = self._project(z)
        self._buffer.append(zp)
        z_norm = float(np.linalg.norm(zp))

        # Need at least 8 samples before the covariance is meaningful
        if len(self._buffer) < 8:
            return ReasoningTick(t=t, z_norm=z_norm, surprise_z=0.0,
                                  is_surprised=False, n_pushes=self._n)

        X = np.stack(self._buffer)
        mean = X.mean(axis=0)
        S = _shrunk_covariance(X, alpha=self._shrink)
        S = S + self._reg * np.eye(S.shape[0], dtype=S.dtype)
        # Mahalanobis distance via Cholesky solve
        try:
            L = np.linalg.cholesky(S)
            diff = (zp - mean).reshape(-1, 1)
            y = np.linalg.solve(L, diff)
            d2 = float((y * y).sum())
            d  = float(np.sqrt(max(d2, 0.0)))
            # convert to "z-score" relative to chi-square expected value
            # under H0; approximate via the eff_dim sqrt
            sz = (d - np.sqrt(self._eff_dim)) / max(np.sqrt(2 * self._eff_dim), 1.0)
        except np.linalg.LinAlgError:
            sz = 0.0

        is_surprised = bool(sz > self.surprise_threshold)
        return ReasoningTick(t=t, z_norm=z_norm,
                              surprise_z=float(sz),
                              is_surprised=is_surprised,
                              n_pushes=self._n)
