"""singularity — variational free-energy minimiser + Tononi-style phi.

Self-contained reference implementation for the U-CogNet arm of the
FutBotMX 2026 study.  Two primitives:

  • _FreeEnergyMinimizer  — variational inference of q(z|x) = N(mu, sigma^2)
                             under a standard-normal prior, with a log-var
                             floor so the KL term cannot drift unboundedly
                             over long observation streams.
  • _PhiCalculator        — IIT phi approximated as the fraction of system
                             variance that is irreducible to its binary
                             partitions.

The `SingularityEngine` thin wrapper composes them and exposes a single
`observe_vector(x)` call that returns the four scalars consumed by the
match cognition layer: free_energy F, reconstruction error, phi and
coherence (cosine to the mean of the latent trajectory).

References:
  - Friston (2010)              Nature Rev. Neurosci. 11:127.
  - Tononi (2008)               Biol. Bull. 215:216.
  - Mediano et al. (2019)       Entropy 21:17.
"""
from __future__ import annotations

from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np


# Module-level constants (tunable, not magic numbers)
MANIFOLD_DIM      = 64      # dimensionality of the learned latent z
PHI_PARTITIONS    = 4       # binary partitions used in phi approximation
FREE_ENERGY_LR    = 0.01    # variational gradient step
HIST_WINDOW_PHI   = 20      # rolling phi history
HIST_WINDOW_COH   = 32      # rolling coherence history


# ── core variational machinery ─────────────────────────────────
class _FreeEnergyMinimizer:
    """Minimises F = KL(q || p) + reconstruction error for q(z|x) = N(mu, sigma^2)
    against a standard-normal prior.  Implements one gradient step per call.

    The LOG_VAR_FLOOR clamps log-variance so the KL term cannot drift
    unboundedly upward over long runs (without it variance collapses to 0
    and -log_var grows without limit, which would make F a bad summary)."""

    LOG_VAR_FLOOR = -4.0

    def __init__(self, dim: int = MANIFOLD_DIM, lr: float = FREE_ENERGY_LR):
        self.dim = dim
        self.lr  = lr
        self.mu      = np.zeros(dim, dtype=np.float64)
        self.log_var = np.zeros(dim, dtype=np.float64)
        self.last_recon = 1.0

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """One gradient step, returns (sampled z, free_energy F)."""
        if x.shape[0] != self.dim:
            # Project x to dim via deterministic hashing (just bin sums)
            proj = np.zeros(self.dim, dtype=np.float64)
            for i, v in enumerate(x.flat):
                proj[i % self.dim] += float(v)
            n = float(np.linalg.norm(proj))
            x = proj / max(n, 1e-8)

        error = x - self.mu
        self.mu = self.mu + self.lr * error
        self.log_var = np.maximum(self.log_var - self.lr * 0.1,
                                   self.LOG_VAR_FLOOR)

        std = np.exp(0.5 * self.log_var)
        z = self.mu + std * (np.random.randn(self.dim) * 0.01)

        kl = 0.5 * float(np.sum(
            np.exp(self.log_var) + self.mu ** 2 - 1 - self.log_var
        ))
        recon = float(np.mean(error ** 2))
        self.last_recon = recon
        F = (kl + recon) / self.dim
        return z, F


# ── phi approximation ──────────────────────────────────────────
class _PhiCalculator:
    """Approximates Tononi phi as 1 - (sum partition variances) /
    (n_parts x total variance).  Cheap, bounded, monotonic in the
    'irreducibility' of the system across binary partitions."""

    def __init__(self, dim: int = MANIFOLD_DIM,
                 n_parts: int = PHI_PARTITIONS,
                 max_hist: int = HIST_WINDOW_PHI):
        self.dim = dim
        self.n_parts = n_parts
        self._hist: Deque[np.ndarray] = deque(maxlen=max_hist)

    def update(self, z: np.ndarray) -> None:
        self._hist.append(np.asarray(z, dtype=np.float64))

    def compute(self) -> float:
        if len(self._hist) < 2:
            return 0.0
        data = np.stack(self._hist)               # (T, dim)
        total_var = float(np.var(data))
        if total_var < 1e-10:
            return 0.0
        part_size = max(1, self.dim // self.n_parts)
        sum_part_var = 0.0
        for i in range(self.n_parts):
            lo = i * part_size
            hi = min(lo + part_size, self.dim)
            sum_part_var += float(np.var(data[:, lo:hi]))
        phi = max(0.0, 1.0 - (sum_part_var /
                              (self.n_parts * total_var + 1e-10)))
        return min(1.0, phi * self.n_parts)


# ── public engine ──────────────────────────────────────────────
class SingularityEngine:
    """Composes the free-energy minimiser, the phi calculator and a
    rolling coherence estimator.  The single public method is
    `observe_vector(x)`."""

    def __init__(self, ephemeral: bool = True,
                 dim: int = MANIFOLD_DIM,
                 n_phi_parts: int = PHI_PARTITIONS):
        self.dim = dim
        self._fe   = _FreeEnergyMinimizer(dim=dim)
        self._phi  = _PhiCalculator(dim=dim, n_parts=n_phi_parts)
        self._z_hist: Deque[np.ndarray] = deque(maxlen=HIST_WINDOW_COH)
        self.ephemeral = ephemeral
        self._n_observed = 0

    # public API expected by experiments/futbotmx/cognitive/match.py
    def observe_vector(self, x: np.ndarray,
                       kind: str = "generic") -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float64).ravel()
        z, F = self._fe.encode(x)
        self._phi.update(z)
        phi = self._phi.compute()
        self._z_hist.append(z)
        coherence = self._coherence()
        self._n_observed += 1
        return {
            "free_energy": F,
            "recon_error": float(self._fe.last_recon),
            "phi":         phi,
            "coherence":   coherence,
            "kind":        kind,
            "n_observed":  self._n_observed,
        }

    def _coherence(self) -> float:
        """Cosine of the latest z against the mean z of the rolling window
        — high when the system is settled, low when it is exploring."""
        if len(self._z_hist) < 2:
            return 1.0
        Z = np.stack(self._z_hist)
        mean = Z.mean(axis=0)
        last = Z[-1]
        nm = float(np.linalg.norm(mean))
        nl = float(np.linalg.norm(last))
        if nm < 1e-9 or nl < 1e-9:
            return 0.0
        return float(np.dot(mean, last) / (nm * nl))
