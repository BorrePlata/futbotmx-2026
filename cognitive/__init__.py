"""futbotmx.cognitive — self-contained cognitive primitives for this submission.

A minimal re-implementation of the variational free-energy minimisation,
integrated information (phi) approximation, geometric surprise reasoner
and occupancy-based match-cognition wrapper used by the U-CogNet arm of
this study.  All Apache 2.0 licensed; no dependency on any private
upstream module.

The references for the underlying ideas are listed in `paper/PAPER.md`:
Friston (2010) for free energy, Tononi (2008) and Mediano et al. (2019)
for integrated information, the Sohn et al. (2020) weakly-supervised
tradition for the surrounding evaluation framework.

The full U-CogNet research platform that informs this work is open to
collaborators at https://ucognet.pro.
"""
from .singularity import SingularityEngine, MANIFOLD_DIM, PHI_PARTITIONS
from .reasoner import RealtimeReasoner, ReasoningTick
from .match import (
    MatchCognition, CognitionTick,
    encode_raw_observation, occupancy_grid, correlate_surprise,
    GRID_W, GRID_H,
)

__all__ = [
    "SingularityEngine", "MANIFOLD_DIM", "PHI_PARTITIONS",
    "RealtimeReasoner", "ReasoningTick",
    "MatchCognition", "CognitionTick",
    "encode_raw_observation", "occupancy_grid", "correlate_surprise",
    "GRID_W", "GRID_H",
]
