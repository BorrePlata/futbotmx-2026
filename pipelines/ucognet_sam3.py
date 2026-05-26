"""ucognet_sam3 — U-CogNet integrated cognitive stack ON TOP of SAM 3.

This is the EXPERIMENTAL arm of the scientific comparison.  Identical
SAM 3 perception to the baseline; the difference is the cognitive
layer that consumes SAM 3 detections and forms ITS OWN representation,
reports free energy F, integrated information φ, reconstruction-error
understanding, surprise spikes, and an honest spatial correlation of
where the perceptual map changed when the model broke.

Architecture (same SAM 3 weights, same prompts, same frames as baseline):

  per-frame SAM 3 masks ──► foot-points / ball position
                                  │
                                  ▼
                       encode_raw_observation
                       (24×16×4 occupancy grid)
                                  │
                                  ▼
                       MatchCognition           ── from experiments/sports_vision
                       (SingularityEngine        — free-energy minimisation
                        + RealtimeReasoner       — surprise on the manifold
                        + spatial correlation)
                                  │
                                  ▼
                       CognitionTick (F, φ, understanding, surprise, …)
                                  │
                                  ▼
                       augmented frame + cognitive HUD overlay

What the U-CogNet arm produces vs the baseline:

  baseline:  per-class counts + scores + latency.
  ucognet :  baseline    PLUS  free-energy F, integrated information φ,
                              reconstruction-error understanding [0,1],
                              surprise z-score, model-break flag,
                              honest spatial read when surprise fires,
                              and a Friston-style temporal trace.

That trace is what `evaluation/cognitive.py` consumes for paper-grade
plots and what the live_demo dual-pane shows alongside the baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
_FUTBOTMX  = _REPO_ROOT / "experiments" / "futbotmx"
DEFAULT_OUT = _FUTBOTMX / "output"

from experiments.futbotmx.pipelines.baseline_sam3 import (
    BaselineSam3Pipeline, DEFAULT_PROMPTS, CLASS_COLOURS_BGR,
    _load_dotenv, build_evidence_manifest, _aggregate_metrics, FrameMetric,
)


# ── lazy imports (cognitive subpackage — fully self-contained) ──
def _import_cognition():
    """Imported lazily to avoid loading numpy/SingularityEngine until the
    pipeline actually runs.  All symbols come from the local
    `experiments/futbotmx/cognitive/` Apache-2.0 subpackage — no
    dependency on any upstream private module."""
    from experiments.futbotmx.cognitive import (
        MatchCognition, encode_raw_observation, occupancy_grid,
        CognitionTick,
    )
    return MatchCognition, encode_raw_observation, occupancy_grid, CognitionTick


# ── foot-point extraction (SAM 3 mask → world-space anchor) ─────
def _foot_point(bbox: Optional[List[float]], mask: np.ndarray
                ) -> Optional[Tuple[float, float]]:
    """Anchor point for a detection in image space.

    For robots and people we use the BOTTOM-CENTRE of the bbox — that
    is the point that touches the ground, which is what an occupancy
    map needs.  For the ball we use the bbox centre (it doesn't
    have a 'foot').  The mask is a fallback when bbox is None.
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, y2)
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return (float(xs.mean()), float(ys.max()))


def _ball_point(bbox: Optional[List[float]], mask: np.ndarray
                ) -> Optional[Tuple[float, float]]:
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return (float(xs.mean()), float(ys.mean()))


def split_teams_by_x(robot_dets: List[Dict], frame_w: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Simple left-half / right-half team split.

    Without per-robot colour tagging (a v0.2 fine-tune deliverable),
    we use position as a proxy: left robots → team A, right → team B.
    This is honest about its limitation — it is a heuristic for the
    DEMO, and the cognitive layer's outputs (F, φ, surprise) are
    largely invariant to the team label, since they read STRUCTURE in
    the occupancy map, not identity.
    """
    A: List[Tuple[float, float]] = []
    B: List[Tuple[float, float]] = []
    mid = frame_w * 0.5
    for d in robot_dets:
        pt = _foot_point(d.get("bbox"), d.get("mask"))
        if pt is None:
            continue
        (A if pt[0] < mid else B).append(pt)
    return (np.array(A, dtype=np.float64).reshape(-1, 2),
            np.array(B, dtype=np.float64).reshape(-1, 2))


def hand_points(hand_dets: List[Dict]) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    for d in hand_dets:
        pt = _foot_point(d.get("bbox"), d.get("mask"))
        if pt is not None:
            pts.append(pt)
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def ball_xy(ball_dets: List[Dict]) -> Optional[Tuple[float, float]]:
    if not ball_dets:
        return None
    best = max(ball_dets, key=lambda d: d["score"])
    return _ball_point(best.get("bbox"), best.get("mask"))


def field_coverage(field_dets: List[Dict], frame_shape: Tuple[int, int]) -> float:
    if not field_dets:
        return 0.0
    h, w = frame_shape
    total = float(h * w)
    return float(max(d["mask"].sum() / total for d in field_dets))


# ── extended per-frame record ───────────────────────────────────
@dataclass
class UCogNetFrameMetric(FrameMetric):
    free_energy:    float = 0.0
    phi:            float = 0.0
    recon_error:    float = 0.0
    understanding:  float = 0.0
    coherence:      float = 0.0
    surprise_z:     float = 0.0
    is_surprised:   bool  = False
    warming_up:     bool  = True
    spatial_read:   str   = ""
    cognitive_ms:   float = 0.0
    # Calibrated score from the safety layer (when wired).  For v0.1 we
    # use a simple max(0, score - epistemic_uncertainty) — the
    # full AnankeShield + temperature-scaling lives in `evaluation/`.
    calibrated_top_score: Dict[str, float] = field(default_factory=dict)


# ── ucognet pipeline ────────────────────────────────────────────
class UCogNetSam3Pipeline(BaselineSam3Pipeline):
    """SAM 3 perception + post-humanist cognitive stack on top."""

    def __init__(self, *, surprise_threshold: float = 2.0,
                 refractory: int = 8, team_split: str = "left_right",
                 **base_kwargs):
        super().__init__(**base_kwargs)
        self.surprise_threshold = surprise_threshold
        self.refractory = refractory
        self.team_split = team_split
        self._cog = None         # MatchCognition (lazy)
        self._enc = None         # encode_raw_observation
        self._grid = None        # occupancy_grid
        self._CognitionTick = None
        self._traces = None      # TraceBuffers — created lazily

    def _ensure_cognition(self):
        if self._cog is not None:
            return
        MC, enc, grid, CT = _import_cognition()
        self._cog = MC(window_size=64,
                      surprise_threshold=self.surprise_threshold,
                      refractory=self.refractory,
                      team_names={0: "Team Left", 1: "Team Right"})
        self._enc = enc
        self._grid = grid
        self._CognitionTick = CT

    # ── one frame: perception → cognition ───────────────────────
    def cognize_frame(self, frame_bgr: np.ndarray,
                      per_class: Dict[str, List[Dict]],
                      frame_idx: int) -> Tuple[Dict, float]:
        """Returns (cognition_dict, cognitive_ms).

        cognition_dict has F, φ, recon, surprise, etc — flat schema."""
        self._ensure_cognition()
        H, W = frame_bgr.shape[:2]

        robots = per_class.get("robot", [])
        hands  = per_class.get("hand", [])
        balls  = per_class.get("ball", [])
        fields = per_class.get("field", [])

        team_a, team_b = split_teams_by_x(robots, W)
        others = hand_points(hands)
        ball_pt = ball_xy(balls)
        coverage = field_coverage(fields, (H, W))

        t0 = time.time()
        raw_obs = self._enc(
            frame_shape=(H, W),
            team_a_pts=team_a, team_b_pts=team_b,
            other_pts=others, ball_xy=ball_pt,
            pitch_coverage=coverage,
            recenter=False,
        )
        grid_abc = self._grid(team_a, team_b, ball_pt, (H, W))
        tick = self._cog.observe(raw_obs, frame_idx, grid_abc=grid_abc)
        elapsed = (time.time() - t0) * 1000

        return tick.to_dict(), elapsed

    # ── augmented rendering ─────────────────────────────────────
    def render_frame_ucognet(self, frame_bgr: np.ndarray,
                             per_class: Dict[str, List[Dict]],
                             cognition: Dict,
                             hud_lines: Optional[List[str]] = None) -> np.ndarray:
        """Like baseline render but with the cognitive HUD + a surprise
        spike indicator when the model breaks."""
        import cv2
        vis = self.render_frame(frame_bgr, per_class, hud_lines=None)

        # Overwrite the baseline watermark with U-CogNet branding
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, h - 28), (260, h), (0, 0, 0), -1)
        cv2.putText(vis, "U-CogNet over SAM 3",
                    (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (100, 240, 170), 1, cv2.LINE_AA)

        # Cognitive HUD — bottom-right
        F   = cognition.get("free_energy", 0.0)
        phi = cognition.get("phi", 0.0)
        rec = cognition.get("recon_error", 0.0)
        und = cognition.get("understanding", 0.0)
        sz  = cognition.get("surprise_z", 0.0)
        warming = cognition.get("warming_up", False)
        spike = cognition.get("is_surprised", False)

        bx, by = w - 280, h - 110
        ov = vis.copy()
        cv2.rectangle(ov, (bx, by), (w - 8, h - 8), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, vis, 0.4, 0, dst=vis)
        cv2.putText(vis, "COGNITIVE LAYER", (bx + 8, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 240, 170), 1,
                    cv2.LINE_AA)
        lines = [
            f"F  = {F:+.3f}",
            f"phi= {phi:.3f}",
            f"recon = {rec:.3f}",
            f"understanding = {und*100:.0f}%",
            f"surprise z = {sz:+.2f}{'  WARMUP' if warming else ''}",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(vis, ln, (bx + 8, by + 36 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 240, 255), 1,
                        cv2.LINE_AA)

        # Surprise spike indicator — flashes red border for one frame
        if spike:
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (50, 50, 240), 8)
            cv2.putText(vis, "! MODEL BREAK !",
                        (w // 2 - 130, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (50, 50, 240), 2, cv2.LINE_AA)
            sr = cognition.get("spatial_read", "")
            if sr:
                cv2.putText(vis, sr, (10, h - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (250, 250, 250), 1, cv2.LINE_AA)

        if hud_lines:
            self._draw_hud(vis, hud_lines)
        return vis

    # ── whole-video run ─────────────────────────────────────────
    def run_video(self, video_path: Path, out_dir: Path,
                  max_frames: Optional[int] = None,
                  output_fps: Optional[float] = None,
                  aesthetic: bool = False,
                  panel_w: int = 1280, panel_h: int = 720) -> Dict:
        import cv2
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = video_path.stem
        suffix = "ucognet_aesthetic" if aesthetic else "ucognet"
        out_video_path    = out_dir / f"{stem}_{suffix}.mp4"
        out_metrics_path  = out_dir / f"{stem}_{suffix}.metrics.json"
        out_manifest_path = out_dir / f"{stem}_{suffix}.manifest.json"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 cannot open {video_path}")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nf      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        scale   = min(1.0, self.max_side / max(W, H))
        vid_W, vid_H = int(W * scale), int(H * scale)
        fps_out = output_fps or fps_src

        if aesthetic:
            out_W, out_H = panel_w, panel_h
            from experiments.futbotmx.viz.aesthetic import (
                AestheticCompositor, SidebarState, TraceBuffers,
            )
            self._aesthetic = AestheticCompositor(panel_w=panel_w,
                                                    panel_h=panel_h)
            self._traces = TraceBuffers(window=120)
            self._SidebarState = SidebarState
        else:
            out_W, out_H = vid_W, vid_H

        print(f"[ucognet] video: {video_path.name}  {W}x{H} @ {fps_src:.1f}fps  "
              f"({nf} frames) → vid {vid_W}x{vid_H}{' · panel ' + str(panel_w) + 'x' + str(panel_h) if aesthetic else ''}",
              file=sys.stderr)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc,
                                  fps_out, (out_W, out_H))

        self._ensure_model()
        self._ensure_cognition()

        per_frame: List[Dict] = []
        n_written = 0
        t_start = time.time()
        try:
            for idx in range(nf):
                ok, frame = cap.read()
                if not ok:
                    break
                if scale < 1.0:
                    frame = cv2.resize(frame, (vid_W, vid_H),
                                        interpolation=cv2.INTER_AREA)

                per_class, infer_ms = self.detect_frame(frame)
                cog, cog_ms = self.cognize_frame(frame, per_class, idx)

                t_render = time.time()
                if aesthetic:
                    clean = self.render_clean(frame, per_class)
                    self._traces.push(cog["free_energy"], cog["phi"],
                                       cog["surprise_z"])
                    Fl, Pl, Sl = self._traces.as_lists()
                    state = self._SidebarState(
                        arm_label="U-CogNet   |   SAM 3 + cognitive stack",
                        arm_colour=(170, 235, 110),
                        frame_idx=idx, total_frames=nf,
                        infer_ms=infer_ms, cognitive_ms=cog_ms,
                        per_class={c: {"count": len(d),
                                        "top": max((dd["score"] for dd in d), default=0.0)}
                                    for c, d in per_class.items() if d},
                        free_energy=cog["free_energy"],
                        phi=cog["phi"],
                        recon_error=cog["recon_error"],
                        understanding=cog["understanding"],
                        surprise_z=cog["surprise_z"],
                        warming_up=bool(cog.get("warming_up", False)),
                        model_break=bool(cog.get("is_surprised", False)),
                        spatial_read=cog.get("spatial_read", ""),
                        F_trace=Fl, phi_trace=Pl, surprise_trace=Sl,
                    )
                    vis = self._aesthetic.render(clean, state)
                else:
                    hud = [
                        f"frame {idx+1}/{nf}  ({(idx+1)/max(nf,1)*100:.0f}%)",
                        f"SAM3 {infer_ms:.0f} ms  + cog {cog_ms:.0f} ms",
                    ]
                    for c, dets in per_class.items():
                        if dets:
                            top = max(d["score"] for d in dets)
                            hud.append(f"  {c}: {len(dets)} (top {top:.2f})")
                    vis = self.render_frame_ucognet(frame, per_class, cog,
                                                     hud_lines=hud)
                render_ms = (time.time() - t_render) * 1000
                writer.write(vis)
                n_written += 1

                counts = {c: len(d) for c, d in per_class.items()}
                tops   = {c: max((dd["score"] for dd in d), default=0.0)
                          for c, d in per_class.items()}
                means  = {c: (float(np.mean([dd["score"] for dd in d])) if d else 0.0)
                          for c, d in per_class.items()}
                per_frame.append(asdict(UCogNetFrameMetric(
                    frame_idx=idx, timestamp_s=round(idx / fps_src, 4),
                    per_class_counts=counts,
                    per_class_top_score=tops,
                    per_class_mean_score=means,
                    infer_ms=round(infer_ms, 2),
                    render_ms=round(render_ms, 2),
                    free_energy=cog["free_energy"],
                    phi=cog["phi"],
                    recon_error=cog["recon_error"],
                    understanding=cog["understanding"],
                    coherence=cog["coherence"],
                    surprise_z=cog["surprise_z"],
                    is_surprised=bool(cog.get("is_surprised", False)),
                    warming_up=bool(cog.get("warming_up", False)),
                    spatial_read=cog.get("spatial_read", ""),
                    cognitive_ms=round(cog_ms, 2),
                )))

                if max_frames and n_written >= max_frames:
                    break
        finally:
            cap.release()
            writer.release()

        total_s = time.time() - t_start
        agg = _aggregate_metrics(per_frame)
        agg["video_path"]     = str(video_path)
        agg["output_video"]   = str(out_video_path)
        agg["frames_written"] = n_written
        agg["wall_seconds"]   = round(total_s, 2)
        agg["wall_fps"]       = round(n_written / max(total_s, 1e-3), 2)
        # Cognitive aggregate
        cog_summary = self._cog.summary()
        agg["cognitive"] = cog_summary
        agg["cognitive_latency_ms"] = {
            "mean": round(float(np.mean([r["cognitive_ms"] for r in per_frame])), 2),
            "p95":  round(float(np.percentile(
                [r["cognitive_ms"] for r in per_frame], 95)), 2),
        }
        out_metrics_path.write_text(json.dumps(
            {"summary": agg, "per_frame": per_frame},
            indent=2, ensure_ascii=False
        ), encoding="utf-8")

        manifest = build_evidence_manifest(
            video=video_path, output_video=out_video_path,
            metrics_path=out_metrics_path, prompts=self.prompts,
            args_dict={"device": self.device, "score_min": self.score_min,
                       "max_side": self.max_side,
                       "max_frames": max_frames,
                       "surprise_threshold": self.surprise_threshold,
                       "refractory": self.refractory,
                       "team_split": self.team_split},
            total_frames=n_written, total_seconds=total_s,
        )
        manifest["arm"]    = "ucognet_sam3"
        manifest["schema"] = "futbotmx.ucognet.v1"
        manifest["cognitive_modules"] = [
            "SingularityEngine (free-energy + phi)",
            "RealtimeReasoner (surprise on manifold)",
            "spatial correlation read",
        ]
        out_manifest_path.write_text(json.dumps(manifest, indent=2),
                                      encoding="utf-8")

        print(f"\n[ucognet] ✅ {n_written} frames in {total_s:.1f}s "
              f"({n_written/max(total_s,1e-3):.2f} fps)", file=sys.stderr)
        print(f"[ucognet]   video    : {out_video_path}", file=sys.stderr)
        print(f"[ucognet]   metrics  : {out_metrics_path}", file=sys.stderr)
        print(f"[ucognet]   manifest : {out_manifest_path}", file=sys.stderr)
        print(f"[ucognet]   cog summary: F={cog_summary.get('free_energy_final')}  "
              f"phi={cog_summary.get('phi_final')}  "
              f"breaks={cog_summary.get('n_model_breaks')}", file=sys.stderr)
        return agg


def main() -> int:
    ap = argparse.ArgumentParser(description="U-CogNet on SAM 3 (cognitive arm)")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=720)
    ap.add_argument("--score-min", type=float, default=0.30)
    ap.add_argument("--surprise-threshold", type=float, default=2.0)
    ap.add_argument("--refractory", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aesthetic", action="store_true",
                    help="Paper-grade composition: clean video + sidebar + sparklines")
    ap.add_argument("--panel-w", type=int, default=1280)
    ap.add_argument("--panel-h", type=int, default=720)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    _load_dotenv(_REPO_ROOT / ".env")
    os.environ.setdefault("HF_HOME", "F:/U-CogNet-ToGo/sam3")

    pipe = UCogNetSam3Pipeline(
        device=args.device, score_min=args.score_min, max_side=args.max_side,
        surprise_threshold=args.surprise_threshold, refractory=args.refractory,
    )
    summary = pipe.run_video(args.video, args.out_dir, max_frames=args.max_frames,
                              aesthetic=args.aesthetic,
                              panel_w=args.panel_w, panel_h=args.panel_h)

    print("\n[ucognet] summary →")
    print(json.dumps(summary, indent=2))

    suffix = "ucognet_aesthetic" if args.aesthetic else "ucognet"
    out_video = args.out_dir / f"{args.video.stem}_{suffix}.mp4"
    if args.open and out_video.exists():
        if sys.platform == "win32":
            os.startfile(out_video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
