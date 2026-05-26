"""sam3_detector — thin wrapper around Meta's SAM 3 for the FutBotMX pipeline.

Two operating modes:

  • IMAGE mode (`Sam3ImageDetector`)
        per-frame inference with text + box prompts.  No temporal state.
        Used for the SEGMENTATION stage of the pipeline and for the
        cold-start of each video shot.

  • VIDEO mode (`Sam3VideoTracker`)
        wraps `build_sam3_video_predictor` — exploits SAM 3's native
        mask-propagation across frames.  Used for STABLE TRACKING of
        the field + robots + ball through a whole shot.

Both modes share a single `Sam3Backend` that loads the model exactly
once and exposes both predictors through it (saves ~10s of cold start
per mode switch and keeps VRAM headroom).

Dtype:  SAM 3's vitdet trunk ships in bfloat16; we wrap every forward
in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` so the
input activations match weight dtype across the whole graph.

Hardware target: 8.6 GB VRAM (RTX 4060 Laptop).  Peak observed during
single-image text-prompt inference: 5.34 GB.  Leaves ~3 GB for activations
during fine-tune-LoRA or for batched frames.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── lightweight types ─────────────────────────────────────────────
@dataclass
class Detection:
    """One segmented instance from a single frame."""
    class_name: str               # 'field', 'robot_blue', 'robot_red', 'ball', …
    mask: np.ndarray              # bool, (H, W)
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    score: float
    prompt_source: str = "text"   # 'text' | 'box' | 'mask' | 'five_axis'

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1) * (y2 - y1))

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class Sam3Config:
    device: str = "cuda"
    autocast_dtype: str = "bfloat16"          # 'bfloat16' | 'float16' | 'float32'
    # Per-class score floors.  The "ball" class is the tiny / hard case
    # (same lesson as our sports_vision: small object → looser threshold,
    # rely on a tracker to clean up false positives).
    default_score_threshold: float = 0.30
    ball_score_threshold: float = 0.20
    field_score_threshold: float = 0.40
    # Speed knobs
    compile_model: bool = False               # torch.compile — slow first run
    keep_top_k_per_class: int = 12            # cap clutter before NMS


# Class prompts for robot soccer.  Open-vocabulary, English first; SAM 3
# supports multilingual but English is best-tested.  The colour split is
# the cheapest way to separate teams; richer prompts (jersey colour,
# uniform pattern) plug in via `prompts.py` (5-axis perception).
DEFAULT_CLASS_PROMPTS: Dict[str, str] = {
    "field":      "soccer field, green carpet playing area",
    "robot_blue": "blue robot, robot with blue marker",
    "robot_red":  "red robot, robot with red marker",
    "ball":       "orange soccer ball, small round ball on the field",
}


# ── backend singleton ─────────────────────────────────────────────
class Sam3Backend:
    """Lazily loads the SAM 3 model exactly once and shares it across
    image / video predictors."""

    _instance: Optional["Sam3Backend"] = None

    def __init__(self, cfg: Sam3Config):
        self.cfg = cfg
        self._image_model = None
        self._image_processor = None
        self._video_predictor = None
        self._torch = None
        self._loaded_at: Optional[float] = None

    @classmethod
    def get(cls, cfg: Optional[Sam3Config] = None) -> "Sam3Backend":
        if cls._instance is None:
            cls._instance = cls(cfg or Sam3Config())
        return cls._instance

    def _ensure_torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def _autocast_dtype(self):
        torch = self._ensure_torch()
        return {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }[self.cfg.autocast_dtype]

    # ── image predictor ──────────────────────────────────────────
    def get_image_predictor(self):
        if self._image_model is None:
            t0 = time.time()
            print(f"[sam3] loading image model on {self.cfg.device} …",
                  file=sys.stderr)
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self._image_model = build_sam3_image_model(
                device=self.cfg.device,
                load_from_HF=True,
                compile=self.cfg.compile_model,
            )
            self._image_processor = Sam3Processor(self._image_model)
            self._loaded_at = time.time()
            print(f"[sam3] image model loaded in "
                  f"{self._loaded_at - t0:.1f}s", file=sys.stderr)
        return self._image_model, self._image_processor

    # ── video predictor ──────────────────────────────────────────
    def get_video_predictor(self):
        if self._video_predictor is None:
            t0 = time.time()
            print(f"[sam3] loading video predictor on {self.cfg.device} …",
                  file=sys.stderr)
            from sam3.model_builder import build_sam3_video_predictor
            self._video_predictor = build_sam3_video_predictor()
            print(f"[sam3] video predictor loaded in "
                  f"{time.time() - t0:.1f}s", file=sys.stderr)
        return self._video_predictor


# ── image detector (per-frame, cold-start of every shot) ─────────
class Sam3ImageDetector:
    """Wraps SAM 3's image model.  One instance per pipeline run; the
    underlying weights are shared via `Sam3Backend`.

    Typical usage:

        det = Sam3ImageDetector()
        dets = det.detect(frame_rgb,
                          class_prompts={'ball': 'orange ball',
                                         'robot_blue': 'blue robot'})
        for d in dets:
            print(d.class_name, d.score, d.bbox)
    """

    def __init__(self, cfg: Optional[Sam3Config] = None,
                 class_prompts: Optional[Dict[str, str]] = None):
        self.cfg = cfg or Sam3Config()
        self.backend = Sam3Backend.get(self.cfg)
        self.class_prompts = dict(class_prompts or DEFAULT_CLASS_PROMPTS)

    def _score_floor(self, class_name: str) -> float:
        return {
            "ball":  self.cfg.ball_score_threshold,
            "field": self.cfg.field_score_threshold,
        }.get(class_name, self.cfg.default_score_threshold)

    # ── core detection ───────────────────────────────────────────
    def detect(self, image_rgb: np.ndarray,
               class_prompts: Optional[Dict[str, str]] = None,
               box_prompts: Optional[Dict[str, Sequence[Tuple[float, float, float, float]]]] = None,
               ) -> List[Detection]:
        """Run SAM 3 once per class prompt and return all detections.

        `image_rgb` is an (H, W, 3) uint8 numpy array (cv2 frames are BGR
        — convert with cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) first).

        `box_prompts` is optional — when supplied, box seeds are passed
        to the same prompt round to bias the open-vocabulary mask.
        """
        from PIL import Image as PILImage
        torch = self.backend._ensure_torch()

        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        pil = PILImage.fromarray(image_rgb)

        _, processor = self.backend.get_image_predictor()
        autocast_dtype = self.backend._autocast_dtype()

        prompts = dict(self.class_prompts)
        if class_prompts:
            prompts.update(class_prompts)
        box_prompts = box_prompts or {}

        all_dets: List[Detection] = []
        with torch.inference_mode(), \
             torch.autocast(device_type=self.cfg.device,
                            dtype=autocast_dtype):
            state = processor.set_image(pil)
            for class_name, text in prompts.items():
                # ── text prompt round ──────────────────────────
                out = processor.set_text_prompt(state=state, prompt=text)
                masks  = out.get("masks")
                boxes  = out.get("boxes")
                scores = out.get("scores")
                if masks is None or len(masks) == 0:
                    continue
                masks_np  = masks.detach().cpu().numpy()
                boxes_np  = boxes.detach().cpu().numpy() if boxes is not None else None
                scores_np = scores.detach().cpu().numpy() if scores is not None else None

                threshold = self._score_floor(class_name)
                keep_top_k = self.cfg.keep_top_k_per_class
                idxs = list(range(len(masks_np)))
                if scores_np is not None:
                    idxs.sort(key=lambda i: -float(scores_np[i]))
                    idxs = idxs[:keep_top_k]

                for i in idxs:
                    s = float(scores_np[i]) if scores_np is not None else 0.0
                    if s < threshold:
                        continue
                    mask = np.asarray(masks_np[i]).astype(bool)
                    # masks_np can be (N, 1, H, W) or (N, H, W); squeeze
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
                    bbox = (
                        float(boxes_np[i][0]), float(boxes_np[i][1]),
                        float(boxes_np[i][2]), float(boxes_np[i][3]),
                    ) if boxes_np is not None else _bbox_from_mask(mask)
                    all_dets.append(Detection(
                        class_name=class_name, mask=mask, bbox=bbox,
                        score=s, prompt_source="text",
                    ))
        return all_dets

    # ── convenience: just for one class ──────────────────────────
    def detect_class(self, image_rgb: np.ndarray, class_name: str,
                     text_prompt: Optional[str] = None) -> List[Detection]:
        prompt = text_prompt or self.class_prompts.get(class_name, class_name)
        return self.detect(image_rgb, class_prompts={class_name: prompt})


# ── small helpers ────────────────────────────────────────────────
def _bbox_from_mask(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()),
            float(xs.max() + 1), float(ys.max() + 1))


def detections_to_supervision(dets: List[Detection]):
    """Bridge to roboflow `supervision.Detections` for drawing / tracking.
    Imported lazily so the module is usable without supervision installed."""
    import supervision as sv
    if not dets:
        return sv.Detections.empty()
    xyxy   = np.array([d.bbox for d in dets], dtype=np.float32)
    scores = np.array([d.score for d in dets], dtype=np.float32)
    # We synthesise a class-id by hashing the class_name; the real
    # mapping lives in `class_names` so palette tools can label them.
    names  = sorted({d.class_name for d in dets})
    name2id = {n: i for i, n in enumerate(names)}
    cls    = np.array([name2id[d.class_name] for d in dets], dtype=np.int32)
    masks  = np.stack([d.mask for d in dets], axis=0)
    out = sv.Detections(xyxy=xyxy, confidence=scores,
                         class_id=cls, mask=masks)
    out.data = {"class_name": np.array([d.class_name for d in dets])}
    return out
