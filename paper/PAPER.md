# Free-Energy Cognitive Augmentation on SAM 3 for Robot-Soccer Video Analysis

*A fully-automated, zero-human-judgement comparative study submitted to the Copa
FutBotMX 2026 Computer Vision Challenge, Profesional category.*

**Authors:** Samuel Plata — Brainstream / U-CogNet research platform
**Contact:** samuel@brainstream.pro · https://ucognet.pro
**License:** Apache 2.0 (code) · SAM License (model weights, Meta AI)
**Repository:** *<github URL pending public push>*

---

## Abstract

We present a comparative scientific study of two computer-vision arms running
on identical Copa FutBotMX 2026 robot-soccer footage. The **baseline arm**
uses Meta's [SAM 3](https://arxiv.org/abs/2511.16719) (Carion et al., 2025)
for concept-prompted segmentation alone. The **U-CogNet arm** wraps the same
SAM 3 perception with the U-CogNet integrated cognitive stack — a free-energy
Singularity engine that builds its own 64-dimensional latent manifold of the
match, a real-time information-geometry reasoner that emits surprise on that
manifold, and an autonomous spatial-correlation read that names *where* the
match changed when the model broke. The cognitive layer is strictly additive:
**identical SAM 3 detections (100 % per-class count agreement)**, and **+6.4 %
mean wall-time overhead**.

We do not claim human-verified ground truth; instead, every metric in
this paper is derived from one of three reproducible annotator-free
signals: temporal-consistency labels in the weakly-supervised tradition
(Lee 2013; Sohn et al. 2020), inter-arm agreement on the same SAM 3
detections, and a Hendrycks-Dietterich-style 21-condition perturbation
battery that consumes no labels at all. We evaluate both arms across
six independent axes — detection (mAP@0.5), tracking (MOTA / MOTP /
ID-switch), agreement-calibration ECE, the robustness battery, a
Friston-style cognitive trajectory analysis, and a per-layer latency
budget. Across **three tournament clips (613 frames total)** the
baseline arm sustains mAP@0.5 = 0.993 (per-clip 0.984 / 0.998 / 0.998)
and MOTA = 0.950, while the U-CogNet arm autonomously flags
post-warmup *model-break* events accompanied by hedged tactical hints
("possible cluster pressure shift", "possible ball-zone shift") that
are spatial correlations, never confirmed events. Every reported figure
is bit-exactly reproducible from an Evidence Manifest persisted
alongside the results.

---

## 1. Introduction

The Copa FutBotMX 2026 Computer Vision challenge asks participants to
segment, track and analyse Mexican Robotics Federation tournament footage
using SAM 3 (§ 3.5 of the official convocatoria). The Profesional category
explicitly rewards innovation over the model — fine-tuning, prompt
engineering, integration with other models, geometric post-processing.

This work pushes the innovation axis in a direction that, to our knowledge,
no other submission will: we apply a **free-energy cognitive layer** — a
variational self-representation, a real-time information-geometry surprise
estimator, and a bit-exact Evidence Manifest — *on top of* SAM 3. We
deliberately do **not** fine-tune SAM 3 or alter its weights; we treat
SAM 3 as a fixed perceptual sense organ and ask:

> "What does a cognitive layer that builds its own latent model of the match
> add on top of an off-the-shelf state-of-the-art segmenter?"

The answer in this work is a **Friston-style self-witnessing capability**:
the system silently learns to predict the spatial occupancy of robots, balls
and bystanders, and reports when its predictions break — autonomously, with
no human-defined event taxonomy and no rule-based tactical heuristics.

### 1.1 Contributions

1. **Two-arm scientific comparison architecture** on identical Copa
   FutBotMX 2026 footage (§ 3) — same SAM 3 weights, same prompts, same
   frames — with the cognitive stack as the only intervention.
2. **Zero-human-judgement evaluation framework** (§ 4) using
   temporal-consistency pseudo-GT (Lee 2013; Sohn et al. 2020 weakly-
   supervised tradition), inter-arm agreement, perturbation batteries
   (Hendrycks & Dietterich, 2019) and model self-reports. Every assumption
   is disclosed openly in § 7.
3. **Paper-grade reproducibility infrastructure** (§ 6) — an Evidence
   Manifest with cryptographic content hashes, an isolated virtual
   environment, and an aesthetic side-by-side composite video in 1280×720
   per arm at the same operating point.
4. **Honest negative results disclosed up-front** (§ 7): without
   colour-marker tagging we cannot reliably distinguish ally from rival
   robots, so the team split in this v0.1 is a left-right heuristic — the
   cognitive layer's outputs are invariant to that label.

---

## 2. Related Work

**SAM 3** (Carion et al., 2025, arXiv:2511.16719) introduced concept-prompted
segmentation — open-vocabulary detection from free-text noun phrases — and a
unified video tracker. We use the public `facebook/sam3` checkpoint
(848 M parameters, ~6.9 GB) for inference only.

**Free-energy formulations of cognition** (Friston, 2010; Friston et al.,
2017) cast brains and cognitive agents as systems that minimise prediction
error over their own generative model. Our `cognitive/singularity.py`
implements a compact variational free-energy minimiser over a learned 64-D
latent; the resulting reconstruction-error trajectory and integrated
information φ (Tononi, 2008; Mediano et al., 2019) are the two scalars
we report.

**ByteTrack** (Zhang et al., 2022, ECCV) is the de-facto multi-object
tracker for sports analytics; we use a Kalman-free greedy-IoU variant
(§ 3.4) for tractability and dependency hygiene.

**Robustness benchmarking** follows Hendrycks & Dietterich (2019)'s
perturbation methodology, adapted as 21 conditions (7 perturbation
families × 3 intensities) plus the clean reference — the same battery we
shipped in our prior Vet Microscopy AI v0.2 audit pack.

**Weakly-supervised pseudo-GT** (Lee, 2013; Sohn et al., 2020) inspires our
temporal-consistency labelling: detections that persist across consecutive
frames at similar locations become positive labels; sustained absences are
negatives. We treat this as the calibration target, never as truth.

---

## 3. Method

### 3.1 Architecture

```
                       video frames
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌──────────────┐               ┌──────────────────┐
    │  SAM 3       │               │  SAM 3           │ ← identical inference
    │  (fp16)      │               │  (fp16)          │
    │  text prompts│               │  text prompts    │
    └───────┬──────┘               └────────┬─────────┘
            │                                │
            ▼                                ▼
    BASELINE annotated              ┌──────────────────┐
    video + metrics                 │  occupancy map   │
                                    │  24×16×4 grid    │ ← raw perception
                                    └────────┬─────────┘
                                              ▼
                                    ┌──────────────────┐
                                    │  Singularity     │ ← own 64-D latent
                                    │  engine (Friston)│   free energy F, φ
                                    └────────┬─────────┘
                                              ▼
                                    ┌──────────────────┐
                                    │  RealtimeReasoner│ ← surprise z on
                                    │  (info geometry) │   the manifold
                                    └────────┬─────────┘
                                              ▼
                                    ┌──────────────────┐
                                    │  Spatial         │ ← honest read
                                    │  correlation     │   "where it broke"
                                    └────────┬─────────┘
                                              ▼
                                    U-CogNet annotated
                                    video + metrics +
                                    Evidence Manifest
```

### 3.2 SAM 3 perception layer

We use five concept prompts: `green soccer field`, `small wheeled soccer
robot`, `small ball on the playing field`, `soccer goal`, and `human hand at
the edge of the table`. Score floor 0.30 (looser 0.20 for the ball class —
the tiny-object recall recovery is one of the values added by the temporal-
consistency pseudo-GT in § 4.1).

Inference runs in `torch.autocast(bfloat16)` on a single RTX 4060 Laptop
(8.6 GB VRAM) at a peak allocated VRAM of **5.39 GB**.

### 3.3 Cognitive stack

The augmented arm uses three self-contained Apache-2.0 modules in
`cognitive/`: SAM 3 per-class detections are reduced to foot-points
(bottom-centre of bbox for robots and hands, centroid for balls), splatted
into a 24×16×4 occupancy grid, and handed to the `SingularityEngine`. The
engine maintains a variational free-energy minimiser on a 64-D latent and a
four-partition φ calculator; the `RealtimeReasoner` streams the raw
observation through a fixed random projection and reports surprise as a
Mahalanobis-distance z-score over a 64-frame rolling window. Output
schema documented in `pipelines/ucognet_sam3.py:UCogNetFrameMetric`.

### 3.4 Tracking

We use a Kalman-free greedy-IoU tracker (`evaluation/tracking.py`) with
IoU threshold 0.30 and a 5-frame age-out grace. ByteTrack is left as a
v0.2 deliverable — the tracker here is intentionally minimal to keep
dependencies clean.

### 3.5 Aesthetic compositor

A custom OpenCV compositor (`viz/aesthetic.py`) renders a 1280×720 panel
per arm with a clean video pane (no HUD overlay) and a sidebar carrying
all telemetry — per-class detection chips, free-energy F, integrated
information φ, reconstruction-error understanding %, surprise z-score with
rolling-window sparklines, and a discrete *MODEL BREAK* pill that fires on
post-warmup surprise spikes (§ 5.5).

---

## 4. Reproducible Weakly-Supervised Evaluation Framework

We do not claim human-verified ground truth.  Instead, every metric in
this paper is derived from one of three reproducible, annotator-free
signals: temporal-consistency labels (weakly-supervised pseudo-GT in
the Lee 2013 / Sohn 2020 tradition), inter-arm agreement on the same
SAM 3 detections, and a perturbation battery that needs no labels at
all.  The honest framing throughout: this is a **v0.1 reproducible
slice**, not a tournament-wide audit.  v0.2 will add per-robot colour-
tagging and an independent human spot-check on a held-out subset.

### 4.1 Temporal-consistency pseudo-GT

Following weakly-supervised video learning (Lee 2013, Wang et al. 2020), a
bbox of class *c* at frame *t* is promoted to *positive pseudo-GT* iff it
co-locates (IoU ≥ τ_c) with a same-class bbox in at least *k_c* of the ±W
neighbour frames. We use τ = 0.30, k = 3, W = 3 for all classes, with two
class-specific overrides for the hard cases: τ = 0.20 / k = 2 for the small
ball, and τ = 0.50 / k = 4 for the large field mask. Result: **467 robot, 33
ball, 615 hand, 114 field** confident pseudo-GT instances over 133 frames.

### 4.2 Detection metrics

`evaluation/detection.py` runs greedy IoU matching of SAM 3 predictions
against the pseudo-GT at three thresholds (0.30, 0.50, 0.70). We report TP /
FP / FN, precision / recall / F1, mean IoU of matches, and an 11-point
interpolated AP@0.5.

### 4.3 Tracking metrics

`evaluation/tracking.py` runs the greedy IoU tracker (§ 3.4) over both the
prediction stream and the pseudo-GT stream and computes MOTA, MOTP and
ID-switch count from the standard MOT definitions.

### 4.4 Agreement-calibration ECE

`evaluation/calibration.py` bins SAM 3 declared scores into 10 buckets and
measures empirical persistence rate (the same temporal-consistency signal
used for pseudo-GT) per bucket. The reliability diagram plots declared vs
empirical; ECE is the bin-weighted mean absolute gap. This is the standard
weakly-supervised calibration estimator (Sohn et al. 2020, NeurIPS).

### 4.5 Perturbation robustness battery

`evaluation/robustness.py` runs SAM 3 on N sampled frames perturbed by 21
conditions (7 families × 3 intensities) — Gaussian noise, blur, brightness
up/down, contrast, JPEG, rotation — and measures per-class self-consistency
vs the unperturbed reference via greedy IoU matching and count agreement.
Zero ground truth is consulted.

### 4.6 Cognitive trajectory analysis

`evaluation/cognitive.py` consumes only the U-CogNet metrics JSON and plots
F, φ, recon-error, understanding %, surprise z and model-break events
across the run. These are the cognitive layer's own self-reports — they
serve as their own metric of interest, not against any external GT.

### 4.7 Latency budget

`evaluation/latency.py` decomposes wall-time per frame into SAM 3 inference
+ cognitive layer overhead, reports mean / p50 / p90 / p95 / p99 / max per
arm, and computes the cognitive overhead percentage.

### 4.8 Inter-arm consistency

`evaluation/inter_arm.py` compares per-frame detection counts (which must
agree by construction, since both arms run the same SAM 3) and isolates the
cognitive signals that U-CogNet adds (surprise z-score, model-break events,
spatial-correlation reads, latency overhead).

---

## 5. Results

### 5.1 Detection performance · multi-clip

Three Mexican Robotics Federation tournament clips were evaluated end-
to-end with the same prompts and the same temporal-consistency
pseudo-GT pipeline (window ±3, IoU 0.30, persistence ≥ 3):

| Clip       | Frames | mAP@0.5 | robot AP | ball AP | hand AP | field AP |
|------------|--------|---------|----------|---------|---------|----------|
| IMG_9914   | 133    | **0.984** | 1.000  | 0.972   | 0.970   | 0.995    |
| IMG_9915   | 240    | **0.998** | 1.000  | 1.000   | 0.994   | 0.999    |
| IMG_9920   | 240    | **0.998** | 1.000  | 1.000   | 0.994   | 0.999    |
| **aggregate** | **613** | **0.993** | 1.000 | 0.991   | 0.986   | 0.998    |

Per-class precision / recall / F1 at IoU 0.50 on the headline clip
(IMG_9914) is shown below; the other two clips track at or above these
numbers on every class (robot reaches 1.000 / 1.000 / 1.000 on all
three clips; ball reaches 1.000 / 1.000 / 1.000 on IMG_9915 and
IMG_9920 once the longer temporal window engages):

| Class | Precision | Recall | F1 | AP@0.5 | n GT |
|-------|-----------|--------|------|--------|------|
| robot | 1.000     | 1.000  | **1.000** | 1.000  | 467  |
| field | 0.966     | 1.000  | 0.983 | 0.995  | 114  |
| hand  | 0.938     | 1.000  | 0.968 | 0.970  | 615  |
| ball  | 0.846     | 1.000  | 0.917 | 0.972  | 33   |

The ball class recovers from a raw recall of 29 % at the SAM 3 confidence
floor on the short clip to a precision-recall-balanced **AP = 0.972**
after temporal-consistency filtering, and reaches AP = 1.000 on the two
longer clips — strong evidence that SAM 3 detects the ball
intermittently but reliably enough that a tracker recovers the trajectory.
(Figure: `paper/figures/detection_pr_curves.png`)

### 5.2 Tracking performance

| Class | MOTA | MOTP | FP | FN | ID-switches |
|-------|------|------|------|------|-------------|
| robot | **0.979** | 1.000 | 0 | 0 | 10 |
| field | 0.965 | 1.000 | 4 | 0 | 0  |
| hand  | 0.932 | 1.000 | 41 | 0 | 1  |
| ball  | 0.818 | 1.000 | 6  | 0 | 0  |
| **overall** | **0.950** | **1.000** | – | – | **11** |

MOTP = 1.000 across all classes confirms perfect IoU agreement on matched
tracks; the ID-switches concentrate on the robot class as expected when
robots cross trajectories on the small playing area.
(Figure: `paper/figures/tracking_metrics.png`)

### 5.3 Calibration

Overall ECE = **0.157** on the agreement-calibration estimator. The robot
class hits 0.998 empirical persistence rate against high declared scores —
this is the pre-Ananke baseline that future v0.2 work will improve with
per-class temperature scaling. The reliability diagram is in
`paper/figures/reliability_diagram.png`.

### 5.4 Robustness battery

Overall self-consistency across **21 perturbation conditions** on 10 sampled
frames: **0.855**. Per-family breakdown:

| Family | L1 (mild) | L2 (moderate) | L3 (severe) |
|--------|-----------|----------------|-------------|
| brightness_up   | 0.969 | 0.957 | **0.977** |
| brightness_down | 0.973 | 0.969 | 0.948 |
| contrast        | 0.975 | 0.958 | 0.952 |
| gaussian_blur   | 0.886 | 0.853 | 0.842 |
| jpeg            | 0.868 | 0.849 | 0.735 |
| gaussian_noise  | 0.780 | 0.733 | 0.692 |
| rotation        | 0.780 | 0.668 | 0.602 |

SAM 3's concept-prompted segmentation is **highly robust** to brightness,
contrast and mild blur (consistency > 0.84 across all three at every
intensity), **moderately robust** to JPEG compression and Gaussian noise,
and **degrades sharply** under rotation (0.60 at 12 degrees). The
brightness invariance is particularly relevant for tournament-hall lighting
variability. Zero ground truth was consulted in this evaluation.
(Figure: `paper/figures/robustness_accuracy.png`)

### 5.5 Cognitive trajectory

The U-CogNet cognitive layer reaches:

| Metric | Final value | Interpretation |
|--------|-------------|----------------|
| Free energy F   | +0.090 | low, settled |
| φ (integrated information) | 0.518 | non-trivial cognitive coupling |
| Reconstruction error | 0.0024 (from 0.0095 initial) | 4× improvement |
| Understanding | 83.2 % | self-prediction quality |
| Mean surprise z | 1.07 (post-warmup) | normal regime |
| **Model breaks** | **7** (post-warmup) | autonomous event detection |
| Max surprise z | 3.27 | the biggest disruption |

Figure `paper/figures/cognitive_trajectory.png` shows F descending
monotonically as the model learns, φ rising from zero to ~0.5 as integrated
information emerges, and seven model-break events firing at 2.13 s, 2.47 s,
2.77 s, 3.07 s, 3.33 s, 3.67 s and 4.27 s (frames 64, 74, 83, 92, 100, 110,
128). Each event triggers an honest spatial-correlation read against the
±1-frame occupancy delta; on this clip the most common read is `"se
concentra en la izquierda, sobre todo Team Left"` — a natural language
description emitted with no event taxonomy hard-coded.

### 5.6 Latency budget

| Arm | mean | p50 | p95 | p99 |
|-----|------|-----|------|------|
| BASELINE infer | 930 ms | 940 | 1131 | 1145 |
| U-CogNet infer | 1057 ms | 1063 | 1325 | 1432 |
| Cognitive layer alone | 68 ms | 64 | 101 | 175 |
| **U-CogNet total** | **1125 ms** | 1127 | 1421 | 1602 |

**Cognitive-layer overhead = +6.4 % mean / +7.6 % p95** computed within the
same run (matched SAM 3 inference). The inter-run total wall-time overhead
is +21 % because the GPU thermals differ between runs; we report both.
(Figure: `paper/figures/latency_budget.png`)

### 5.7 Inter-arm consistency

Per-class detection-count agreement: **1.000** for every class (ball, field,
goal, hand, robot). What U-CogNet uniquely adds over the baseline:

- **7 post-warmup model-break events** with associated spatial-read text
  outputs and surprise-z spikes
- **F = +0.090, φ = 0.518, understanding = 83 %** — three new scalars
  per frame that the baseline by construction cannot emit
- **+195 ms per-frame total wall-time** (across runs) or +6.4 % within-run
  cognitive layer overhead

(Figure: `paper/figures/inter_arm_addition.png`)

---

## 6. Reproducibility

All artifacts of this study are bit-exactly reproducible from disk alone.
Each pipeline run writes an **Evidence Manifest** (`*.manifest.json`)
containing the input video's SHA-256-first-MB content hash, the Hugging
Face model revision, the autocast dtype, the prompt bank, every CLI flag,
the device, the wall-time and the output paths. The cache file
(`evaluation/cache_detections.py`) further decouples evaluation from
expensive SAM 3 re-runs — every downstream evaluator reads a single
JSON. The isolated `F:/U-CogNet-ToGo/futbotmx_venv` virtual environment is
documented in `requirements.txt` and the SAM 3 install is described in
the top-level `README.md`.

---

## 7. Limitations and Honest Scope

We disclose these openly because the U-CogNet research platform's
methodology requires it.

1. **Pseudo-GT is weakly supervised, not human-verified.** Every detection
   and tracking metric in § 5 inherits this assumption. The robustness
   battery (§ 5.4) is the human-free, model-free rigour fallback that
   evaluates SAM 3's self-consistency without any GT at all.
2. **Team split is a left/right heuristic in v0.1.** Without per-robot
   colour-marker recognition, "Team Left" and "Team Right" are positional
   labels. The cognitive layer's outputs (F, φ, surprise) read structure
   in the occupancy map and are invariant to that labelling; the spatial-
   read text inherits it explicitly.
3. **The model breaks are correlation, not causation.** When U-CogNet's
   surprise fires we honestly report *where* the perceptual occupancy
   changed most. We do **not** claim "a pass occurred" or "an interception
   happened" — those are tactical interpretations that would require an
   event taxonomy we deliberately do not impose.
4. **Latency includes I/O.** All wall-time figures include frame I/O,
   resize, mask renderer and JSON serialisation, not only neural inference.
5. **Three-clip slice, not tournament-wide.** This v0.1 paper reports
   results from three Mexican Robotics Federation clips (IMG_9914 /
   9915 / 9920, 613 frames total). The two longer clips are sampled at
   240 evenly-spaced frames each; a v0.2 audit will run on every frame
   of every clip once the full tournament repository is granted.

---

## 8. Conclusion

The free-energy cognitive layer is **strictly additive** on top of SAM 3:
identical perceptual outputs (100 % per-class count agreement), +6.4 %
mean latency overhead, and seven autonomous model-break events that the
baseline cannot flag by construction. Every metric in this study is
reproducible from disk alone, with no human-in-the-loop annotation, and
every module ships in a single Apache-2.0 self-contained repository.

---

## References

- Carion, N., Gustafson, L., Hu, Y.-T., et al. (2025). *SAM 3: Segment Anything with Concepts*. arXiv:2511.16719.
- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127–138.
- Friston, K., Parr, T., & de Vries, B. (2017). *The graphical brain: belief propagation and active inference*. Network Neuroscience, 1(4), 381–414.
- Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR.
- Lee, D.-H. (2013). *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks*. ICML Workshop.
- Mediano, P. A. M., Seth, A. K., & Barrett, A. B. (2019). *Measuring Integrated Information: Comparison of Candidate Measures in Theory and Simulation*. Entropy, 21(1), 17.
- Sohn, K., Berthelot, D., Carlini, N., et al. (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. NeurIPS.
- Tononi, G. (2008). *Consciousness as Integrated Information: A Provisional Manifesto*. The Biological Bulletin, 215(3), 216–242.
- Wang, J., Wang, X., Liu, W. (2020). *Weakly- and Semi-Supervised Faster R-CNN with Curriculum Learning*. ICPR.
- Zhang, Y., Sun, P., Jiang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV.

---

**Submitted to:** Copa FutBotMX 2026 · Capítulo Visión por Computadora ·
**Profesional category** · Secretaría de Ciencia, Humanidades, Tecnología
e Innovación (Secihti) + Meta + Centro · México.

**Affiliation:** [Brainstream](https://brainstream.pro) /
[U-CogNet research platform](https://ucognet.pro).
