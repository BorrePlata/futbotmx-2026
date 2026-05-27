# Copa FutBotMX 2026 — Free-Energy Cognitive Augmentation on SAM 3

> Two-arm comparative study of robot-soccer video analysis using
> [SAM 3](https://arxiv.org/abs/2511.16719) (Meta AI). The **baseline arm**
> uses SAM 3 alone; the **U-CogNet arm** adds a free-energy variational
> self-representation, an information-geometry surprise reasoner and an
> autonomous spatial-correlation read. Submission to the
> [Copa FutBotMX 2026](https://secihti.mx/futbotmx/) **Profesional**
> category.
>
> 📄 **Paper:** [`paper/PAPER.md`](paper/PAPER.md) (3,800 words, 8 figures)
> 🎬 **Demo video (≤ 2 min):** `output/IMG_9914_FINAL_sxs_v2.mp4` — see § 4
> 📺 **Reel Instagram (≥ 30 s):** _pending publication_
> 🌐 **Challenge site:** https://secihti.mx/futbotmx/
> 📜 **License:** Apache 2.0 (code) · SAM License (model weights, not bundled)

---

## 1. What this submission delivers

The Copa FutBotMX 2026 § 3.5.1 requires a SAM 3 pipeline that segments
field, robots and ball, tracks their trajectories, and detects key events.
We deliver that as the **baseline arm**, then add a cognitive layer on top
in the **U-CogNet arm** and compare both rigorously:

| Component | Status | What it produces |
|---|---|---|
| `pipelines/baseline_sam3.py`           | ✅ | SAM 3 + greedy IoU tracker, annotated mp4 |
| `pipelines/ucognet_sam3.py`            | ✅ | SAM 3 + cognitive stack, annotated mp4 with F/φ/surprise HUD |
| `viz/aesthetic.py`                     | ✅ | Paper-grade compositor (sidebar + sparklines, no overlay on video) |
| `scripts/compose_side_by_side.py`      | ✅ | Final demo video — both arms in one frame |
| `evaluation/cache_detections.py`       | ✅ | Single-pass SAM 3 cache, every downstream evaluator reads from it |
| `evaluation/pseudo_gt.py`              | ✅ | Temporal-consistency pseudo-GT (no human labels) |
| `evaluation/detection.py`              | ✅ | mAP@0.5 = **0.984** vs pseudo-GT |
| `evaluation/tracking.py`               | ✅ | MOTA = **0.950**, MOTP = **1.000** |
| `evaluation/calibration.py`            | ✅ | Agreement-calibration ECE = 0.157 |
| `evaluation/robustness.py`             | ✅ | 21-condition perturbation battery, consistency = **0.855** |
| `evaluation/cognitive.py`              | ✅ | F, φ, surprise trajectory plots |
| `evaluation/latency.py`                | ✅ | Per-layer budget, cognitive overhead = +6.4 % mean |
| `evaluation/inter_arm.py`              | ✅ | 100 % detection agreement + 7 model-break events the baseline cannot flag |
| `paper/PAPER.md`                       | ✅ | Full scientific writeup with figures and references |

---

## 2. Installation

### 2.1 Requirements

- **OS:** Windows 10 / 11, Linux Ubuntu 22.04+, macOS 13+
- **Python:** 3.11 or 3.12
- **GPU:** NVIDIA with CUDA 12.1+, **≥ 8 GB VRAM** for fp16 inference
  (tested on RTX 4060 Laptop, 8.6 GB, peak allocated = 5.4 GB)
- **Disk:** ~12 GB free (~6 GB SAM 3 weights + workspace)
- **A Hugging Face account** that has accepted the SAM License at
  https://huggingface.co/facebook/sam3 (one-time click).
- **HF token** with "Access public gated repositories" enabled in
  https://huggingface.co/settings/tokens (a classic Read token works).

### 2.2 Setup

```bash
# 1. Clone the submission repo
git clone https://github.com/BorrePlata/futbotmx-2026.git futbotmx
cd futbotmx

# 2. Create a dedicated virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate      # Linux/macOS

# 3. Install base dependencies
pip install -r requirements.txt

# 4. Install SAM 3 from the official source (no PyPI release yet)
pip install --no-build-isolation \
            "git+https://github.com/facebookresearch/sam3.git"

# 5. (Windows only) install triton-windows for the SAM 3 video tracker
pip install triton-windows

# 6. Authenticate with Hugging Face and download the SAM 3 weights
huggingface-cli login                    # paste your HF token
export HF_HOME=/data/sam3                # Linux/macOS
# $env:HF_HOME = "F:/U-CogNet-ToGo/sam3" # Windows PowerShell
python scripts/download_sam3.py
```

### 2.3 Tournament videos

The convocatoria's video repository (Federación Mexicana de Robótica) is
**not** redistributed here.  Download to a local path and pass it via
`--video`:

```bash
# example
python -m pipelines.baseline_sam3 \
       --video /path/to/IMG_9914.MOV --max-side 720 --aesthetic
```

---

## 3. Reproducing the headline numbers

The full evaluation chain on a sample video, end-to-end:

```bash
# 1.  Run both arms (each ~3 min on a 4.4 s 1080p clip, RTX 4060)
python -m pipelines.baseline_sam3  --video <video> --max-side 720 --aesthetic
python -m pipelines.ucognet_sam3   --video <video> --max-side 720 --aesthetic

# 2.  Compose the side-by-side demo video (no SAM 3 re-run)
python -m scripts.compose_side_by_side \
       --baseline output/<stem>_baseline_aesthetic.mp4 \
       --ucognet  output/<stem>_ucognet_aesthetic.mp4 \
       --out      output/<stem>_FINAL_sxs.mp4

# 3.  Cache detections once for the evaluation chain
python -m evaluation.cache_detections --video <video> --max-side 720

# 4.  Pseudo-GT + detection metrics (mAP@0.5)
python -m evaluation.pseudo_gt --cache output/<stem>.cache.json
python -m evaluation.detection \
       --cache output/<stem>.cache.json \
       --pseudo-gt output/<stem>.pseudo_gt.json

# 5.  Tracking (MOTA / MOTP / ID-switch)
python -m evaluation.tracking \
       --cache output/<stem>.cache.json \
       --pseudo-gt output/<stem>.pseudo_gt.json

# 6.  Calibration (ECE) + Cognitive trajectory + Latency budget + Inter-arm
python -m evaluation.calibration --cache output/<stem>.cache.json
python -m evaluation.cognitive   --metrics output/<stem>_ucognet.metrics.json
python -m evaluation.latency     --baseline-metrics output/<stem>_baseline.metrics.json \
                                  --ucognet-metrics  output/<stem>_ucognet.metrics.json
python -m evaluation.inter_arm   --baseline-metrics output/<stem>_baseline.metrics.json \
                                  --ucognet-metrics  output/<stem>_ucognet.metrics.json

# 7.  Robustness battery (21 conditions, ~6 min on the sample video)
python -m evaluation.robustness  --video <video> --n-sample 10
```

Each `paper/figures/*.png` and each `paper/*.json` is regenerated from
disk; the `Evidence Manifest` files (`output/*.manifest.json`) record the
input hash, the model revision, and every CLI flag for bit-exact
reproducibility.

---

## 4. Results

Headline numbers reproduced on `IMG_9914.MOV` (133 frames @ 30 fps, 1920×1080
resized to 720 px max-side) on a single RTX 4060 Laptop 8.6 GB:

| Axis | Number | Figure |
|---|---|---|
| Detection mAP@0.5            | **0.984** | `paper/figures/detection_pr_curves.png` |
| Per-class P / R / F1         | robot 1.000 / 1.000 / 1.000 · field 0.966 / 1.000 / 0.983 · hand 0.938 / 1.000 / 0.968 · ball 0.846 / 1.000 / 0.917 | `paper/figures/detection_per_class.png` |
| Tracking MOTA / MOTP / IDSW  | **0.950 / 1.000 / 11** | `paper/figures/tracking_metrics.png` |
| Calibration ECE              | 0.157 (baseline before temperature scaling) | `paper/figures/reliability_diagram.png` |
| Robustness battery (21 cond) | **overall 0.855** · brightness ≥ 0.95 · rotation L3 0.60 | `paper/figures/robustness_accuracy.png` |
| Cognitive (U-CogNet only)    | F final +0.090 · φ final 0.518 · understanding 83 % · **7 model breaks** | `paper/figures/cognitive_trajectory.png` |
| Latency budget               | baseline 930 ms mean · U-CogNet 1125 ms · cognitive overhead **+6.4 % mean** | `paper/figures/latency_budget.png` |
| Inter-arm consistency        | detection agreement 100 % all classes · U-CogNet adds 7 model-break events | `paper/figures/inter_arm_addition.png` |

Demo video (the file the Copa FutBotMX evaluator opens):
`output/IMG_9914_FINAL_sxs_v2.mp4` — 6.7 MB, 2568×776, 4.4 s, side-by-side
BASELINE | U-CogNet with full sidebar telemetry. The reel for Instagram
(≥ 30 s) is published separately and linked from this README upon submission.

---

## 5. Project structure

```
futbotmx/
├── README.md                    ← this file
├── LICENSE                      ← Apache 2.0 (our code)
├── LICENSE-SAM                  ← SAM License copy (SAM 3 weights, NOT bundled)
├── NOTICE                       ← attributions
├── requirements.txt
├── .gitignore
│
├── pipelines/
│   ├── baseline_sam3.py         ← SAM 3 reference arm
│   └── ucognet_sam3.py          ← SAM 3 + cognitive stack
│
├── cognitive/                   ← Apache 2.0 self-contained subpackage
│   ├── singularity.py           ← variational free-energy + φ
│   ├── reasoner.py              ← real-time Mahalanobis surprise
│   └── match.py                 ← MatchCognition + occupancy encoder
│
├── viz/
│   └── aesthetic.py             ← paper-grade compositor (sidebar + sparklines)
│
├── evaluation/
│   ├── palette.py               ← paper-grade matplotlib style
│   ├── cache_detections.py
│   ├── pseudo_gt.py             ← weakly-supervised temporal-consistency GT
│   ├── detection.py             ← IoU + mAP
│   ├── tracking.py              ← MOTA / MOTP / IDSW
│   ├── calibration.py           ← agreement-calibration ECE
│   ├── robustness.py            ← 21-condition perturbation battery
│   ├── cognitive.py             ← F, φ, surprise plots
│   ├── latency.py               ← per-layer budget
│   └── inter_arm.py             ← what U-CogNet adds over baseline
│
├── scripts/
│   ├── download_sam3.py         ← downloads model weights into HF_HOME
│   ├── smoke_test.py            ← verify SAM 3 loads + segments truck.jpg
│   ├── extract_frame.py         ← pull a single video frame
│   ├── visualize_inference.py   ← multi-prompt static visualisation
│   ├── live_view.py             ← OpenCV live preview (cv2.imshow)
│   └── compose_side_by_side.py  ← final 2-pane comparison video
│
├── paper/
│   ├── PAPER.md                 ← full scientific writeup
│   └── figures/                 ← 10 paper-grade PNGs
│
└── output/                      ← gitignored, all run artefacts land here
```

---

## 6. Innovations (Copa FutBotMX § 3.7.3 — Profesional category)

| Innovation axis | This work |
|---|---|
| **Prompts & context**  | Five concept prompts hand-tuned for robot soccer (field, robot, ball, goal, hand) with class-specific score floors (looser for the tiny ball) |
| **Fine-tuning**        | Deliberately NOT applied — we use the public `facebook/sam3` checkpoint as a fixed sensory layer to isolate the cognitive layer's contribution |
| **Tracker integration** | Greedy IoU multi-object tracker (no Kalman, no ByteTrack dependency) reproducing standard MOT metrics |
| **Post-processing**     | Foot-point extraction → 24×16×4 occupancy grid → variational free-energy minimisation → Mahalanobis surprise z-score → spatial-correlation read on the perception delta |

---

## 7. Licensing

- **All source code in this repository** is released under the **Apache
  License, Version 2.0** (see [`LICENSE`](LICENSE)).
- **SAM 3 model weights** (downloaded at install time from Hugging Face)
  are governed by Meta's **SAM License** (a reference copy is included
  in [`LICENSE-SAM`](LICENSE-SAM)). We do not redistribute the weights.
- **Tournament videos** are property of the Federación Mexicana de
  Robótica. We do not redistribute them.
- **Third-party Python packages** are governed by their own licenses
  (Apache 2.0 / MIT / BSD) — see [`NOTICE`](NOTICE) for the full list.

---

## 8. Citation

```bibtex
@misc{plata2026futbotmx,
  title  = {Free-Energy Cognitive Augmentation on SAM 3 for Robot-Soccer Video Analysis},
  author = {Plata, Samuel},
  year   = {2026},
  note   = {Copa FutBotMX 2026 Computer Vision Challenge, Profesional category},
  url    = {https://github.com/BorrePlata/futbotmx-2026},
}
```

---

## 9. Contact

- **Author:** Samuel Plata
- **Email:** samuel@brainstream.pro
- **Affiliation:** [Brainstream](https://brainstream.pro) /
  [U-CogNet research platform](https://ucognet.pro)

Built for the Copa FutBotMX 2026, in the spirit of the
2026 FIFA World Cup hosted by México–Canada–USA. 🇲🇽⚽🤖
