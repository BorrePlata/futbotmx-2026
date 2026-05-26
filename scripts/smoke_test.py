"""smoke_test — verify SAM 3 loads + runs inference on the isolated venv.

Run with the isolated venv directly:
  & F:/U-CogNet-ToGo/futbotmx_venv/Scripts/python.exe `
    -m experiments.futbotmx.scripts.smoke_test
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    _load_dotenv(_REPO_ROOT / ".env")
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "F:/U-CogNet-ToGo/sam3")

    import torch
    from PIL import Image
    print(f"[smoke] torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[smoke] GPU: {p.name}  {p.total_memory/1e9:.1f} GB")

    print("[smoke] loading SAM 3 image model …")
    t0 = time.time()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device="cuda", load_from_HF=True)
    processor = Sam3Processor(model)
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s")

    img_path = Path("F:/U-CogNet-ToGo/sam3_src/assets/images/truck.jpg")
    if not img_path.exists():
        print(f"[smoke] ERROR: sample image not found at {img_path}")
        return 1

    print(f"[smoke] running text-prompt inference on {img_path.name} …")
    image = Image.open(img_path).convert("RGB")
    t1 = time.time()
    # SAM 3 backbone is loaded in bfloat16; wrap in autocast so input
    # activations match weight dtype across the vitdet trunk.
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        out = processor.set_text_prompt(state=state, prompt="truck")
    print(f"[smoke] inference: {time.time()-t1:.2f}s")

    masks = out.get("masks")
    boxes = out.get("boxes")
    scores = out.get("scores")
    print(f"[smoke] result: masks={getattr(masks,'shape',None)} "
          f"boxes={getattr(boxes,'shape',None)} scores={getattr(scores,'shape',None)}")
    if scores is not None and len(scores) > 0:
        topk = min(5, len(scores))
        for i in range(topk):
            s = float(scores[i])
            b = boxes[i].tolist() if boxes is not None else None
            print(f"  #{i}: score={s:.3f}  box={b}")

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] peak VRAM allocated: {vram:.2f} GB")
    print("[smoke] OK ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
