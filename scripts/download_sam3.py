"""download_sam3 — fetch facebook/sam3 weights to F:/U-CogNet-ToGo/sam3/.

Idempotent: if files already exist in the HF cache they are skipped.
Requires the SAM License to be accepted at huggingface.co/facebook/sam3
and a valid HF_TOKEN (read from the project-root .env or env var).

Usage:
  python -m experiments.futbotmx.scripts.download_sam3
  python -m experiments.futbotmx.scripts.download_sam3 --revision main
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HF_HOME = Path("F:/U-CogNet-ToGo/sam3")


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — no external dep, no shell injection."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", default="facebook/sam3")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME,
                    help="Where HuggingFace caches models")
    args = ap.parse_args()

    _load_dotenv(_REPO_ROOT / ".env")

    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_home)

    token = os.environ.get("HF_TOKEN", "")
    if not token.startswith("hf_"):
        print("ERROR: HF_TOKEN not found (.env or env). Set HF_TOKEN=hf_xxx",
              file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)

    who = api.whoami()
    print(f"[sam3] logged in as: {who.get('name')}")
    print(f"[sam3] HF_HOME     : {args.hf_home}")
    print(f"[sam3] repo        : {args.repo_id} @ {args.revision}")

    t0 = time.time()
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        token=token,
        cache_dir=str(args.hf_home),
        local_dir_use_symlinks=False,
    )
    elapsed = time.time() - t0
    print(f"[sam3] downloaded in {elapsed:.1f}s")
    print(f"[sam3] local path  : {path}")

    # Report disk usage
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    print(f"[sam3] total size  : {total/1e9:.2f} GB")

    # Brief file listing
    print("[sam3] files:")
    for p in sorted(Path(path).rglob("*"), key=lambda p: -p.stat().st_size if p.is_file() else 0):
        if p.is_file():
            print(f"  {p.stat().st_size/1e6:8.1f} MB  {p.relative_to(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
