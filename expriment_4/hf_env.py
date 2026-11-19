"""Load HF_TOKEN from this folder's .env ONLY; ignore any ambient/cached login."""
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"   # never auto-use a cached/global token
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
env = Path(__file__).parent / ".env"
for line in (env.read_text().splitlines() if env.exists() else []):   # public models work without a token
    k, _, v = line.partition("=")
    if k.strip() == "HF_TOKEN" and v.strip():
        os.environ["HF_TOKEN"] = v.strip()
