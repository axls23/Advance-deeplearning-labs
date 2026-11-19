import argparse
import os
import sys
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain YOLO from sidecar pseudo-label samples")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.sidecar).resolve()
    lock_file = root / "training.lock"

    if lock_file.exists():
        print("Sidecar training is already running")
        return

    lock_file.write_text(str(os.getpid()), encoding="utf-8")
    try:
        model = YOLO(args.model)
        output = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            imgsz=640,
            project=str(root / "weights"),
            name="run",
            exist_ok=True,
            verbose=False,
        )
        trained = Path(output.save_dir) / "weights" / "best.pt"
        destination = root / "weights" / "best.pt"
        if trained.exists():
            destination.write_bytes(trained.read_bytes())
            print("Sidecar model updated:", destination)
    finally:
        lock_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
