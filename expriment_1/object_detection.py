import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from open_vocabulary_sidecar import ClosedLoopAnnotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time object detection using YOLO")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, video file path, or RTSP/HTTP stream URL",
    )
    parser.add_argument("--model", default="yolo12n.pt", help="YOLOv12 model weights")
    parser.add_argument("--confidence", type=float, default=0.45, help="Display confidence")
    parser.add_argument("--low-confidence", type=float, default=0.10, help="Mining threshold")
    parser.add_argument("--device", default="", help="Device: cpu, 0, 0,1, or mps")
    parser.add_argument("--output", default="", help="Optional output video path")
    parser.add_argument("--sidecar", default="sidecar", help="Active-learning sidecar directory")
    parser.add_argument("--jit-min-samples", type=int, default=50, help="Candidates before a sidecar training run")
    parser.add_argument("--jit-epochs", type=int, default=3, help="Epochs for each sidecar training run")
    parser.add_argument("--jit-cooldown", type=float, default=1.0, help="Seconds between sidecar samples")
    parser.add_argument("--closed-loop", action="store_true", help="Use OWL-ViT to name and automatically label uncertain detections")
    parser.add_argument("--viit-threshold", type=float, default=0.25, help="OWL-ViT open-vocabulary label confidence")
    parser.add_argument("--closed-loop-min-votes", type=int, default=3, help="Repeated ViT label votes required before accepting a name")
    parser.add_argument("--vocabulary", default="person,vehicle,robot,tool,phone,laptop,box,bottle,cup,bag,chair,drone,package,machine,keyboard,mouse,screen", help="Comma-separated open-vocabulary labels")
    return parser.parse_args()


def open_video_source(source: str) -> cv2.VideoCapture:
    try:
        source_value = int(source)
    except ValueError:
        source_value = source

    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}. Check the webcam index or stream URL."
        )
    return capture


def initialize_sidecar(sidecar: str, model_names: dict[int, str]) -> tuple[Path, Path, Path]:
    root = Path(sidecar)
    image_dir = root / "dataset" / "images" / "train"
    label_dir = root / "dataset" / "labels" / "train"
    weight_dir = root / "weights"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    data_file = root / "dataset" / "data.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in sorted(model_names.items()))
    data_file.write_text(f"path: {root / 'dataset'}\ntrain: images/train\nval: images/train\nnames:\n{names}\n", encoding="utf-8")
    return image_dir, label_dir, weight_dir


def collect_candidates(
    frame,
    result,
    image_dir: Path,
    label_dir: Path,
    low_confidence: float,
    display_confidence: float,
) -> int:
    collected = 0
    height, width = frame.shape[:2]

    for box in result.boxes:
        confidence = float(box.conf.item())
        if not low_confidence <= confidence < display_confidence:
            continue

        timestamp = int(time.time() * 1000)
        sample_id = f"candidate_{timestamp}_{collected}"
        image_path = image_dir / f"{sample_id}.jpg"
        label_path = label_dir / f"{sample_id}.txt"

        if not cv2.imwrite(str(image_path), frame):
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        class_id = int(box.cls.item())
        label_path.write_text(
            f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n",
            encoding="utf-8",
        )
        collected += 1

    return collected


def sidecar_training_running(sidecar_root: Path) -> bool:
    lock_file = sidecar_root / "training.lock"
    if not lock_file.exists():
        return False
    try:
        pid = int(lock_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        lock_file.unlink(missing_ok=True)
        return False


def start_sidecar_training(args: argparse.Namespace, sidecar_root: Path) -> None:
    if sidecar_training_running(sidecar_root):
        return

    label_dir = sidecar_root / "dataset" / "labels" / "train"
    image_dir = sidecar_root / "dataset" / "images" / "train"
    label_count = len(list(label_dir.glob("*.txt")))
    image_count = len(list(image_dir.glob("*.jpg")))
    if label_count < args.jit_min_samples or image_count < args.jit_min_samples:
        return

    trainer = Path(__file__).with_name("sidecar_trainer.py")
    command = [
        sys.executable,
        str(trainer),
        "--model",
        args.model,
        "--data",
        str(sidecar_root / "dataset" / "data.yaml"),
        "--sidecar",
        str(sidecar_root),
        "--epochs",
        str(args.jit_epochs),
        "--device",
        args.device,
    ]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_detection(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    capture = open_video_source(args.source)

    sidecar_root = Path(args.sidecar).resolve()
    image_dir, label_dir, weight_dir = initialize_sidecar(sidecar_root, model.names)

    closed_loop = None
    if args.closed_loop:
        closed_loop = ClosedLoopAnnotator(
            sidecar_root=sidecar_root,
            initial_names=model.names,
            candidate_labels=[name.strip() for name in args.vocabulary.split(",") if name.strip()],
            label_threshold=args.viit_threshold,
            minimum_votes=args.closed_loop_min_votes,
        )

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {args.output}")

    previous_time = time.perf_counter()
    previous_sample_time = 0.0
    previous_sidecar_check = time.perf_counter()
    previous_sidecar_mtime = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video stream ended or frame could not be read.")
                break

            inference_threshold = min(args.confidence, args.low_confidence)
            results = model.predict(
                frame,
                conf=inference_threshold,
                device=args.device,
                verbose=False,
            )
            result = results[0]

            if time.perf_counter() - previous_sample_time >= args.jit_cooldown:
                collected = collect_candidates(
                    frame,
                    result,
                    image_dir,
                    label_dir,
                    args.low_confidence,
                    args.confidence,
                )

                if closed_loop is not None:
                    collected += closed_loop.collect_auto_labeled(
                        frame=frame,
                        result=result,
                        image_dir=image_dir,
                        label_dir=label_dir,
                        low_confidence=args.low_confidence,
                        display_confidence=args.confidence,
                        votable_ids=set(model.names),
                    )

                if collected:
                    previous_sample_time = time.perf_counter()
            else:
                collected = 0

            annotated_frame = result.plot(conf=True)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            live_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                annotated_frame,
                f"FPS: {live_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if time.perf_counter() - previous_sidecar_check >= 2:
                previous_sidecar_check = time.perf_counter()
                start_sidecar_training(args, sidecar_root)

                updated_model = weight_dir / "best.pt"
                if updated_model.exists():
                    current_mtime = updated_model.stat().st_mtime_ns
                    if current_mtime != previous_sidecar_mtime:
                        previous_sidecar_mtime = current_mtime
                        model = YOLO(str(updated_model))
                        initialize_sidecar(sidecar_root, model.names)
                        print("Sidecar model loaded:", updated_model)

            cv2.imshow("YOLOv12 Object Detection", annotated_frame)
            if writer:
                writer.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    arguments = parse_args()
    run_detection(arguments)
