import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ClosedLoopAnnotator:
    """Names uncertain crops with an open-vocabulary ViT and maintains YOLO class IDs."""

    def __init__(
        self,
        sidecar_root: Path,
        initial_names: dict[int, str],
        candidate_labels: list[str],
        label_threshold: float,
        minimum_votes: int,
    ) -> None:
        self.sidecar_root = Path(sidecar_root)
        self.registry_file = self.sidecar_root / "closed_loop_classes.json"
        self.names = dict(initial_names)
        self.candidate_labels = candidate_labels
        self.label_threshold = label_threshold
        self.minimum_votes = minimum_votes
        self.votes: dict[str, int] = defaultdict(int)
        self.detector = None

        if self.registry_file.exists():
            try:
                saved = json.loads(self.registry_file.read_text(encoding="utf-8"))
                self.names = {int(index): name for index, name in saved.items()}
            except (json.JSONDecodeError, ValueError):
                pass

    @staticmethod
    def normalize_name(name: str) -> str:
        name = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
        return name.strip("_") or "unknown_object"

    def load_detector(self) -> bool:
        if self.detector is not None:
            return True
        try:
            from transformers import pipeline

            self.detector = pipeline(
                task="zero-shot-object-detection",
                model="google/owlvit-base-patch32",
                device=-1,
            )
            return True
        except Exception as exc:
            print("Closed-loop ViT unavailable:", exc)
            return False

    def get_or_add_name(self, name: str) -> int:
        name = self.normalize_name(name)
        for class_id, existing_name in self.names.items():
            if existing_name == name:
                return class_id

        class_id = max(self.names, default=-1) + 1
        self.names[class_id] = name
        self.registry_file.write_text(
            json.dumps({str(index): value for index, value in self.names.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"Closed loop added vocabulary: {class_id}={name}")
        return class_id

    def update_vocabulary_file(self) -> Path:
        data_file = self.sidecar_root / "dataset" / "data.yaml"
        names = "\n".join(
            f"  {index}: {name}" for index, name in sorted(self.names.items())
        )
        data_file.write_text(
            f"path: {self.sidecar_root / 'dataset'}\n"
            "train: images/train\n"
            "val: images/train\n"
            f"names:\n{names}\n",
            encoding="utf-8",
        )
        return data_file

    def name_crop(self, frame: np.ndarray, box) -> tuple[str | None, float]:
        if not self.load_detector():
            return None, 0.0

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        pad_x = max(4, int((x2 - x1) * 0.15))
        pad_y = max(4, int((y2 - y1) * 0.15))

        left = max(0, int(x1 - pad_x))
        top = max(0, int(y1 - pad_y))
        right = min(width, int(x2 + pad_x))
        bottom = min(height, int(y2 + pad_y))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None, 0.0

        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        try:
            detections = self.detector(
                image,
                candidate_labels=self.candidate_labels,
                threshold=self.label_threshold,
            )
        except Exception as exc:
            print("Closed-loop ViT inference failed:", exc)
            return None, 0.0

        if not detections:
            return None, 0.0

        best = max(detections, key=lambda item: float(item["score"]))
        name = self.normalize_name(str(best["label"]))
        score = float(best["score"])
        return name, score

    def collect_auto_labeled(
        self,
        frame: np.ndarray,
        result,
        image_dir: Path,
        label_dir: Path,
        low_confidence: float,
        display_confidence: float,
        votable_ids: set[int],
    ) -> int:
        height, width = frame.shape[:2]
        collected = 0

        for box_index, box in enumerate(result.boxes):
            confidence = float(box.conf.item())
            if confidence >= display_confidence:
                continue

            original_class_id = int(box.cls.item())
            if original_class_id not in votable_ids:
                continue

            name, viit_score = self.name_crop(frame, box)
            if name is None:
                continue

            self.votes[name] += 1
            if self.votes[name] < self.minimum_votes:
                continue

            class_id = self.get_or_add_name(name)
            timestamp = int(time.time() * 1000)
            sample_id = f"closed_loop_{timestamp}_{box_index}"
            image_path = image_dir / f"{sample_id}.jpg"
            label_path = label_dir / f"{sample_id}.txt"

            if not cv2.imwrite(str(image_path), frame):
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height

            label_path.write_text(
                f"{class_id} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n",
                encoding="utf-8",
            )
            collected += 1

        if collected:
            self.update_vocabulary_file()
        return collected
