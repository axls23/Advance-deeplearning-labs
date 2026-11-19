# Experiment 1: YOLO Object Detection

This experiment uses a pretrained Ultralytics YOLO model to detect and recognize objects in a live webcam stream, video file, or IP/RTSP stream.

## Setup

```bash
cd expriment_1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Webcam:

```bash
python object_detection.py --source 0
```

Video file:

```bash
python object_detection.py --source input.mp4 --output result.mp4
```

RTSP or HTTP stream:

```bash
python object_detection.py --source "rtsp://camera-address"
```

GPU (if available):

```bash
python object_detection.py --source 0 --device 0
```

Press `q` or `Esc` to stop.

## Options

- `--model`: YOLO weights; default is `yolo12n.pt`
- `--confidence`: minimum confidence, for example `0.45`
- `--device`: `cpu`, GPU index such as `0`, or another supported device
- `--output`: save the annotated video to a file

The YOLO model may download pretrained weights on the first run.

## Low-confidence sidecar training

The detector can collect uncertain detections into a separate active-learning dataset and start a background training process after enough candidates are available.

```bash
python object_detection.py --source 0 --confidence 0.45 --low-confidence 0.10 --jit-min-samples 50 --jit-epochs 3
```

Options:

- `--low-confidence`: minimum confidence for a candidate; defaults to `0.10`
- `--sidecar`: sidecar directory; defaults to `sidecar`
- `--jit-min-samples`: unlabeled candidates required before retraining starts
- `--jit-epochs`: epochs used by a background retraining run
- `--jit-cooldown`: minimum seconds between saved candidate frames

Candidates are stored under `sidecar/dataset/` as pseudo-labeled YOLO images and labels. The updated `sidecar/weights/best.pt` is loaded automatically. This feature learns only from the model's existing class names and low-confidence detections; it cannot discover or name a genuinely unknown object class on its own. Unknown objects should be manually reviewed and assigned a new class label before building a new training dataset.

## Automatic closed-loop sidecar

The sidecar can use OWL-ViT to assign names to uncertain YOLO detections, automatically create YOLO labels, and retrain the YOLO checkpoint.

```bash
python object_detection.py --source 0 --closed-loop --confidence 0.45 --low-confidence 0.10 --viit-threshold 0.25 --closed-loop-min-votes 3 --jit-min-samples 50 --jit-epochs 3
```

Additional closed-loop options:

- `--closed-loop`: enables OWL-ViT automatic vocabulary naming
- `--vocabulary`: comma-separated candidate object names
- `--viit-threshold`: minimum OWL-ViT name confidence
- `--closed-loop-min-votes`: repeated sightings required before a name becomes a new YOLO class

The sidecar stores the vocabulary in `closed_loop_classes.json`, writes YOLO class IDs to `data.yaml`, saves ViT-labeled bounding boxes, and starts background YOLO retraining. Candidate names are chosen only from `--vocabulary`; adding an exact, domain-specific label allows the cycle to expand beyond the original COCO vocabulary.

This mode is not fully human-free for safety-critical use. OWL-ViT may hallucinate names or use incorrect candidate boxes. Keep a conservative ViT threshold and minimum vote count, separate held-out validation data, and a fallback original checkpoint.
