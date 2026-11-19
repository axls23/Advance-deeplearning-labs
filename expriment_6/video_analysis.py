"""Experiment 6: Video analysis - fine-tune YOLO11 (checkpoint from experiment 1)
and use it for multi-object tracking + motion detection.

  python video_analysis.py train   --epochs 30
  python video_analysis.py analyze --source video.mp4
  python video_analysis.py all     --source video.mp4
"""
import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
CKPT = HERE.parent / "expriment_1" / "yolo11n.pt"   # checkpoint from experiment 1
RUNS = HERE / "runs"


def train(a):
    model = YOLO(str(a.checkpoint))
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device or 0, project=str(RUNS), name="finetune",
                exist_ok=True, workers=2, plots=True)
    best = RUNS / "finetune" / "weights" / "best.pt"
    print("fine-tuned weights:", best)
    return best


def estimate_ego(prev, cur, prev_boxes):
    """Camera (ego) motion prev->cur as a 2x3 similarity transform.
    Features are taken from background only (detected objects masked out) and
    fitted with RANSAC, so independently moving objects are rejected as outliers."""
    mask = np.full(prev.shape, 255, np.uint8)
    for x1, y1, x2, y2 in prev_boxes:
        mask[max(0, y1):y2, max(0, x1):x2] = 0
    p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8, mask=mask)
    ident = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    if p0 is None or len(p0) < 8:
        return ident
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return ident
    M, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.5)
    return ident if M is None else M.astype(np.float32)


def residual_motion_mask(prev, cur, M, thr=25):
    """Warp previous frame by the ego motion, then difference: what is left
    is motion NOT explained by the camera (independent object motion)."""
    h, w = cur.shape
    warped = cv2.warpAffine(prev, M, (w, h))
    valid = cv2.warpAffine(np.full_like(prev, 255), M, (w, h)) > 0
    valid = cv2.erode(valid.astype(np.uint8), None, iterations=6) > 0
    d = cv2.absdiff(cv2.GaussianBlur(warped, (5, 5), 0), cv2.GaussianBlur(cur, (5, 5), 0))
    m = (cv2.threshold(d, thr, 255, cv2.THRESH_BINARY)[1] > 0) & valid
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, None, iterations=2)


def analyze(a, weights=None):
    weights = weights or a.weights or a.checkpoint   # experiment-1 detector by default
    print("using weights:", weights)
    model = YOLO(str(weights))

    src = int(a.source) if str(a.source).isdigit() else str(a.source)  # webcam index or file
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    OUT = RUNS / time.strftime("session_%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "tracked.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (W, H))
    dt = a.stride / fps

    trails = defaultdict(lambda: deque(maxlen=60))   # stored in the stabilised (world) frame
    last, ema, state = {}, {}, {}
    rows, ego_series, obj_series = [], [], []
    prev_gray, prev_boxes, idx = None, [], -1
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or (a.max_frames and idx >= a.max_frames * a.stride):
                break
            idx += 1
            if idx % a.stride:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- 1. ego motion (camera) --------------------------------------
            if prev_gray is None:
                M, rmask = np.array([[1, 0, 0], [0, 1, 0]], np.float32), np.zeros_like(gray)
            else:
                M = estimate_ego(prev_gray, gray, prev_boxes)
                rmask = residual_motion_mask(prev_gray, gray, M)
            ego_px = float(np.hypot(M[0, 2], M[1, 2]))
            ego_rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
            ego_series.append(ego_px / dt)
            obj_series.append(float((rmask > 0).mean()))
            # move existing trails into the new frame so static things stay put
            for k, tr in trails.items():
                if tr:
                    pts = np.array(tr, np.float32)
                    tr.clear(); tr.extend(map(tuple, pts @ M[:, :2].T + M[:, 2]))
            prev_gray = gray

            # ---- 2. detect + track with the experiment-1 detector ------------
            r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=a.conf,
                            device=a.device or 0, verbose=False)[0]
            vis = frame.copy()
            vis[rmask > 0] = (0.5 * vis[rmask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            prev_boxes, n_moving = [], 0
            if r.boxes.id is not None:
                for box, tid, cls, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().tolist(),
                                             r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    x1, y1, x2, y2 = box.astype(int)
                    prev_boxes.append((x1, y1, x2, y2))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    # ---- 3. object motion, measured in the stabilised frame ------
                    trails[tid].append((cx, cy))
                    # trail points live in the stabilised frame, so displacement over a
                    # short window is pure object motion; the window averages out box jitter
                    tr = trails[tid]
                    k = min(len(tr) - 1, a.window)
                    res = 0.0 if k < 3 else float(np.hypot(tr[-1][0] - tr[-1 - k][0], tr[-1][1] - tr[-1 - k][1]) / (k * dt))
                    ema[tid] = res
                    # hysteresis so a label does not flicker between static and moving
                    was = state.get(tid, False)
                    moving = ema[tid] > (a.move_thr * 0.5 if was else a.move_thr)
                    state[tid] = moving
                    rows.append(dict(frame=idx, time=idx / fps, id=tid, cls=r.names[cls], conf=cf,
                                     cx=cx, cy=cy, obj_speed_px_s=ema[tid], moving=moving))
                    if moving:
                        n_moving += 1
                        color = tuple(int(c) for c in np.random.RandomState(tid).randint(60, 255, 3))
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis, f"#{tid} {r.names[cls]} {ema[tid]:.0f}px/s", (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.polylines(vis, [np.array(trails[tid], np.int32).reshape(-1, 1, 2)], False, color, 2)
                    elif not a.hide_static:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (170, 170, 170), 1)
                        cv2.putText(vis, f"#{tid} {r.names[cls]} static", (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
            cv2.rectangle(vis, (0, 0), (W, 32), (0, 0, 0), -1)
            cv2.putText(vis, f"EGO  {ego_series[-1]:.0f}px/s rot {ego_rot:+.2f}deg | OBJECT motion {obj_series[-1]*100:.1f}% | moving objs {n_moving}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vw.write(vis)
            if a.show:
                cv2.imshow("tracking", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("interrupted - saving results")
    finally:
        cap.release(); vw.release(); cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tracks.csv", index=False)
    pd.DataFrame(dict(ego_px_s=ego_series, object_motion_fraction=obj_series)).to_csv(OUT / "motion.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(ego_series)) * dt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, ego_series, color="tab:blue"); ax[0].set(title="Ego (camera) motion", xlabel="s", ylabel="px/s")
    ax[1].plot(t, np.array(obj_series) * 100, color="tab:red")
    ax[1].set(title="Independent object motion (ego removed)", xlabel="s", ylabel="% pixels")
    fig.tight_layout(); fig.savefig(OUT / "analysis.png", dpi=120)

    if len(df):
        s = df.groupby("id").agg(cls=("cls", lambda x: x.mode()[0]), frames=("frame", "count"),
                                 moving_frames=("moving", "sum"), mean_obj_speed=("obj_speed_px_s", "mean"))
        print(f"{len(s)} tracks, {(s.moving_frames > 0).sum()} with independent motion\n",
              s[s.moving_frames > 0].sort_values("moving_frames", ascending=False).head(10))
    print(f"mean ego {np.mean(ego_series):.0f}px/s | mean object-motion {np.mean(obj_series)*100:.1f}% of pixels")
    print("video:", out_path, "\ncsv/plot in:", OUT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "analyze", "all"])
    p.add_argument("--checkpoint", type=Path, default=CKPT, help="experiment-1 yolo11n checkpoint")
    p.add_argument("--data", default="coco8.yaml", help="YOLO dataset yaml for fine-tuning")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="")
    p.add_argument("--source", default="0", help="webcam index (as in experiment 1) or video file")
    p.add_argument("--show", action="store_true", help="live preview window, q/Esc to stop")
    p.add_argument("--weights", type=Path)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    p.add_argument("--move-thr", type=float, default=40, help="px/s of ego-compensated motion to count as moving")
    p.add_argument("--window", type=int, default=8, help="frames over which object displacement is measured")
    p.add_argument("--hide-static", action="store_true", help="do not draw static objects")
    p.add_argument("--max-frames", type=int, default=0)
    a = p.parse_args()
    w = train(a) if a.mode in ("train", "all") else None
    if a.mode in ("analyze", "all"):
        analyze(a, w)


if __name__ == "__main__":
    main()
