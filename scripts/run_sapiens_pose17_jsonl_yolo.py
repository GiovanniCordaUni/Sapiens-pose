import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


COCO17_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]


def preprocess_bgr(frame_bgr: np.ndarray, in_w: int, in_h: int) -> torch.Tensor:
    resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float()


def heatmaps_to_keypoints(heatmaps: np.ndarray, out_w: int, out_h: int):
    K, Hm, Wm = heatmaps.shape
    flat = heatmaps.reshape(K, -1)
    idx = np.argmax(flat, axis=1)
    scores = flat[np.arange(K), idx]

    ys = (idx // Wm).astype(np.float32)
    xs = (idx %  Wm).astype(np.float32)

    xs = xs * (out_w / max(Wm - 1, 1))
    ys = ys * (out_h / max(Hm - 1, 1))

    keypoints = np.stack([xs, ys], axis=1)
    return keypoints, scores


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return [x1, y1, x2, y2]


def add_padding_xyxy(box, W, H, pad=0.20):
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    return clamp_box_xyxy([x1 - px, y1 - py, x2 + px, y2 + py], W, H)


def pick_person_box(yolo_result, prev_box=None, iou_bias=0.7):
    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls  = boxes.cls.cpu().numpy().astype(int)

    candidates = [(xyxy[i], float(conf[i])) for i in range(len(cls)) if cls[i] == 0]
    if not candidates:
        return None, None

    if prev_box is None:
        best = max(candidates, key=lambda t: t[1])
        return best[0].tolist(), best[1]

    def score_fn(item):
        box, c = item
        return (1 - iou_bias) * c + iou_bias * iou_xyxy(box, prev_box)

    best = max(candidates, key=score_fn)
    return best[0].tolist(), best[1]


def draw_coco17(frame_bgr, kpts17_xy, scores17, kpt_thr=0.35, radius=5, thickness=2):
    out = frame_bgr.copy()

    for i in range(17):
        if float(scores17[i]) < kpt_thr:
            continue
        x, y = int(kpts17_xy[i, 0]), int(kpts17_xy[i, 1])
        cv2.circle(out, (x, y), radius, (0, 255, 0), -1)

    for a, b in COCO17_EDGES:
        if float(scores17[a]) < kpt_thr or float(scores17[b]) < kpt_thr:
            continue
        ax, ay = int(kpts17_xy[a, 0]), int(kpts17_xy[a, 1])
        bx, by = int(kpts17_xy[b, 0]), int(kpts17_xy[b, 1])
        cv2.line(out, (ax, ay), (bx, by), (0, 255, 0), thickness)

    return out


def main():
    # ==== INPUT VIDEO ====
    video_path = Path("/mnt/c/Users/giova/OneDrive/Desktop/tesi/project/data/output/videos/yolov8_gaussian_conf05/soggetto004/004_4sst_4.mp4")

    # ==== OUTPUT ====
    out_jsonl = Path("/home/giovanni/sapiens_pose/pose17_yolo.jsonl")
    out_video = Path("/home/giovanni/sapiens_pose/pose_overlay_17_yolo.mp4")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # ==== MODELS ====
    yolo_weights = "yolov8n.pt"
    SAPIENS_HOST = Path(os.environ.get("SAPIENS_HOST", str(Path.home() / "sapiens_pose" / "sapiens_host")))
    sapiens_ckpt = SAPIENS_HOST / "pose" / "checkpoints" / "sapiens_0.3b" / "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"

    # ==== PARAMS ====
    in_w, in_h = 768, 1024
    frame_stride = 10
    yolo_conf = 0.30
    pad = 0.20

    # Skeleton rendering
    kpt_thr = 0.35
    radius = 5
    thickness = 2

    # 308 -> 17 mapping (iniziale, da correggere se serve)
    KP17_IDXS = list(range(17))

    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    if not sapiens_ckpt.exists():
        raise FileNotFoundError(f"Sapiens checkpoint non trovato: {sapiens_ckpt}")

    print("[INFO] Loading YOLO:", yolo_weights)
    yolo = YOLO(yolo_weights)

    print("[INFO] Loading Sapiens:", sapiens_ckpt)
    sapiens = torch.jit.load(str(sapiens_ckpt), map_location=torch.device("cpu"))
    sapiens.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire il video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps / frame_stride, (W, H))

    print(f"[INFO] Video: {video_path.name} {W}x{H} fps={fps:.2f} frames={nframes}")
    print(f"[INFO] stride={frame_stride} yolo_conf={yolo_conf} pad={pad} kpt_thr={kpt_thr}")

    prev_box = None
    kept = 0

    with out_jsonl.open("w", encoding="utf-8") as fjson:
        pbar = tqdm(total=nframes, desc="Pose (YOLO+Sapiens)")
        frame_idx = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            pbar.update(1)

            if frame_idx % frame_stride != 0:
                continue

            # YOLO person bbox
            yres = yolo.predict(frame, conf=yolo_conf, imgsz=640, verbose=False)[0]
            box_xyxy, box_conf = pick_person_box(yres, prev_box=prev_box, iou_bias=0.7)

            if box_xyxy is None:
                rec = {
                    "frame": frame_idx,
                    "time_sec": float(frame_idx / fps),
                    "person_box_xyxy": None,
                    "person_box_conf": None,
                    "status": "no_person",
                }
                fjson.write(json.dumps(rec) + "\n")
                writer.write(frame)
                kept += 1
                continue

            prev_box = box_xyxy
            box_xyxy = clamp_box_xyxy(box_xyxy, W, H)
            box_xyxy = add_padding_xyxy(box_xyxy, W, H, pad=pad)
            x1, y1, x2, y2 = box_xyxy

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                rec = {
                    "frame": frame_idx,
                    "time_sec": float(frame_idx / fps),
                    "person_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "person_box_conf": box_conf,
                    "status": "empty_crop",
                }
                fjson.write(json.dumps(rec) + "\n")
                writer.write(frame)
                kept += 1
                continue

            # Sapiens on crop
            x = preprocess_bgr(crop, in_w, in_h)
            with torch.no_grad():
                y = sapiens(x)

            if not torch.is_tensor(y) or y.ndim != 4:
                raise RuntimeError(f"Output Sapiens non atteso: type={type(y)}, shape={getattr(y,'shape',None)}")

            heatmaps = y[0].detach().cpu().numpy()   # [308, 256, 192]
            K = int(heatmaps.shape[0])

            kpts_in, scores = heatmaps_to_keypoints(heatmaps, out_w=in_w, out_h=in_h)

            # select 17
            k17 = kpts_in[KP17_IDXS].copy()
            s17 = scores[KP17_IDXS].copy()

            # map to frame coords
            cw, ch = (x2 - x1), (y2 - y1)
            k17[:, 0] *= (cw / in_w)
            k17[:, 1] *= (ch / in_h)
            k17[:, 0] += x1
            k17[:, 1] += y1

            # draw overlay
            vis = frame.copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            vis = draw_coco17(vis, k17, s17, kpt_thr=kpt_thr, radius=radius, thickness=thickness)

            writer.write(vis)

            rec = {
                "frame": frame_idx,
                "time_sec": float(frame_idx / fps),
                "person_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "person_box_conf": box_conf,
                "num_keypoints_total": int(K),
                "kp17_indices": KP17_IDXS,
                "keypoints17": [[float(k17[i,0]), float(k17[i,1]), float(s17[i])] for i in range(17)],
                "status": "ok",
            }
            fjson.write(json.dumps(rec) + "\n")
            kept += 1

        pbar.close()

    cap.release()
    writer.release()
    print("[DONE] Frames written:", kept)
    print("[DONE] JSONL:", out_jsonl)
    print("[DONE] Overlay video:", out_video)
    print("[NOTE] Se lo skeleton è strano, cambiamo KP17_IDXS (mapping 308->17).")


if __name__ == "__main__":
    main()
