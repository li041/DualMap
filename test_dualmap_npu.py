#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""End-to-end smoke test of DualMap's NPU detector path on the board.

Loads the three RKNPU3 backends, runs one image through
YOLO-World -> MobileSAM -> CLIP crops and prints detections, mask stats and
timings. Run from the DualMap checkout root:

    python3 scripts/test_dualmap_npu.py --image IMG --model-dir model/npu
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model-dir", default="model/npu")
    ap.add_argument("--sam-weights", default="model/mobile_sam.pt")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    from utils.npu_backends import ClipImageNPU, MobileSamNPU, YOLOWorldNPU

    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    print(f"image {img.shape}")

    yolo = YOLOWorldNPU(args.model_dir)
    sam = MobileSamNPU(args.model_dir, args.sam_weights)
    clip = ClipImageNPU(args.model_dir)

    t0 = time.time()
    res = yolo.predict(img, conf=args.conf)
    t_yolo = time.time() - t0
    r = res[0].boxes
    print(f"yolo: {len(r.conf)} boxes in {t_yolo*1000:.0f}ms (npu {yolo.rknn.last_ms:.0f}ms)")
    names = yolo.names
    order = np.argsort(-r.conf.numpy())[:10]
    for i in order:
        x1, y1, x2, y2 = r.xyxy.numpy()[i]
        print(f"  {names[int(r.cls.numpy()[i])]:12s} {float(r.conf.numpy()[i]):.3f} "
              f"[{x1:6.1f} {y1:6.1f} {x2:6.1f} {y2:6.1f}]")

    if len(r.conf) == 0:
        print("no detections; stopping")
        return
    t0 = time.time()
    sam_res = sam.predict(img, bboxes=r.xyxy.numpy())
    t_sam = time.time() - t0
    masks = sam_res[0].masks.data.numpy()
    print(f"sam: {masks.shape} in {t_sam*1000:.0f}ms (npu encoder {sam.rknn.last_ms:.0f}ms), "
          f"mask px/box: {[int(m.sum()) for m in masks[:6]]}")
    if args.repeat > 1:  # steady-state decoder timing (first call warms torch up)
        for i in range(args.repeat - 1):
            t0 = time.time()
            sam.predict(img, bboxes=r.xyxy.numpy())
            print(f"sam[{i + 1}]: {(time.time() - t0)*1000:.0f}ms (npu {sam.rknn.last_ms:.0f}ms)")

    t0 = time.time()
    feats = clip.encode_crops(img, r.xyxy.numpy())
    t_clip = time.time() - t0
    fn = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
    print(f"clip: {feats.shape} in {t_clip*1000:.0f}ms ({len(feats)} crops x "
          f"{clip.rknn.last_ms:.0f}ms last), norms ok: {bool(np.allclose(np.linalg.norm(fn, axis=1), 1, atol=1e-3))}")
    print(f"total: {(t_yolo + t_sam + t_clip)*1000:.0f}ms")


if __name__ == "__main__":
    main()
