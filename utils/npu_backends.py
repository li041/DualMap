# SPDX-License-Identifier: Apache-2.0
"""RKNPU3 (RK1828) inference backends for DualMap's per-frame models.

Replaces the heavy torch models inside ``Detector`` when ``use_npu=True``:
the YOLO-World detector (airockchip 2-input graph, text embeddings
precomputed), MobileSAM's image encoder and prompt/mask decoder, and the
CLIP image tower (openai ViT-B/32). Letterbox/NMS/mask post-processing runs
on the CPU in numpy/torch, mirroring ultralytics' preprocessing exactly.

Artifacts (``<dir>/``): ``yoloworld.rknn``/``.weight`` (fp16),
``yoloworld_text.npy`` ([1,80,512] L2-normalized), ``mobilesam.rknn``/
``.weight`` (fp16), ``mobilesam_decoder.rknn``/``.weight`` (fp16),
``clip_image.rknn``/``.weight`` (fp16), ``npu_meta.json`` (nc + class list
the text embeddings were built for).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# torch 2.6 changed `torch.load(weights_only=True)` to the default, and
# open_clip + ultralytics' older checkpoints (jit pickles, custom types) get
# rejected. The board's PyTorch is 2.7+; restore the legacy behaviour so
# `open_clip.create_model_and_transforms(pretrained=...)` and ultralytics'
# SAM loaders succeed. Apply this unconditionally so the CLIP load that
# runs in __init__ before the NPU guard also sees it.
_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

# Torch still handles CLIP text encoding and mask upsampling on the CPU.
if os.environ.get("OMP_NUM_THREADS") is None and torch.get_num_threads() < 4:
    torch.set_num_threads(max(4, min(8, os.cpu_count() or 4)))

_PAD = (114, 114, 114)  # ultralytics LetterBox default


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Ultralytics SAM/YOLO letterbox: scale to fit, pad bottom/right
    (center=False), no stride rounding beyond ``size``."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nu = (int(round(w * r)), int(round(h * r)))
    out = cv2.resize(img, nu, interpolation=cv2.INTER_LINEAR)
    dh, dw = size - nu[1], size - nu[0]
    return cv2.copyMakeBorder(out, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=_PAD), r


class _RKNN:
    """One loaded rknn3lite model; the board device is single-tenant."""

    _lock = threading.Lock()

    def __init__(self, rknn_path: str, weight_path: str, name: str) -> None:
        from rknn3lite.api import RKNN3Lite

        self.rk = RKNN3Lite()
        t0 = time.time()
        if self.rk.load_rknn(model_path=rknn_path, weight_path=weight_path) != 0:
            raise RuntimeError(f"[npu:{name}] load_rknn failed")
        devices = self.rk.get_devices_id()
        if not devices:
            raise RuntimeError(f"[npu:{name}] no RKNPU3 device found")
        if self.rk.init_runtime(target="rk1820", core_mask=0x01, device_id=devices[0]) != 0:
            raise RuntimeError(f"[npu:{name}] init_runtime failed")
        self.name = name
        self.last_ms = 0.0
        logger.info("[npu:%s] loaded in %.1fs (device %s)", name, time.time() - t0, devices[0])

    def inference(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        with _RKNN._lock:
            t0 = time.time()
            outs = self.rk.inference(inputs=inputs)
            self.last_ms = (time.time() - t0) * 1000.0
        if outs is None:
            raise RuntimeError(f"[npu:{self.name}] inference failed")
        return outs

    def release(self) -> None:
        try:
            self.rk.release()
        except Exception:  # noqa: BLE001
            pass


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.as_tensor(xyxy, dtype=torch.float32)
        self.conf = torch.as_tensor(conf, dtype=torch.float32)
        self.cls = torch.as_tensor(cls, dtype=torch.float32)


class _Result:
    def __init__(self, xyxy, conf, cls):
        self.boxes = _Boxes(xyxy, conf, cls)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(ovr <= iou_th)[0] + 1]
    return keep


class YOLOWorldNPU:
    """Fixed-vocabulary YOLO-World graphs, two bundle formats:

    - ``npu_meta.json`` without ``format`` (zoo bundles): two-input airockchip
      graph (image float NCHW 0-255, text embeds), six per-FPN-level outputs
      (cls sigmoid, ltrb box).
    - ``"format": "pre_nms"`` (ultralytics self-export): ``set_classes``
      vocabulary baked into the weights, single uint8 NHWC input, one pre-NMS
      output [1,4+nc,8400] (cx,cy,w,h in the 640 frame + sigmoided scores).

    ``predict()`` mimics the ultralytics results DualMap consumes; the
    vocabulary is fixed at compile time (``npu_meta.json`` records it)."""

    def __init__(self, model_dir: str, imgsz: int = 640, max_prompts: int = 80) -> None:
        self.imgsz = imgsz
        self.max_prompts = max_prompts
        with open(os.path.join(model_dir, "npu_meta.json")) as fh:
            meta = json.load(fh)
        self.names = list(meta["classes"])
        self.nc = len(self.names)
        self.pre_nms = meta.get("format") == "pre_nms"
        if self.pre_nms:
            # set_classes baked the text embeddings into the weights; the
            # graph has no text input and no text sidecar exists.
            self.text = None
        else:
            text = np.load(os.path.join(model_dir, "yoloworld_text.npy")).astype(np.float32)
            if text.ndim == 2:
                text = text[None]
            if text.shape[1] < self.nc:
                raise RuntimeError(f"text embeddings cover {text.shape[1]} < {self.nc} classes")
            # Rows past the vocabulary are padding slots; blank them like the
            # validated artifact does.
            text[:, self.nc:] = 0.0
            self.text = text
        self.rknn = _RKNN(os.path.join(model_dir, "yoloworld.rknn"),
                          os.path.join(model_dir, "yoloworld.weight"), "yoloworld")

    def predict(self, color: np.ndarray, conf: float = 0.2, iou: float = 0.45,
                verbose: bool = False) -> list:
        r = min(self.imgsz / color.shape[0], self.imgsz / color.shape[1])
        lb, _ = _letterbox(color, self.imgsz)
        if self.pre_nms:
            # uint8 NHWC per the input attr; the build config absorbs /255.
            out = np.asarray(self.rknn.inference([lb[None]])[0], dtype=np.float32)[0]
            cx, cy, bw, bh = out[:4]
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
            cls_raw = out[4:self.nc + 4]
            best = cls_raw.max(axis=0)
            best_cls = cls_raw.argmax(axis=0)
            pos = best >= conf
            if not pos.any():
                return [_Result(np.zeros((0, 4)), np.zeros(0), np.zeros(0))]
            boxes, scores, classes = boxes[pos], best[pos], best_cls[pos]
        else:
            x = lb.astype(np.float32).transpose(2, 0, 1)[None]
            outs = self.rknn.inference([x, self.text])

            boxes_all, scores_all, cls_all = [], [], []
            for lvl in range(3):
                cls_map = np.asarray(outs[2 * lvl], dtype=np.float32)     # (1,80,h,w)
                box_map = np.asarray(outs[2 * lvl + 1], dtype=np.float32)  # (1,4,h,w)
                _, _, h, w = cls_map.shape
                stride = self.imgsz / h
                col, row = np.meshgrid(np.arange(w), np.arange(h))
                grid = np.stack([col, row], axis=0).astype(np.float32)  # (2,h,w)
                ctr = grid + 0.5
                xyxy = np.concatenate([(ctr - box_map[0, 0:2]) * stride,
                                       (ctr + box_map[0, 2:4]) * stride], axis=0)  # (4,h,w)
                scores = cls_map[0, :self.nc].reshape(self.nc, -1)  # padded slots excluded
                best = scores.max(axis=0)
                best_cls = scores.argmax(axis=0)
                pos = best >= conf
                if pos.any():
                    boxes_all.append(xyxy.reshape(4, -1).T[pos])
                    scores_all.append(best[pos])
                    cls_all.append(best_cls[pos])
            if not boxes_all:
                return [_Result(np.zeros((0, 4)), np.zeros(0), np.zeros(0))]
            boxes = np.concatenate(boxes_all)
            scores = np.concatenate(scores_all)
            classes = np.concatenate(cls_all)
        keep_box, keep_conf, keep_cls = [], [], []
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            k = _nms(boxes[idx], scores[idx], iou)
            keep_box.append(boxes[idx][k])
            keep_conf.append(scores[idx][k])
            keep_cls.append(classes[idx][k])
        boxes = np.concatenate(keep_box)
        scores = np.concatenate(keep_conf)
        classes = np.concatenate(keep_cls)
        boxes /= r  # letterbox -> original pixels
        # ultralytics scale_boxes clamps boxes into the frame; without it a
        # stray box makes the proximity filter slice an empty crop and divide
        # a zero-norm histogram (NaN). Drop boxes the clamp leaves degenerate.
        h, w = color.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        ok = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        if not ok.all():
            boxes, scores, classes = boxes[ok], scores[ok], classes[ok]
        return [_Result(boxes, scores, classes)]


class _Masks:
    def __init__(self, data: torch.Tensor):
        self.data = data


class _SamResult:
    def __init__(self, masks: torch.Tensor):
        self.masks = _Masks(masks)


class MobileSamNPU:
    """MobileSAM with the TinyViT encoder AND the prompt/mask decoder on
    the NPU (the decoder graph is SamOnnxModel @1024 with the mask resize
    cut - it emits 256x256 low_res logits, token 0 = the single-mask head).
    Mirrors ultralytics' ``SAM.predict(color, bboxes=...)``: LetterBox(1024,
    center=False), mean/std normalization (absorbed by the graph), box
    prompts fed as two corner points with labels (2, 3), logits bilinearly
    resized to the original image on CPU, threshold 0. No Torch SAM model or
    checkpoint is loaded on this path."""

    def __init__(self, model_dir: str, sam_model=None) -> None:
        self.imgsz = 1024
        self.rknn = _RKNN(os.path.join(model_dir, "mobilesam.rknn"),
                          os.path.join(model_dir, "mobilesam.weight"), "mobilesam")
        self.rknn_dec = _RKNN(os.path.join(model_dir, "mobilesam_decoder.rknn"),
                              os.path.join(model_dir, "mobilesam_decoder.weight"),
                              "mobilesam_decoder")

    def predict(self, color: np.ndarray, bboxes: np.ndarray, verbose: bool = False) -> list:
        h, w = color.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        lb, _ = _letterbox(color, self.imgsz)
        emb = np.asarray(self.rknn.inference([np.expand_dims(lb, 0)])[0], dtype=np.float32)

        n = len(bboxes)
        if n == 0:
            return [_SamResult(torch.zeros((0, h, w)))]
        # one box = two corner points with labels (2, 3), in the 1024 frame
        low_res = []
        for x1, y1, x2, y2 in np.asarray(bboxes, dtype=np.float32).reshape(-1, 4) * r:
            coords = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
            out = self.rknn_dec.inference([
                emb, coords, np.array([[2.0, 3.0]], dtype=np.float32),
                np.zeros((1, 1, 256, 256), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
            ])
            low_res.append(torch.from_numpy(np.asarray(out[0], dtype=np.float32)[:, :1]))
        low = torch.cat(low_res)
        # The decoder sees the whole 1024x1024 letterbox; cut the padding back
        # off (at the logits' own resolution) before resizing, or the mask
        # lands squeezed into the content's fraction of the frame (580/1024 of
        # the height on room0's 680x1200).
        _, _, lh, lw = low.shape
        low = low[:, :, :round(lh * h * r / self.imgsz), :round(lw * w * r / self.imgsz)]
        logits = torch.nn.functional.interpolate(low, size=(h, w),
                                                 mode="bilinear", align_corners=False)
        return [_SamResult((logits[:, 0] > 0.0))]


class ClipImageNPU:
    """Openai CLIP ViT-B/32 image tower on the NPU, one crop per call.
    Preprocessing matches open_clip's transform for this arch: bicubic
    resize of the shorter side to 224, center crop, CLIP normalization
    (absorbed by the graph's mean/std on the uint8 input)."""

    SIZE = 224

    def __init__(self, model_dir: str) -> None:
        self.rknn = _RKNN(os.path.join(model_dir, "clip_image.rknn"),
                          os.path.join(model_dir, "clip_image.weight"), "clip_image")

    def encode_crops(self, image_rgb: np.ndarray, xyxy: np.ndarray,
                     padding: int = 20) -> np.ndarray:
        """Features for each (padded) crop, matching DualMap's crop logic;
        returns unnormalized [N,512] (the caller L2-normalizes)."""
        h, w = image_rgb.shape[:2]
        feats = []
        for x1, y1, x2, y2 in np.asarray(xyxy, dtype=np.float32).reshape(-1, 4):
            x1 = max(0.0, x1 - min(padding, x1))
            y1 = max(0.0, y1 - min(padding, y1))
            x2 = min(float(w), x2 + min(padding, w - x2))
            y2 = min(float(h), y2 + min(padding, h - y2))
            crop = image_rgb[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                feats.append(np.zeros(512, dtype=np.float32))
                continue
            ch, cw = crop.shape[:2]
            r = self.SIZE / min(ch, cw)
            nu = (max(1, int(round(cw * r))), max(1, int(round(ch * r))))
            rs = cv2.resize(crop, nu, interpolation=cv2.INTER_CUBIC)
            rh, rw = rs.shape[:2]
            top, left = (rh - self.SIZE) // 2, (rw - self.SIZE) // 2
            rs = rs[top:top + self.SIZE, left:left + self.SIZE]
            if rs.shape[0] != self.SIZE or rs.shape[1] != self.SIZE:
                rs = cv2.resize(rs, (self.SIZE, self.SIZE), interpolation=cv2.INTER_CUBIC)
            out = self.rknn.inference([np.expand_dims(rs, 0)])[0]
            feats.append(np.asarray(out, dtype=np.float32).reshape(-1))
        return np.stack(feats) if feats else np.zeros((0, 512), dtype=np.float32)
