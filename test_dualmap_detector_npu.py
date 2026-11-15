#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Board test: full DualMap Detector tick on the NPU path.

Composes DualMap's hydra config exactly like Scene does (same overrides as
``perception_dualmap._load_dualmap`` with ``SCENE_DUALMAP_NPU_DIR`` set),
runs one synthetic RGB-D frame through detector + local map and prints the
observations. Validates the integration end to end before the service runs.

    python3 test_dualmap_detector_npu.py --image IMG --dualmap-root /root/dualmap-npu \
        --npu-dir model/npu --classes scene_eval_classes.txt --clip model/ViT-B-32.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--dualmap-root", default="/root/dualmap-npu")
    ap.add_argument("--npu-dir", default="model/npu")
    ap.add_argument("--sam", default="model/mobile_sam.pt")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--clip", required=True)
    args = ap.parse_args()

    root = os.path.abspath(args.dualmap_root)
    sys.path.insert(0, root)
    npu_dir = os.path.abspath(args.npu_dir)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    out_dir = os.path.join(tempfile.gettempdir(), "scene-dualmap-test")
    os.makedirs(out_dir, exist_ok=True)
    overrides = [
        "use_rerun=false", "use_parallel=false", "run_local_mapping_only=true",
        "save_local_map=false", "save_global_map=false", "save_detection=false",
        "visualize_detection=false", "run_detection=true", "device=cpu",
        f"output_path={out_dir}", f"sam.model_path={os.path.abspath(args.sam)}",
        "use_fastsam=false",
        "use_npu=true", f"npu_model_dir={npu_dir}",
        "clip.model_name=ViT-B-32", f"clip.pretrained={os.path.abspath(args.clip)}",
        "yolo.use_given_classes=true", f"yolo.given_classes_path={os.path.abspath(args.classes)}",
        f"logging_config={os.path.join(root, 'config', 'support_config', 'logging_config.yaml')}",
        f"config_file_path={os.path.join(root, 'config', 'actions.yaml')}",
    ]
    with initialize_config_dir(version_base=None, config_dir=os.path.join(root, "config")):
        cfg = compose(config_name="runner_dataset", overrides=overrides)

    from utils.global_map_manager import GlobalMapManager
    from utils.local_map_manager import LocalMapManager
    from utils.object_detector import Detector
    from utils.types import DataInput
    from utils.visualizer import ReRunVisualizer

    vis = ReRunVisualizer(cfg)
    vis.set_use_rerun(False)
    t0 = time.time()
    det = Detector(cfg)
    lm = LocalMapManager(cfg)
    print(f"[test] detector+map init: {time.time()-t0:.1f}s")
    print(f"[test] filter={hasattr(det,'filter')} sam={hasattr(det,'sam')} "
          f"yolo={hasattr(det,'yolo')} clip_npu={det.clip_image_npu is not None}")

    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    depth = np.full((h, w, 1), 2.5, dtype=np.float32)  # flat 2.5 m plane
    K = np.array([[500.0, 0, w / 2], [0, 500.0, h / 2], [0, 0, 1]])
    data = DataInput(idx=0, time_stamp=time.time(), color=np.ascontiguousarray(img),
                     depth=depth, color_name="test0", intrinsics=K, pose=np.eye(4))

    t0 = time.time()
    det.set_data_input(data)
    det.process_detections()
    det.calculate_observations()
    obs = det.get_curr_observations()
    det.update_state()
    det.update_data()
    lm.set_curr_idx(0)
    lm.process_observations(obs)
    print(f"[test] tick: {time.time()-t0:.2f}s -> {len(obs)} observations, "
          f"{len(lm.local_map)} local-map tracks")
    for o in obs[:10]:
        name = det.obj_classes.get_classes_arr()[o.class_id] if o.class_id < len(det.obj_classes.get_classes_arr()) else "?"
        print(f"  obs: {name:12s} conf={o.conf:.3f} pts={len(o.pcd.points)} "
              f"bbox_z={o.bbox.get_center()[2]:.2f} low_mobility={o.is_low_mobility}")


if __name__ == "__main__":
    main()
