# YOLOHeatmapFusion.py
"""
YOLO‑v8 × CountGD heat‑map fusion
================================
* **Full image**   → `YOLOHeatmapFusion.predict(imgs, centroids_list)`
* **SAHI slicing** →  

    ```python
    from yolo_heatmap_fusion import (
        YOLOHeatmapFusion,
        register_sahi_fusion,
    )
    register_sahi_fusion()                                   # 1️⃣ once
    fusion = YOLOHeatmapFusion("best.pt", "size_priors.npy",
                               num_classes=80)               # 2️⃣ core
    model = AutoDetectionModel.from_pretrained(              # 3️⃣ SAHI
        model_type="yolov8_fusion",
        model_path="best.pt",
        confidence_threshold=0.01,
        device="cuda:0",
        fusion=fusion,                                       # pass fusion
        centroids_map=json.load(open("centroids.json")),     # pre‑computed
    )
    ```

Everything else in your SAHI workflow (slice size, overlap, export, …) stays
unchanged.  No re‑training, no weight updates – the fusion is pure forward‑pass
tensor math.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# Helper – map global CountGD centroids to *slice‑local* pixel coords
# ──────────────────────────────────────────────────────────────────────────────
def centroids_for_slice(
    centroids_full: np.ndarray,               # (K,3) – [x, y, class]
    slice_bbox: Tuple[int, int, int, int],    # (x0, y0, w, h)
) -> np.ndarray:
    """Crop and shift full‑image centroids to the slice coordinate frame."""
    if centroids_full.size == 0:
        return centroids_full

    x0, y0, w, h = slice_bbox
    x1, y1 = x0 + w, y0 + h
    keep = (
        (centroids_full[:, 0] >= x0)
        & (centroids_full[:, 0] < x1)
        & (centroids_full[:, 1] >= y0)
        & (centroids_full[:, 1] < y1)
    )
    if not keep.any():
        return np.empty((0, 3), dtype=np.float32)

    cent = centroids_full[keep].copy()
    cent[:, 0] -= x0
    cent[:, 1] -= y0
    return cent.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Light GPU‑friendly ops
# ──────────────────────────────────────────────────────────────────────────────
def _gaussian_blur(heat: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Cheap separable blur approximated with a k×k mean filter."""
    pad = k // 2
    return torch.nn.functional.avg_pool2d(
        heat.unsqueeze(0), kernel_size=k, stride=1, padding=pad
    ).squeeze(0)


def _find_peaks(heat: torch.Tensor, tau: float) -> torch.Tensor:
    """Local maxima > tau, returned as (N,2) tensor of [gy, gx]."""
    max_filt = torch.nn.functional.max_pool2d(heat.unsqueeze(0), 3, 1, 1)
    mask = (heat == max_filt.squeeze(0)) & (heat > tau)
    ys, xs = torch.where(mask)
    return torch.stack((ys, xs), dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Main fusion class (unchanged core inference logic)
# ──────────────────────────────────────────────────────────────────────────────
class YOLOHeatmapFusion:
    """Fuse CountGD centroids with a frozen YOLO‑v8 detector."""

    def __init__(
        self,
        model_path: str | Path,
        size_prior_path: str | Path,
        num_classes: int,
        beta: float = 3.0,                   # strength of logit boost
        tau: float = 0.30,                   # peak threshold for new boxes
        device: str | torch.device | None = None,
        stride: int | None = None,           # override if you fuse on P4/P5
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = YOLO(model_path).to(self.device)
        self.model.fuse()

        self.beta, self.tau = beta, tau
        self.num_classes = num_classes
        self.size_prior = torch.from_numpy(np.load(size_prior_path)).to(self.device)  # (C,2)

        # detect head stride (P3 = 8 px for v8)
        dummy = torch.zeros(1, 3, 640, 640, device=self.device)
        _ = self.model(dummy, verbose=False)
        self.stride = stride or int(self.model.model.stride.max())

    # ─────────────────────────── full‑image prediction ───────────────────────
    def predict(
        self,
        imgs: List[str | np.ndarray | torch.Tensor],
        centroids_list: List[np.ndarray],          # list of (K,3) arrays
        conf_yolo: float = 0.01,
        iou_thresh: float = 0.45,
    ) -> List[Dict[str, Any]]:
        assert len(imgs) == len(centroids_list), "imgs and centroids_list length mismatch"

        results = self.model.predict(imgs, conf=conf_yolo, verbose=False)
        outs: List[Dict[str, Any]] = []

        for res, c_np in zip(results, centroids_list):
            boxes = res.boxes.xyxy.to(self.device)
            obj   = res.boxes.obj.to(self.device)
            cls_raw = res.boxes.cls.long()               # (M,)

            H, W  = res.orig_shape
            gh, gw = H // self.stride, W // self.stride
            C = self.num_classes

            # 1.  heat‑map per class
            if c_np.size == 0:
                heat = torch.zeros(C, gh, gw, device=self.device)
            else:
                ct = torch.from_numpy(c_np).to(self.device)         # (K,3)
                gx = (ct[:, 0] / self.stride).long().clamp_(0, gw - 1)
                gy = (ct[:, 1] / self.stride).long().clamp_(0, gh - 1)
                heat = torch.zeros(C, gh, gw, device=self.device)
                heat.index_put_((ct[:, 2].long(), gy, gx),
                                torch.ones_like(gx, dtype=heat.dtype),
                                accumulate=True)
                heat = _gaussian_blur(heat)
                heat /= (heat.amax(dim=(1, 2), keepdim=True) + 1e-6)

            # 2.  boost objectness for existing YOLO boxes
            logits = torch.logit(obj.clamp(1e-6, 1 - 1e-6))
            cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / self.stride).long().clamp_(0, gw - 1)
            cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / self.stride).long().clamp_(0, gh - 1)
            logits += self.beta * heat[cls_raw, cy, cx]
            obj_fused = torch.sigmoid(logits)

            # 3.  add synthetic anchors where YOLO missed
            synth_b, synth_s, synth_c = [], [], []
            for c in range(C):
                for gy, gx in _find_peaks(heat[c], self.tau).tolist():
                    # skip if YOLO already has a box of this class in the cell
                    if ((cls_raw == c) & (cx == gx) & (cy == gy)).any():
                        continue
                    cx_px, cy_px = (gx + 0.5) * self.stride, (gy + 0.5) * self.stride
                    w, h = self.size_prior[c]
                    synth_b.append([cx_px - w / 2, cy_px - h / 2,
                                    cx_px + w / 2, cy_px + h / 2])
                    synth_s.append(float(self.beta))
                    synth_c.append(c)

            if synth_b:
                boxes_all  = torch.cat([boxes, torch.stack(synth_b)])
                scores_all = torch.cat([obj_fused,
                                        torch.sigmoid(torch.tensor(synth_s, device=self.device))])
                cls_all    = torch.cat([cls_raw, torch.tensor(synth_c, device=self.device)])
            else:
                boxes_all, scores_all, cls_all = boxes, obj_fused, cls_raw

            keep = batched_nms(boxes_all, scores_all, cls_all, iou_thresh)
            outs.append({
                "boxes":   boxes_all[keep].cpu().numpy(),
                "scores":  scores_all[keep].cpu().numpy(),
                "classes": cls_all[keep].cpu().numpy(),
            })
        return outs


# ──────────────────────────────────────────────────────────────────────────────
# SAHI integration – register once, then use model_type="yolov8_fusion"
# ──────────────────────────────────────────────────────────────────────────────
def register_sahi_fusion() -> None:
    """
    After importing this module call `register_sahi_fusion()` **once** before
    you ask `AutoDetectionModel` for `model_type="yolov8_fusion"`.
    """
    try:
        from sahi.models.yolov8 import Yolov8DetectionModel
        from sahi.auto_model import AutoDetectionModel
        from sahi.utils.yolov8 import yolov8_results_to_sahi_predictions
    except ImportError as err:
        raise ImportError("SAHI not installed – `pip install sahi`") from err

    class YoloSahiFusion(Yolov8DetectionModel):
        """Inject heat‑map fusion into every SAHI slice."""

        def __init__(
            self,
            fusion: YOLOHeatmapFusion,
            centroids_map: Dict[str, np.ndarray],     # filename → (K,3)
            *args,
            **kwargs,
        ):
            # pop kwargs SAHI passes to Yolov8DetectionModel
            super().__init__(*args, **kwargs)
            self.fusion = fusion
            self.centroids_map = centroids_map

        # SAHI calls `_predict` per *slice*
        def _predict(
            self,
            image: np.ndarray,
            slice_bbox: Tuple[int, int, int, int],
            full_shape=None,
            image_id=None,
            full_image_path: str | None = None,
            **kwargs,
        ):
            if full_image_path is None:
                raise ValueError("SAHI must pass full_image_path")

            global_centroids = self.centroids_map.get(
                Path(full_image_path).name, np.empty((0, 3), dtype=np.float32)
            )
            cent_slice = centroids_for_slice(global_centroids, slice_bbox)
            det = self.fusion.predict([image], [cent_slice])[0]

            # convert numpy arrays → SAHI Prediction objects
            return yolov8_results_to_sahi_predictions(
                boxes_xyxy=torch.from_numpy(det["boxes"]),
                scores=torch.from_numpy(det["scores"]),
                class_ids=torch.from_numpy(det["classes"]),
                original_height=image.shape[0],
                original_width=image.shape[1],
            )

    # make AutoDetectionModel aware of the new model type
    AutoDetectionModel._model_type_class_map["yolov8_fusion"] = YoloSahiFusion


# ──────────────────────────────────────────────────────────────────────────────
# Quick CLI check on full images (no SAHI)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser("YOLO + CountGD fusion (full images)")
    parser.add_argument("model")
    parser.add_argument("size_priors")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--centroids", required=True,
                        help="json mapping filename → [[x,y,class], …]")
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--tau",  type=float, default=0.3)
    args = parser.parse_args()

    with open(args.centroids) as f:
        centroids_dict = json.load(f)

    all_cls_ids = {c for lst in centroids_dict.values() for *_, c in lst}
    fusion = YOLOHeatmapFusion(args.model, args.size_priors,
                               num_classes=len(all_cls_ids),
                               beta=args.beta, tau=args.tau)

    outs = fusion.predict(
        args.images,
        [np.array(centroids_dict.get(Path(p).name, []), dtype=np.float32) for p in args.images],
    )
    for img, det in zip(args.images, outs):
        print(f"{Path(img).name}: {len(det['boxes'])} boxes")
