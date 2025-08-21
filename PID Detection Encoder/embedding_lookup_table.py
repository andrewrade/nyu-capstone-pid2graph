from typing import List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from data_utils import UnifiedPreprocessor
from models import SAMBackbone, DinoV2Backbone


class ObjectLookupTable:
    def __init__(self, labels, gt_image_paths, encoder, img_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.eval().to(self.device)
        self.class_labels = labels
        self.gt_image_paths = gt_image_paths
        self.img_size = img_size

        self.normalize = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.753, 0.753, 0.754],
                         std=[0.333, 0.333, 0.332])
        ])
        self.gt_embeddings = self._get_embeddings(gt_image_paths)

    def _square_pad(self, image):
        _, h, w = image.shape
        if h > self.img_size or w > self.img_size:
            scale = self.img_size / max(h, w)
            h, w = int(h * scale), int(w * scale)
            image = v2.functional.resize(image, (h, w))
        hp = (self.img_size - w) // 2
        vp = (self.img_size - h) // 2
        return v2.functional.pad(image,
                                  (hp, self.img_size - w - hp,
                                   vp, self.img_size - h - vp),
                                  fill=1)

    def _get_embeddings(self, paths):
        embs = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            t = self.normalize(img).unsqueeze(0).to(self.device)
            t = self._square_pad(t.squeeze(0)).unsqueeze(0)
            with torch.no_grad():
                e = self.encoder(t)
            e = e / e.norm(p=2)
            embs.append(e.squeeze(0))
        return torch.stack(embs)

    def classify(self, detection_paths):
        det_emb = self._get_embeddings(detection_paths)
        sim = torch.matmul(det_emb, self.gt_embeddings.T)
        return torch.max(sim, dim=1)



@dataclass
class ClassificationResult:
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    label: str
    score: float

class ObjectLookupTable:
    def __init__(
        self,
        class_labels: List[str],
        legend_image_paths: List[Path],
        backbone: str = "dinov2",
        img_size: int = 224,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = UnifiedPreprocessor(backbone, crop_size=img_size)
        if backbone.lower() == "sam":
            self.encoder = SAMBackbone().to(self.device).eval()
        elif backbone.lower() == "dinov2":
            self.encoder = DinoV2Backbone().to(self.device).eval()
        else:
            raise ValueError("backbone must be 'dinov2' or 'sam'")
        self.class_labels = class_labels
        self._load_legend_embeddings(legend_image_paths)

    def _load_legend_embeddings(self, paths: List[Path]):
        embs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            pix = self.preprocessor([img]).to(self.device)
            with torch.no_grad():
                e = self.encoder(pix)               # [1, D]
            embs.append(F.normalize(e.squeeze(0), dim=0))
        self.legend_embeddings = torch.stack(embs)  # [K, D]

    def classify(
        self,
        drawing: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        top_k: int = 1
    ) -> List[List[ClassificationResult]]:
        """
        Returns, for each bbox, a ranked list of top_k ClassificationResult.
        """
        # Crop & preprocess
        crops = [drawing.crop((x, y, x + w, y + h)) for (x, y, w, h) in bboxes]
        pix   = self.preprocessor(crops).to(self.device)  # [N, C, H, W]

        # Embed
        with torch.no_grad():
            det_emb = self.encoder(pix)                   # [N, D]
        det_emb = F.normalize(det_emb, dim=1)             # [N, D]

        # Compute similarities and take top_k
        sims = det_emb @ self.legend_embeddings.T         # [N, K]
        topk_scores, topk_idxs = sims.topk(top_k, dim=1)  # each [N, top_k]

        all_results = []
        for i in range(len(bboxes)):
            res_i = []
            for score, idx in zip(topk_scores[i], topk_idxs[i]):
                res_i.append(
                    ClassificationResult(
                        bbox=bboxes[i],
                        label=self.class_labels[idx],
                        score=score.item()
                    )
                )
            all_results.append(res_i)

        return all_results

def evaluate_ranked(
    ranked_results: List[List[ClassificationResult]],
    gt_labels: List[str],
    k: int
) -> Tuple[float, float]:
    """
    Compute Recall@k and mAP over the batch.
    - Recall@k = fraction of boxes where GT label ∈ top-k predictions.
    - mAP = mean over boxes of (1 / rank_of_GT), or 0 if not predicted in top-k.
    """
    assert len(ranked_results) == len(gt_labels)
    hits = 0
    ap_sum = 0.0

    for preds, gt in zip(ranked_results, gt_labels):
        labels = [p.label for p in preds]
        if gt in labels:
            hits += 1
            rank = labels.index(gt) + 1
            ap_sum += 1.0 / rank
        else:
            # no hit in top-k, contributes 0
            pass

    recall_at_k = hits / len(gt_labels)
    mAP = ap_sum / len(gt_labels)
    return recall_at_k, mAP

