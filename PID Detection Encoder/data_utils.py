import glob, re, random
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import AutoImageProcessor, SamImageProcessor

class PIDObjects(Dataset):
    def __init__(self, img_dir: str, crop_size: int = 224):
        self.img_paths = glob.glob(str(Path(img_dir) / "*.png"))
        self.labels    = [int(re.findall(r"(\d+)(?:_[a-z])?\.png", p)[0])
                          for p in self.img_paths]
        self.size = crop_size

    def __len__(self):  
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return img, self.labels[idx]

class UnifiedPreprocessor:
    """
    Lightweight wrapper around either the SAM or DINOv2 HF image‑processor.
    * Accepts a list of PIL images **or** torch tensors (CHW or HWC).
    * Handles resize / normalise once – no torchvision‑v2 needed.
    """
    _SIZES = {"sam": 1024, "dinov2": 224}

    def __init__(self, backbone: str, crop_size: Optional[int] = None):
        name = backbone.lower()
        if name == "sam":
            self.proc = SamImageProcessor.from_pretrained("facebook/sam-vit-base")
        elif name == "dinov2":
            self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        else:
            raise ValueError("backbone must be 'sam' or 'dinov2'")

        tgt = crop_size or self._SIZES[name]
        # unify resize semantics across processors
        if hasattr(self.proc, "size"):
            if "longest_edge" in self.proc.size:
                self.proc.size = {"longest_edge": tgt}
            else:
                self.proc.size = {"height": tgt, "width": tgt}
            self.proc.do_resize = True
        elif hasattr(self.proc, "crop_size"):        # SAM
            self.proc.crop_size = {"height": tgt, "width": tgt}
            self.proc.do_resize = True

        # we’ll let the processor normalise and convert to CHW float‑tensors
        self.proc.do_rescale   = True
        self.proc.do_normalize = True

    def __call__(self, images: List[Union[Image.Image, np.ndarray, torch.Tensor]]) -> torch.Tensor:
        
        prepped = []
        for im in images:
            if isinstance(im, list):
                prepped.extend(im)
            else:
                prepped.append(im)

        cooked: List[Union[Image.Image, np.ndarray]] = []
        for im in prepped:
            if isinstance(im, torch.Tensor):
                raise TypeError("Only accepts PIL or numpy arrays")
            cooked.append(im)

        return self.proc(cooked, return_tensors="pt").pixel_values

class AugmentedViewGenerator:
    def __init__(self, transform, n_views=2):
        """
        Doing the augmentation *outside* the dataset so each epoch gets fresh random augmentations
        Parameters:
            transform (callable): Augmentation pipeline to apply to images
            n_views (int): Number of augmented image views to return
        """
        self.transform = transform
        self.n_views = n_views

    def __call__(self, img):
        return [self.transform(img) for _ in range(self.n_views)]
    

def make_view_collate(aug_gen):          # ‹– factory
    """
    Returns a collate_fn that captures aug_gen inside a closure.
    """
    def view_collate(batch):
        imgs, labels = zip(*batch)       
        views = [aug_gen(img) for img in imgs]
        labels  = torch.tensor(labels)
        return views, labels

    return view_collate 

def img_augmentation(img, size: int = 224) -> np.ndarray:
    
    augmentation_pipeline =  T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomGrayscale(p=0.25),
        T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR)
    ])

    return augmentation_pipeline(img)           






