import glob
import re
import math
import datetime
from pathlib import Path

import numpy as np

from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from torchvision.models import resnet18  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import wandb
from tqdm import tqdm

from transformers import SamModel, SamImageProcessor, AutoModel, AutoImageProcessor

from utils import SquarePad


class LeNetBackbone(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model_name = 'lenet'
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=7, stride=1, padding=3), # 64 x 64 x 6
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) # 32 x 32 x 6
        
        self.layer2 = nn.Sequential(
                nn.Conv2d(6, 10, kernel_size=5, stride=1, padding=0), # 28 x 28 x 10
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2)) # 14 x 14 x 10
    
        self.layer3 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1), # 14 x 14 x 16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((5, 5)) # 5 x 5 x 16
            
    def forward(self, x):
        # Conv layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # Flattened
        out = self.avg_pool(out).reshape(out.size(0), -1)
        return out

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'resnet18'
        self.encoder = resnet18(weights='DEFAULT', zero_init_residual=True)
        self.embedding_dim = self.encoder.fc.in_x
        self.encoder.fc = nn.Identity()
    
    def forward(self, x):
        return self.encoder(x)
             
class SAMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'facebook/sam-vit-huge'
        self.embedding_dim = 256
        self.encoder = SamModel.from_pretrained(self.model_name).vision_encoder

        # Pool from 256 x 64 x 64 --> 256 x 8 x 8
        self.pool_layer = nn.AdaptiveAvgPool2d((8, 8))
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 8 * 8, self.embedding_dim * 8),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(self.embedding_dim * 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim * 8, self.embedding_dim),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Frozen SAM ViT Backbone 
        with torch.no_grad():
            sam_embedding = self.encoder(x)
        
        x = self.pool_layer(sam_embedding).flatten(1)
        x = self.mlp(x)
        
        if self.training:
            return x, sam_embedding
        else:
            return x

class DinoV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'facebook/dinov2-base'
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.dino_embedding_dim = self.encoder.config.hidden_size
        self.embedding_dim = 256 #self.encoder.config.hidden_size 

        self.mlp = nn.Sequential(
            nn.Linear(self.dino_embedding_dim, self.dino_embedding_dim // 2, bias=False),
            nn.BatchNorm1d(self.dino_embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.dino_embedding_dim // 2, self.embedding_dim),
        )
    
    def forward(self, x):
        # Frozen SAM ViT Backbone 
        with torch.no_grad():
            dino_embedding = self.encoder(pixel_values=x).last_hidden_state[:, 0]
        
        x = self.mlp(dino_embedding)
        
        if self.training:
            return x, dino_embedding
        else:
            return x
            
    
class ObjectEncoder(nn.Module):
    """
    Based on SimCLR https://arxiv.org/abs/2002.05709
    Resnet18 Encoder & 2 layer projector for loss during training
    """
    def __init__(self, backbone):
        super().__init__()

        self.embedding_dim = backbone.embedding_dim
        self.backbone = backbone
        
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        
    def forward(self, x):
        """
        Parameters:
            x:  Image
        Returns:
            inference: Encoded x
            training: Projections of encoded X
        """
        if self.training:
                match self.backbone.model_name: 
                    case 'lenet' | 'resnet18' :
                        return self.projector(self.backbone(x))
                    case 'facebook/sam-vit-huge' | 'facebook/dinov2-base':
                        x, frozen_embedding = self.backbone(x)
                        x = self.projector(x)
                        return x, frozen_embedding
                    case _:
                        raise NotImplementedError
        else:
            return self.backbone(x)

            

class AugmentedViewGenerator():
    """ Take two random crops of an image"""
    def __init__(self, transform, n_views=2):
        """
        Parameters:
            transform (callable): Augmentation pipeline to apply to images
            n_views (int): Number of augmented image views to return
        """
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        views = [self.transform(x) for _ in range(self.n_views)]
        return views
    
        
class PIDObjects(Dataset):
    """ Extracted PID detection images """
    def __init__(self, img_dir, transform=None, return_pil=False):
        """
        Parameters:
            img_dir (str): Path where detection images are saved
            transform (callable): Augmentation pipeline to apply to images 
        """
        self.img_dir = img_dir
        self.img_paths = glob.glob(str(Path(self.img_dir) / '*.png'))
        self.img_labels = [int(re.findall('(\d+)(?:_[a-z])?\.png', x)[0]) for x in self.img_paths]
        self.normalize = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.753, 0.753, 0.754],
                         std=[0.333, 0.333, 0.332])
        ])
        self.transform = transform
        self.return_pil=return_pil
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        image = Image.open(img_path)
        # Get rid of alpha channel in png
        image = image.convert("RGB")
        #image = self.normalize(image)
        label = self.img_labels[index]

        if self.transform:
            augmented_views = self.transform(image)
        
        if self.return_pil:
            augmented_views = [to_pil_image(x) for x in augmented_views]
        
        return (*augmented_views, label)

def pil_collate(batch):
    # batch is a list of tuples: (view1_pil, view2_pil, …, label)
    *view_lists, labels = zip(*batch)
    # view_lists[i] is a tuple of B PILs for view i
    # we want each element to be a list of PILs
    views = [list(vl) for vl in view_lists]
    labels = torch.tensor(labels, dtype=torch.long)
    return (*views, labels)

def img_augmentation(img_size=224, blur_kernel_size=5):

    augmentation_pipeline = v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        v2.PILToTensor(),
        v2.RandomRotation(10),
        v2.RandomHorizontalFlip(p=0.25),
        v2.RandomVerticalFlip(p=0.25),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=blur_kernel_size)], 
            p=0.3
        ),
        v2.RandomGrayscale(p=0.25)
    ])

    return augmentation_pipeline
    

def multiclass_nt_xent_loss(x, pos_indices, temperature):
    """
    Based on
    https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
    
    Parameters:
        x (tensor): (Batch x Embedding Size) Feature vectors from encoder
        pos_indices (tensor): (N x 2) where N is number of positive pairs, each (row, col) of similarity matrix
        temperature (float): Temperature value for softargmax 
    """
    device = get_device()
    
    pos_indices = torch.cat([
        pos_indices,
        torch.arange(x.size(0), device=device).reshape(x.size(0), 1).expand(-1, 2),
    ], dim=0)
    
    target = torch.zeros(x.size(0), x.size(0), device=device)
    target[pos_indices[:, 0], pos_indices[:, 1]] = 1.0

    # All Pairs Cosine similarity
    xcs = nn.functional.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    xcs[torch.eye(x.size(0)).bool()] = float("inf")

    loss = nn.functional.binary_cross_entropy((xcs / temp).sigmoid(), target, reduction="none")
    
    target_pos = target.bool()
    target_neg = ~target_pos

    loss_pos = torch.zeros(x.size(0), x.size(0), device=device).masked_scatter(target_pos, loss[target_pos])
    loss_neg = torch.zeros(x.size(0), x.size(0), device=device).masked_scatter(target_neg, loss[target_neg])
    loss_pos = loss_pos.sum(dim=1)
    loss_neg = loss_neg.sum(dim=1)
    
    num_pos = target.sum(dim=1)
    num_neg = x.size(0) - num_pos

    loss = ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()

    return loss

def binary_nt_xent_loss(x, temperature):

    bs = x.size(0) 
    
    # Cosine similarity
    xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    xcs[torch.eye(bs).bool()] = float("-inf")

    # Ground truth labels
    target = torch.arange(bs).to(xcs.device)
    target[0::2] += 1
    target[1::2] -= 1
    
    # Standard cross entropy loss
    return F.cross_entropy(xcs / temp, target, reduction="mean")

def supervised_contrastive_loss(x, labels, temperature):
    """
    x: (N, D) embeddings
    labels:   (N,) integer class labels
    """
    device = x.device
    N = x.size(0)

    # 1) Normalize
    f = F.normalize(x, dim=1)

    # 2) Compute similarity matrix
    logits = torch.matmul(f, f.t()) / temperature

    # 3) Mask out self-similarities
    mask = torch.eye(N, device=device).bool()
    logits_masked = logits.masked_fill(mask, -1e9)

    # 4) Compute log‐probabilities
    exp_logits = torch.exp(logits_masked)
    log_prob = logits_masked - torch.log(exp_logits.sum(dim=1, keepdim=True))

    # 5) Build positive‐pair mask: all same‐class, excluding self
    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.t()) & ~mask

    # 6) For each anchor i, average log‐prob over its |P(i)| positives
    positives_per_anchor = positive_mask.sum(dim=1)  # (N,)
    # avoid divide‐by‐zero
    positives_per_anchor = positives_per_anchor.clamp(min=1)

    loss = - (positive_mask * log_prob).sum(dim=1) / positives_per_anchor
    return loss.mean()


def get_pos_indices(labels):
    """
    Parameters:
        labels (tensor): GT labels
    Returns:
        (Row, Col) indices in the bool matrix where classes are equal
    """
    equal_classes_bool_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    return torch.nonzero(equal_classes_bool_matrix, as_tuple=False)


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def tensor_to_pil_image(tensor):
    """
    Convert a tensor to a PIL image and display it.

    Parameters:
        tensor (torch.Tensor): The input tensor to convert and display.
    """
    # Move to CPU and detach from the computation graph if needed
    tensor = tensor.cpu().detach()

    # If batch size > 1, select the first image in the batch
    if tensor.ndimension() == 4:
        tensor = tensor[0]

    # Convert tensor to uint8 and rearrange to HWC (Height, Width, Channels)
    image = tensor.permute(1, 2, 0).numpy()

    # Normalize if image is in the range [0, 1] (assuming it is)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    elif image.max() > 255:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    pil_image.show()

          
def train(train_loader, model, criterion, optimizer, epochs, temperature=0.05):    
    
    device = get_device()
    model.to(device)
    model.train()

    wandb.init(project="simclr-pid", name="contrastive-info-nce-dinov2")
    wandb.watch(model, log="all", log_freq=10)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    for epoch in tqdm(range(epochs)):
        
        for *views, labels in train_loader:

            all_pils = [img for view in views for img in view]
            img_pixels = processor(
                images=all_pils, 
                return_tensors="pt"
            ).pixel_values.to(device)  

            # Forward
            projection, embedding = model(img_pixels)

            # Repeat labels to account for increase in batch size 
            labels = labels.repeat(len(views)).to(device)
            
            # NTBxent Loss
            #pos_indices = get_pos_indices(labels)
            loss = criterion(projection, labels, temperature)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
            "train_loss": loss.item(),
            "epoch": epoch
            })

        
        cosine_scheduler.step()
        print(f'Epoch {epoch+1} Loss: {loss.item():.3f}\n')
    
    return model

                
if __name__ == '__main__':
        
    root = Path().resolve().parents[1]
    img_dir = root / 'Data' / 'SimCLR'
   
    # Image Augmentation pipeline
    augmentation_pipeline = img_augmentation(blur_kernel_size=3)
    
    # Generates 2 augmented samples for each img
    augmentation_generator = AugmentedViewGenerator(augmentation_pipeline, n_views=8)
    
    # 40 P&ID symbols with 2 randomly augmented views of each
    pid_dataset = PIDObjects(img_dir=img_dir, 
                             transform=augmentation_generator, 
                             return_pil=True)

    train_dataloader = DataLoader(
        pid_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=pil_collate
    )

    dino = DinoV2Backbone()
    encoder = ObjectEncoder(dino)
    #optimizer = torch.optim.SGD(encoder.parameters(), lr=1E-3)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=5E-5)
    
    #criterion = multiclass_nt_xent_loss
    #criterion = binary_nt_xent_loss
    criterion = supervised_contrastive_loss
    model = train(train_dataloader, encoder, criterion, optimizer, epochs=500, temperature=0.2)

    today = datetime.date.today()
    save_path = root / 'nyu-capstone-2024-PIDGraph' / 'Object Detection' / 'models'/ f'encoder_dino_{today.strftime("%Y-%m-%d")}.pth'
    torch.save(model.state_dict(), save_path)