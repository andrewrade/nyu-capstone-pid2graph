import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import SamModel, AutoModel


class SAMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'facebook/sam-vit-base'
        self.encoder = SamModel.from_pretrained(self.model_name).vision_encoder.eval()
        
        # Freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.embedding_dim = getattr(self.encoder.config, "output_channels")

    @torch.no_grad()
    def forward(self, x):
        x = self.encoder(pixel_values=x).last_hidden_state
        return x.mean(dim=(2,3)) # Global Avg Pooling over spatial dimension 


class DinoV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'facebook/dinov2-base'
        self.encoder = AutoModel.from_pretrained(self.model_name).eval()
        
        # Freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.embedding_dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, x):
        x = self.encoder(pixel_values=x).last_hidden_state
        return x[:, 0]

class AdapterMLP(nn.Module):
    """
    Learnable adapter after backbone.
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.out_dim = in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim)
        )
    
    def forward(self, x):
        """
        Learn residual to preserve semantics 
        from pre-trained frozen
        backbones
        """
        residual = self.mlp(x)
        x = x + residual 
        return x

class ProjectionHead(nn.Module):
    """
    Projection head for use during contrastive learning
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.projector(x)
    
class ClassificationHead(nn.Module):
    """
    Classification head for supervised learning
    """
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, n_classes)
    
    def forward(self, x):
        return self.classifier(x)


