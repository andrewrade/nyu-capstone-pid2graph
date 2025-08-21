import torch
import torch.nn as nn
import torch.nn.functional as F

def supervised_contrastive_loss(x, labels, temperature):
    """
    InfoNCE Loss
    """
    f = F.normalize(x, dim=1)
    logits = torch.matmul(f, f.t()) / temperature

    mask = torch.eye(x.size(0), device=x.device).bool()
    logits_masked = logits.masked_fill(mask, -1e9)

    exp_logits = torch.exp(logits_masked)
    log_prob = logits_masked - torch.log(exp_logits.sum(dim=1, keepdim=True))

    labels = labels.unsqueeze(1)
    pos_mask = (labels == labels.t()) & ~mask
    pos_per_anchor = pos_mask.sum(dim=1).clamp(min=1)

    loss = - (pos_mask * log_prob).sum(dim=1) / pos_per_anchor
    return loss.mean()
