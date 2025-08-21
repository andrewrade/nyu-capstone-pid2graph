import argparse
import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import wandb

from data_utils import PIDObjects, UnifiedPreprocessor, AugmentedViewGenerator, make_view_collate, img_augmentation
from models import DinoV2Backbone, SAMBackbone, AdapterMLP, ProjectionHead, ClassificationHead
from losses import supervised_contrastive_loss


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    p = argparse.ArgumentParser("PID contrastive / supervised trainer")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--backbone", choices=["dinov2", "sam"], default="dinov2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--mode", choices=["contrastive", "supervised"], default="contrastive")
    p.add_argument("--n_views", type=int, default=1)
    p.add_argument("--loss",
                   choices=["supervised_contrastive", "binary_nt_xent", "multiclass_nt_xent"],
                   default="supervised_contrastive")
    p.add_argument("--num_classes", type=int, default=10)
    return p.parse_args()


def train(loader, model, preprocessor, mode, criterion, n_views, optimizer, epochs, temperature):
    device = get_device()
    model.to(device).train()
    step = 0
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in loader:
            
            if mode == "contrastive":
                
                views, labels = batch
                all_views = []       # list of tensors, one per view index
                for v in range(n_views):
                    imgs_v = [img_list[v] for img_list in views]
                    all_views.append(preprocessor(imgs_v).to(device))

                pix = torch.cat(all_views, dim=0)               # [B * n_views, ...]
                labels = labels.repeat(n_views).to(device)      # [B * n_views]
                embeddings = model(pix)
                loss = criterion(embeddings, labels, temperature)

            else:  # supervised classification
                # batch: (images, labels), where images: List[PIL] of length B
                images, labels = batch

                # single tensor batch
                pix    = preprocessor(images).to(device)  # [B, C, H, W]
                logits = model(pix)                       # [B, num_classes]
                loss   = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item(),
                       "epoch":      epoch,
                       "step":       step})
            step += 1 

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    return model



# ──────────────────────────────── main ──────────────────────────────────
if __name__ == "__main__":
    a = parse_args()
    
    data = PIDObjects(a.img_dir)
    augmentation_generator = AugmentedViewGenerator(
        transform=img_augmentation,
        n_views = a.n_views
    )
    view_collate = make_view_collate(augmentation_generator)
    loader = DataLoader(data, batch_size=a.batch_size, shuffle=True, collate_fn=view_collate)
    preprocess = UnifiedPreprocessor(a.backbone)
    bck = DinoV2Backbone() if a.backbone == "dinov2" else SAMBackbone()

    # heads / loss
    if a.mode == "contrastive":
        adapter = AdapterMLP(bck.embedding_dim, bck.embedding_dim * 2)
        head = ProjectionHead(bck.embedding_dim)
        loss_map = {
            "supervised_contrastive": supervised_contrastive_loss,
        }
        crit = loss_map[a.loss]
    else:
        head = ClassificationHead(bck.embedding_dim, a.num_classes)
        crit = torch.nn.CrossEntropyLoss()

    model = torch.nn.Sequential(bck, adapter, head)
    opt   = torch.optim.AdamW(model.parameters(), lr=a.lr)

    wandb.init(project="pid-infonce-frozenbackbone", name=f"{a.mode}-{a.backbone}")
    wandb.define_metric("step")                 
    wandb.define_metric("train/loss", step_metric="step") 
    wandb.watch(model, log="all", log_freq=100)

    train(loader, model, preprocess, a.mode, crit, n_views=a.n_views,
          optimizer=opt, epochs=a.epochs, temperature=a.temperature)

    out = Path("models")
    out.mkdir(exist_ok=True)
    tgt = out / f"{a.mode}-{a.backbone}-{datetime.date.today()}.pth"
    torch.save(model.state_dict(), tgt)
    print(f"Saved → {tgt}")
