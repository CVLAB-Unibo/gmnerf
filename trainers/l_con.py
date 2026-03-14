import torch

from pathlib import Path
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional

from models.siglip import SiglipLoss
from trainers.base import BaseTrainer
from trainers.utils import get_latest_ckpt_path, progress_bar


class ContrastiveTrainer(BaseTrainer):
    def __init__(
        self,
        mlp_nerf_root: Path,
        mlp_graph_root: str,
        triplane_nerf_root: Path,
        triplane_graph_root: str,
        hash_nerf_root: Path,
        hash_graph_root: str,
        mlp_nerf: nn.Sequential,
        triplane_nerf: nn.Sequential,
        hash_nerf: nn.Sequential,
        num_epochs: int,
        batch_size: int,
        lr: int,
        weight_decay: int,
        gnn_hidden_dim: int,
        num_gnn_layers: int,
        wandb_user: str,
        wandb_project: str,
        wandb_run_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            mlp_nerf_root,
            mlp_graph_root,
            triplane_nerf_root,
            triplane_graph_root,
            hash_nerf_root,
            hash_graph_root,
            mlp_nerf,
            triplane_nerf,
            hash_nerf,
            num_epochs,
            batch_size,
            lr,
            weight_decay,
            gnn_hidden_dim,
            num_gnn_layers,
            wandb_run_name,
            wandb_user,
            wandb_project
        )
        
        self.siglip = SiglipLoss().to(self.device)
        
        self.params = list(self.encoder.parameters()) + list(self.siglip.parameters())
        self.optimizer = AdamW(self.params, self.lr, weight_decay=self.wd)
        self.scheduler = OneCycleLR(self.optimizer, self.lr, total_steps=self.num_steps)

    def train(self) -> None:
        self.wandb_init()
        start_epoch = self.epoch
        
        for epoch in progress_bar(range(start_epoch, self.num_epochs), desc="Epoch"):
            self.epoch = epoch
            self.encoder.train()

            for batch in progress_bar(self.train_loader, desc="Batch"):
                _, _, _, graph_data, _, _, _, _ = batch

                graph_data = graph_data.to(self.device)
                graph_data.x = graph_data.x.float()

                self.optimizer.zero_grad()

                with autocast():
                    embeddings = self.encoder(
                        graph_data.x, 
                        graph_data.edge_index, 
                        graph_data.edge_attr, 
                        graph_data.batch
                    )

                    emb_A = embeddings[::2]
                    emb_B = embeddings[1::2]
                    loss = self.siglip(emb_A, emb_B)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                scale = self.scaler.get_scale()
                self.scaler.update()
                skip_sched = scale > self.scaler.get_scale()
                if not skip_sched:
                    self.scheduler.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss})
                    self.logfn({"train/siglip_loss": loss})
                    self.logfn({"train/siglip_scale": self.siglip.scale.item()})
                    self.logfn({"train/siglip_shift": self.siglip.shift.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})
                
                self.global_step += 1

            if (epoch > 0 and epoch % 5 == 0) or epoch == self.num_epochs - 1:
                self.val(split="validation")
            
            self.save_ckpt()   
    
    @torch.no_grad()
    def val(self, split: str) -> None:
        loader = self.train_loader if split == "train" else self.val_loader

        self.encoder.eval()

        siglips = []
        idx = 0

        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            _, _, _, graph_data, _, _, _, _ = batch

            graph_data = graph_data.to(self.device)
            graph_data.x = graph_data.x.float()

            embeddings = self.encoder(
                graph_data.x, 
                graph_data.edge_index, 
                graph_data.edge_attr, 
                graph_data.batch
            )

            emb_A = embeddings[::2]
            emb_B = embeddings[1::2]
            siglip_loss = self.siglip(emb_A, emb_B)
            siglips.append(siglip_loss.item())

            if idx > 99:
                break
            idx += 1
        
        mean_siglip = sum(siglips) / len(siglips)
        self.logfn({f"{split}/siglip_loss": mean_siglip})
        
        if mean_siglip < self.best_siglip:
            self.best_siglip = mean_siglip
            self.save_ckpt(best=True)
    
                 
    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "siglip": self.siglip.state_dict(),
            "best_siglip": self.best_siglip,
        }

        for previous_ckpt_path in self.ckpts_dir.glob("*.pt"):
            if "best" not in previous_ckpt_path.name:
                previous_ckpt_path.unlink()

        ckpt_path = self.ckpts_dir / f"{self.epoch}.pt"
        torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_dir / "best.pt"
            torch.save(ckpt, ckpt_path)
    
    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_dir.exists():
            ckpt_path = get_latest_ckpt_path(self.ckpts_dir)
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_siglip = ckpt["best_siglip"]

            self.encoder.load_state_dict(ckpt["encoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.siglip.load_state_dict(ckpt["siglip"])
