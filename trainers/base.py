import torch
import wandb

from pathlib import Path
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from typing import Any, Dict, Optional

from nf2vec import config as nerf_cfg

from datasets.single_arch import GraphDataset, ReconstructionDataset
from datasets.multi_arch import BalancedSampler, PaddedHashGridDataset, PaddedMlpDataset, PaddedTriplaneDataset
from models.gnn import GNN


class BaseTrainer:
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
        self.device = "cuda"

        train_mlp_dset = PaddedMlpDataset(mlp_nerf_root, "train", mlp_nerf.state_dict(), "cpu")
        train_mlp_graph_dset = GraphDataset(f"{mlp_graph_root}/train", train_mlp_dset, mlp_nerf)
        train_mlp_rec_dset = ReconstructionDataset(train_mlp_graph_dset, "cpu")

        val_mlp_dset = PaddedMlpDataset(mlp_nerf_root, "val", mlp_nerf.state_dict(), "cpu")
        val_mlp_graph_dset = GraphDataset(f"{mlp_graph_root}/val", val_mlp_dset, mlp_nerf)
        val_mlp_rec_dset = ReconstructionDataset(val_mlp_graph_dset, "cpu")

        train_trip_dset = PaddedTriplaneDataset(triplane_nerf_root, "train", triplane_nerf.state_dict(), "cpu")
        train_trip_graph_dset = GraphDataset(f"{triplane_graph_root}/train", train_trip_dset, triplane_nerf)
        train_trip_rec_dset = ReconstructionDataset(train_trip_graph_dset, "cpu")
        
        val_trip_dset = PaddedTriplaneDataset(triplane_nerf_root, "val", triplane_nerf.state_dict(), "cpu")
        val_trip_graph_dset = GraphDataset(f"{triplane_graph_root}/val", val_trip_dset, triplane_nerf)
        val_trip_rec_dset = ReconstructionDataset(val_trip_graph_dset, "cpu")

        train_hash_dset = PaddedHashGridDataset(hash_nerf_root, "train", hash_nerf.state_dict(), "cpu")
        train_hash_graph_dset = GraphDataset(f"{hash_graph_root}/train", train_hash_dset, hash_nerf)
        train_hash_rec_dset = ReconstructionDataset(train_hash_graph_dset, "cpu")
        
        val_hash_dset = PaddedHashGridDataset(hash_nerf_root, "val", hash_nerf.state_dict(), "cpu")
        val_hash_graph_dset = GraphDataset(f"{hash_graph_root}/val", val_hash_dset, hash_nerf)
        val_hash_rec_dset = ReconstructionDataset(val_hash_graph_dset, "cpu")

        train_mixed_dset = ConcatDataset([train_mlp_rec_dset, train_trip_rec_dset, train_hash_rec_dset])
        val_mixed_dset = ConcatDataset([val_mlp_rec_dset, val_trip_rec_dset, val_hash_rec_dset])

        train_sampler = BalancedSampler(train_mlp_rec_dset, train_trip_rec_dset, train_hash_rec_dset)
        val_sampler = BalancedSampler(val_mlp_rec_dset, val_trip_rec_dset, val_hash_rec_dset)
        
        self.train_loader = DataLoader(
            train_mixed_dset,
            batch_size=batch_size,
            sampler=train_sampler, 
            num_workers=8
        )
        self.val_loader = DataLoader(
            val_mixed_dset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=8
        )
        
        encoder = GNN(
            gnn_hidden_dim,
            nerf_cfg.ENCODER_EMBEDDING_DIM,
            num_gnn_layers
        )
        self.encoder = encoder.to(self.device)

        self.lr = lr
        self.wd = weight_decay
        self.num_epochs = num_epochs
        self.num_steps = self.num_epochs * len(self.train_loader)
        
        self.scaler = torch.cuda.amp.GradScaler(2**10)
        
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = float("-inf")

        self.run_name = wandb_run_name
        self.user = wandb_user
        self.project = wandb_project
        
        self.ckpts_dir = Path(f"ckpts") / wandb_run_name
        if self.ckpts_dir.exists():
            self.restore_from_last_ckpt()
        self.ckpts_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "gnn_hidden_dim": gnn_hidden_dim, 
            "num_gnn_layers": num_gnn_layers,
        }
        self.train_cfg = dict(train_cfg, **nerf_cfg.WANDB_CONFIG)

    def train(self) -> None:
        raise NotImplementedError
    
    @torch.no_grad()
    def val(self, split: str) -> None:
        raise NotImplementedError
    
    @torch.no_grad()
    def plot(self, split: str) -> None:
        raise NotImplementedError
                 
    def save_ckpt(self, best: bool = False) -> None:
        raise NotImplementedError
    
    def restore_from_last_ckpt(self) -> None:
        raise NotImplementedError
    
    def wandb_init(self) -> None:
        wandb.init(
            name=self.run_name,
            entity=self.user,
            project=self.project,
            config=self.train_cfg
        )
    
    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)
