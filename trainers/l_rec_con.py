import math
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from pathlib import Path
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional

from nerfacc import OccupancyGrid
from nf2vec import config as nerf_cfg
from nf2vec.models.idecoder import ImplicitDecoder
from nf2vec.nerf.hash import HashGridRadianceField
from nf2vec.nerf.mlp import MlpRadianceField
from nf2vec.nerf.triplane import TriplaneRadianceField
from nf2vec.nerf.rendering import render_image_from_decoder, render_image_from_nerf
from nf2vec.nerf.utils import Rays

from models.siglip import SiglipLoss
from trainers.base import BaseTrainer
from trainers.utils import get_latest_ckpt_path, progress_bar


class RecontructionContrastiveTrainer(BaseTrainer):
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
        
        decoder = ImplicitDecoder(
            embed_dim=nerf_cfg.ENCODER_EMBEDDING_DIM,
            in_dim=nerf_cfg.DECODER_INPUT_DIM,
            hidden_dim=nerf_cfg.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=nerf_cfg.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=nerf_cfg.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=nerf_cfg.DECODER_OUT_DIM,
            encoding_conf=nerf_cfg.TRIPLANE_ENCODING_CONF,
            aabb=torch.tensor(nerf_cfg.GRID_AABB, dtype=torch.float32, device=self.device)
        )
        self.decoder = decoder.to(self.device)

        occupancy_grid = OccupancyGrid(
            roi_aabb=nerf_cfg.GRID_AABB,
            resolution=nerf_cfg.GRID_RESOLUTION,
            contraction_type=nerf_cfg.GRID_CONTRACTION_TYPE
        )
        self.occupancy_grid = occupancy_grid.to(self.device)
        self.occupancy_grid.eval()

        mlp_nerf = MlpRadianceField(**nerf_cfg.MLP_CONF)
        self.mlp_nerf = mlp_nerf.to(self.device)
        self.mlp_nerf.eval()

        triplane_nerf = TriplaneRadianceField(**nerf_cfg.TRIPLANE_CONF)
        self.triplane_nerf = triplane_nerf.to(self.device)
        self.triplane_nerf.eval()

        hash_nerf = HashGridRadianceField(**nerf_cfg.HASH_CONF)
        self.hash_nerf = hash_nerf.to(self.device)
        self.hash_nerf.eval()

        self.scene_aabb = torch.tensor(nerf_cfg.GRID_AABB, dtype=torch.float32, device=self.device)
        self.render_step_size = (
            (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
            * math.sqrt(3)
            / nerf_cfg.GRID_CONFIG_N_SAMPLES
        ).item()

        self.siglip = SiglipLoss().to(self.device)
        self.siglip_weight = 2e-2
        self.train_cfg["siglip_weight"] =  self.siglip_weight
        
        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.siglip.parameters())
        self.optimizer = AdamW(self.params, self.lr, weight_decay=self.wd)
        self.scheduler = OneCycleLR(self.optimizer, self.lr, total_steps=self.num_steps)

    def train(self) -> None:
        self.wandb_init()
        start_epoch = self.epoch
        
        for epoch in progress_bar(range(start_epoch, self.num_epochs), desc="Epoch"):
            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()

            for batch in progress_bar(self.train_loader, desc="Batch"):
                train_dict, _, nerf_weights, graph_data, grid_weights, background_indices, class_ids, nerf_ids = batch

                rays = train_dict["rays"]
                rays = rays._replace(origins=rays.origins.to(self.device), viewdirs=rays.viewdirs.to(self.device))
                
                color_bkgds = train_dict["color_bkgd"]
                color_bkgds = color_bkgds[0][None].expand(graph_data.num_graphs, -1)
                color_bkgds = color_bkgds.to(self.device)

                graph_data = graph_data.to(self.device)
                graph_data.x = graph_data.x.float()

                self.optimizer.zero_grad()

                with autocast():
                    rgb_mlp, alpha, filtered_rays = render_image_from_nerf(
                        radiance_field={"mlp": self.mlp_nerf, "triplane": self.triplane_nerf, "hash": self.hash_nerf}, 
                        occupancy_grid=self.occupancy_grid, 
                        rays=rays, 
                        scene_aabb=self.scene_aabb, 
                        render_step_size=self.render_step_size,
                        color_bkgds=color_bkgds,
                        grid_weights=grid_weights,
                        ngp_mlp_weights=nerf_weights,
                        device=self.device,
                    )
                    rgb_mlp = rgb_mlp * alpha + color_bkgds.unsqueeze(1) * (1.0 - alpha)

                    embeddings = self.encoder(
                        graph_data.x, 
                        graph_data.edge_index, 
                        graph_data.edge_attr, 
                        graph_data.batch
                    )
                    
                    rgb_dec, _, _, _, bg_rgb_dec, bg_rgb_label = render_image_from_decoder(
                        self.decoder,
                        embeddings,
                        self.occupancy_grid,
                        filtered_rays,
                        self.scene_aabb,
                        render_step_size=self.render_step_size,
                        render_bkgd=color_bkgds,
                        grid_weights=grid_weights,
                        background_indices=background_indices,
                        max_foreground_coordinates=nerf_cfg.MAX_FOREGROUND_COORDINATES,
                        max_background_coordinates=nerf_cfg.MAX_BACKGROUND_COORDINATES,
                        device=self.device
                    )
                    
                    fg_loss = F.smooth_l1_loss(rgb_dec, rgb_mlp) * nerf_cfg.FG_WEIGHT
                    bg_loss = F.smooth_l1_loss(bg_rgb_dec, bg_rgb_label) * nerf_cfg.BG_WEIGHT
                    recontr_loss = fg_loss + bg_loss

                    emb_A = embeddings[::2]
                    emb_B = embeddings[1::2]
                    siglip_loss = self.siglip(emb_A, emb_B)
                    scaled_siglip_loss = self.siglip_weight * siglip_loss
                    
                    loss = recontr_loss + scaled_siglip_loss
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                scale = self.scaler.get_scale()
                self.scaler.update()
                skip_sched = scale > self.scaler.get_scale()
                if not skip_sched:
                    self.scheduler.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss})
                    self.logfn({"train/reconstr_loss": recontr_loss})
                    self.logfn({"train/siglip_loss": siglip_loss})
                    self.logfn({"train/scaled_siglip_loss": scaled_siglip_loss})
                    self.logfn({"train/siglip_scale": self.siglip.scale.item()})
                    self.logfn({"train/siglip_shift": self.siglip.shift.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})
                
                self.global_step += 1

            if (epoch > 0 and epoch % 5 == 0) or epoch == self.num_epochs - 1:
                self.val(split="train")
                self.val(split="validation")

                self.plot(split="train")
                self.plot(split="validation")
            
            self.save_ckpt()   
    
    @torch.no_grad()
    def val(self, split: str) -> None:
        loader = self.train_loader if split == "train" else self.val_loader

        self.encoder.eval()
        self.decoder.eval()

        psnrs = []
        psnrs_bg = []
        siglips = []
        idx = 0

        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            train_dict, _, nerf_weights, graph_data, grid_weights, background_indices, class_ids, nerf_ids = batch
            
            rays = train_dict["rays"]
            rays = rays._replace(origins=rays.origins.to(self.device), viewdirs=rays.viewdirs.to(self.device))
            
            color_bkgds = train_dict["color_bkgd"]
            color_bkgds = color_bkgds[0].unsqueeze(0).expand(graph_data.num_graphs, -1)
            color_bkgds = color_bkgds.to(self.device)

            graph_data = graph_data.to(self.device)
            graph_data.x = graph_data.x.float()

            with autocast():
                rgb_mlp, alpha, filtered_rays = render_image_from_nerf(
                    radiance_field={"mlp": self.mlp_nerf, "triplane": self.triplane_nerf, "hash": self.hash_nerf}, 
                    occupancy_grid=self.occupancy_grid, 
                    rays=rays, 
                    scene_aabb=self.scene_aabb, 
                    render_step_size=self.render_step_size,
                    color_bkgds=color_bkgds,
                    grid_weights=grid_weights,
                    ngp_mlp_weights=nerf_weights,
                    device=self.device,
                    class_ids=class_ids,
                    nerf_ids=nerf_ids
                )
                rgb_mlp = rgb_mlp * alpha + color_bkgds.unsqueeze(1) * (1.0 - alpha)
                
                embeddings = self.encoder(
                    graph_data.x, 
                    graph_data.edge_index, 
                    graph_data.edge_attr, 
                    graph_data.batch
                )
                
                rgb_dec, _, _, _, bg_rgb_dec, bg_rgb_label = render_image_from_decoder(
                    self.decoder,
                    embeddings,
                    self.occupancy_grid,
                    filtered_rays,
                    self.scene_aabb,
                    render_step_size=self.render_step_size,
                    render_bkgd=color_bkgds,
                    grid_weights=grid_weights,
                    background_indices=background_indices,
                    max_foreground_coordinates=nerf_cfg.MAX_FOREGROUND_COORDINATES,
                    max_background_coordinates=nerf_cfg.MAX_BACKGROUND_COORDINATES,
                    device=self.device
                )
                
                fg_mse = F.mse_loss(rgb_dec, rgb_mlp) * nerf_cfg.FG_WEIGHT
                bg_mse = F.mse_loss(bg_rgb_dec, bg_rgb_label) * nerf_cfg.BG_WEIGHT

                mse_bg = fg_mse + bg_mse
                mse = F.mse_loss(rgb_dec, rgb_mlp)

                if split == "validation":
                    emb_A = embeddings[::2]
                    emb_B = embeddings[1::2]
                    siglip_loss = self.siglip(emb_A, emb_B)
            
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())

            psnr_bg = -10.0 * torch.log(mse_bg) / np.log(10.0)
            psnrs_bg.append(psnr_bg.item())

            if split == "validation":
                siglips.append(siglip_loss.item())

            if idx > 99:
                break
            idx += 1
        
        mean_psnr = sum(psnrs) / len(psnrs)
        mean_psnr_bg = sum(psnrs_bg) / len(psnrs_bg)
        if split == "validation":
            mean_siglip = sum(siglips) / len(siglips)

        self.logfn({f"{split}/PSNR": mean_psnr})
        self.logfn({f"{split}/PSNR_BG": mean_psnr_bg})
        if split == "validation":
            self.logfn({f"{split}/siglip_loss": mean_siglip})
        
        if split == "validation" and mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr
            self.save_ckpt(best=True)
    
    @torch.no_grad()
    def plot(self, split: str) -> None:
        loader = self.train_loader if split == "train" else self.val_loader

        self.encoder.eval()
        self.decoder.eval()

        for batch in loader:
            _, test_dict, nerf_weights, graph_data, grid_weights, _, _, _ = batch
            
            rays = test_dict["rays"]
            rays = rays._replace(origins=rays.origins.to(self.device), viewdirs=rays.viewdirs.to(self.device))
            
            color_bkgds = test_dict["color_bkgd"]
            color_bkgds = color_bkgds.to(self.device)
            
            graph_data = graph_data.to(self.device)
            graph_data.x = graph_data.x.float()
            
            with autocast():
                rgb_mlp, alpha, _ = render_image_from_nerf(
                    radiance_field={"mlp": self.mlp_nerf, "triplane": self.triplane_nerf, "hash": self.hash_nerf}, 
                    occupancy_grid=self.occupancy_grid, 
                    rays=rays, 
                    scene_aabb=self.scene_aabb, 
                    render_step_size=self.render_step_size,
                    color_bkgds=color_bkgds,
                    grid_weights=grid_weights,
                    ngp_mlp_weights=nerf_weights,
                    device=self.device,
                    training=False
                )
                rgb_mlp = rgb_mlp * alpha + color_bkgds.unsqueeze(1).unsqueeze(1) * (1.0 - alpha)
            
                embeddings = self.encoder(
                    graph_data.x, 
                    graph_data.edge_index, 
                    graph_data.edge_attr, 
                    graph_data.batch
                )

                for idx in range(graph_data.num_graphs):
                    curr_grid_weights = {
                        "_roi_aabb": [grid_weights["_roi_aabb"][idx]],
                        "_binary": [grid_weights["_binary"][idx]],
                        "resolution": [grid_weights["resolution"][idx]],
                        "occs": [grid_weights["occs"][idx]],
                    }
            
                    rgb_dec, _, _, _, _, _ = render_image_from_decoder(
                        decoder=self.decoder,
                        embeddings=embeddings[idx].unsqueeze(0),
                        occupancy_grid=self.occupancy_grid,
                        rays=Rays(origins=rays.origins[idx].unsqueeze(0), viewdirs=rays.viewdirs[idx].unsqueeze(0)),
                        scene_aabb=self.scene_aabb,
                        render_step_size=self.render_step_size,
                        render_bkgd=color_bkgds[idx].unsqueeze(0),
                        grid_weights=curr_grid_weights,
                        device=self.device
                    )
                    
                    rgb_dec_no_grid, _, _, _, _, _ = render_image_from_decoder(
                        decoder=self.decoder,
                        embeddings=embeddings[idx].unsqueeze(0),
                        occupancy_grid=None,
                        rays=Rays(origins=rays.origins[idx].unsqueeze(0), viewdirs=rays.viewdirs[idx].unsqueeze(0)),
                        scene_aabb=self.scene_aabb,
                        render_step_size=self.render_step_size,
                        render_bkgd=color_bkgds[idx].unsqueeze(0),
                        grid_weights=None,
                        device=self.device
                    )
                    
                    image_mlp = wandb.Image((rgb_mlp.cpu().detach().numpy()[idx] * 255).astype(np.uint8))
                    image_dec = wandb.Image((rgb_dec.cpu().detach().numpy()[0] * 255).astype(np.uint8)) 
                    image_dec_no_grid = wandb.Image((rgb_dec_no_grid.cpu().detach().numpy() * 255).astype(np.uint8))

                    self.logfn({f"{split}/nerf_{idx}": [image_mlp, image_dec, image_dec_no_grid]})

            break
                 
    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "siglip": self.siglip.state_dict(),
            "best_psnr": self.best_psnr,
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
            self.best_psnr = ckpt["best_psnr"]

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.siglip.load_state_dict(ckpt["siglip"])
