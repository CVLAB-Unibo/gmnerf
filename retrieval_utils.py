import imageio.v2 as imageio
import math
import numpy as np
import shutil
import torch
import wandb

from collections import defaultdict
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, List

from nerfacc import OccupancyGrid
from nf2vec import config as nerf_cfg
from nf2vec.nerf.hash import HashGridRadianceField
from nf2vec.nerf.mlp import MlpRadianceField
from nf2vec.nerf.triplane import TriplaneRadianceField
from nf2vec.nerf.rendering import render_image_from_nerf
from nf2vec.nerf.utils import Rays, get_rays

from datasets.emb import RetrievalEmbeddingDataset
from trainers.utils import progress_bar


@torch.no_grad()
def draw_images(
    grid_weights: np.ndarray, 
    nerf_weights: np.ndarray,
    class_ids: np.ndarray,
    nerf_ids: np.ndarray,
    save_path: Path,
    gt_path: Path,
    device: str = "cuda"
) -> None:
    occupancy_grid = OccupancyGrid(
        roi_aabb=nerf_cfg.GRID_AABB,
        resolution=nerf_cfg.GRID_RESOLUTION,
        contraction_type=nerf_cfg.GRID_CONTRACTION_TYPE
    )
    occupancy_grid = occupancy_grid.to(device)
    occupancy_grid.eval()

    mlp_nerf = MlpRadianceField(**nerf_cfg.MLP_CONF)
    mlp_nerf = mlp_nerf.to(device)
    mlp_nerf.eval()

    triplane_nerf = TriplaneRadianceField(**nerf_cfg.TRIPLANE_CONF)
    triplane_nerf = triplane_nerf.to(device)
    triplane_nerf.eval()

    hash_nerf = HashGridRadianceField(**nerf_cfg.HASH_CONF)
    hash_nerf = hash_nerf.to(device)
    hash_nerf.eval()

    scene_aabb = torch.tensor(nerf_cfg.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / nerf_cfg.GRID_CONFIG_N_SAMPLES
    ).item()

    rays = get_rays(device)
    rays = Rays(origins=rays.origins.unsqueeze(0), viewdirs=rays.viewdirs.unsqueeze(0))
    color_bkgd = None

    for i, (grid_w, nerf_w, class_id, nerf_id) in enumerate(zip(grid_weights, nerf_weights, class_ids, nerf_ids)):
        with autocast():
            rgb, alpha, _, _ = render_image_from_nerf(
                radiance_field={"mlp": mlp_nerf, "triplane": triplane_nerf, "hash": hash_nerf}, 
                occupancy_grid=occupancy_grid, 
                rays=rays, 
                scene_aabb=scene_aabb, 
                render_step_size=render_step_size,
                color_bkgds=color_bkgd,
                grid_weights=grid_w,
                ngp_mlp_weights=nerf_w,
                device=device,
                training=False
            )

        rgb = rgb.squeeze(0)                   
        alpha = alpha.squeeze(0)             
        rgb_uint8 = (rgb * 255).clamp(0, 255).byte()
        alpha_uint8 = (alpha * 255).clamp(0, 255).byte()
        rgba = torch.cat([rgb_uint8, alpha_uint8], dim=-1).cpu().numpy()
        
        save_path.mkdir(parents=True, exist_ok=True)
        nerf_arch = nerf_w["nerf_arch"][0]
        imageio.imwrite(save_path / f"{i}_{class_id}_{nerf_id}_{nerf_arch}.png", rgba)
    
    shutil.copy(gt_path, save_path / "query_gt.png")
    

def get_recalls(
    embeddings: Tensor, 
    grid_weights: np.ndarray, 
    nerf_weights: np.ndarray,
    labels: Tensor,
    class_ids: np.ndarray,
    nerf_ids: np.ndarray,
    Rs: List[int],
    save_paths: List[Path],
    gt_paths: List[Path],
    query_arch: str,
    gallery_arch: str
) -> Dict[int, float]:
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    max_nn = max(Rs)
    
    is_gallery_mask = np.array([w["nerf_arch"][0] == gallery_arch for w in nerf_weights])
    gallery = embeddings[is_gallery_mask]
    gallery_grid_weights = grid_weights[is_gallery_mask]
    gallery_nerf_weights = nerf_weights[is_gallery_mask]
    gallery_class_ids = class_ids[is_gallery_mask]
    gallery_nerf_ids = nerf_ids[is_gallery_mask]
    
    tree = NearestNeighbors(n_neighbors=max_nn, metric="cosine")
    tree.fit(gallery)

    retrieval_count = defaultdict(int)
    recalls = defaultdict(float)
    success = False
    
    for query_idx, (query, label, class_id, nerf_id, save_p, gt_p) in enumerate(progress_bar(
        zip(embeddings, labels, class_ids, nerf_ids, save_paths, gt_paths), 
        total=len(embeddings), 
        desc="Computing NNs"
    )):
        current_query_arch = nerf_weights[query_idx]["nerf_arch"][0]
        if current_query_arch != query_arch:
            continue

        query = np.expand_dims(query, 0)
        closest_indices = tree.kneighbors(query, return_distance=False).squeeze()

        for r in Rs:
            closest_class_ids = gallery_class_ids[closest_indices[:r]]
            closest_nerf_ids = gallery_nerf_ids[closest_indices[:r]]
            hit = sum((closest_class_ids == class_id) & (closest_nerf_ids == nerf_id)) > 0
            if hit:
                success = True
            recalls[r] += hit

        if success and retrieval_count[label] < 10:
            draw_images(
                np.concatenate([[grid_weights[query_idx]], gallery_grid_weights[closest_indices]]), 
                np.concatenate([[nerf_weights[query_idx]], gallery_nerf_weights[closest_indices]]),
                np.concatenate([[class_ids[query_idx]], class_ids[closest_indices]]),
                np.concatenate([[nerf_ids[query_idx]], nerf_ids[closest_indices]]),
                save_p,
                gt_p,
            )
            success = False
            retrieval_count[label] += 1
                
    recalls = {r: value / len(gallery) for r, value in recalls.items()}
    return recalls


def run_retrieval(
    emb_root: Path, 
    query_nerf_root: Path, 
    gallery_nerf_root: Path, 
    save_root: Path, 
    split: str,
    query_arch: str,
    gallery_arch: str
) -> None:
    dset = RetrievalEmbeddingDataset(emb_root, query_nerf_root, gallery_nerf_root, split, query_arch, gallery_arch)
    loader = DataLoader(dset, batch_size=1, shuffle=False)

    embeddings = []
    grid_weights = []
    nerf_weights = []
    labels = []
    class_ids = []
    nerf_ids = []
    save_paths = []
    gt_paths = []

    for batch in progress_bar(loader, desc=f"Scanning {split} set"):
        emb, grid_w, nerf_w, label, class_id, nerf_id = batch
        class_id, nerf_id = class_id[0], nerf_id[0]
        embeddings.append(emb.squeeze())
        grid_weights.append(grid_w)
        nerf_weights.append(nerf_w)
        labels.append(label.squeeze())
        class_ids.append(class_id)
        nerf_ids.append(nerf_id)
        save_paths.append(save_root / class_id / nerf_id)
        gt_path = query_nerf_root / class_id / nerf_id / "train" / "00.png"
        gt_paths.append(gt_path)
        
    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)
    grid_weights = np.array(grid_weights)
    nerf_weights = np.array(nerf_weights)
    class_ids = np.array(class_ids)
    nerf_ids = np.array(nerf_ids)

    recalls = get_recalls(
        embeddings, 
        grid_weights, 
        nerf_weights,
        labels,
        class_ids, 
        nerf_ids, 
        [1, 5, 10], 
        save_paths, 
        gt_paths, 
        query_arch, 
        gallery_arch
    )

    for r, value in recalls.items():
        wandb.log({f"{split}/recall@{r}": value})
