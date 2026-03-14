import numpy as np
import random
import torch

from pathlib import Path
from torch import Tensor
from torch.utils.data import Sampler
from typing import Dict, Iterator, Tuple

from nf2vec import config as nerf_cfg

from datasets.single_arch import HashGridDataset, MlpDataset, ReconstructionDataset, TriplaneDataset


class PaddedMlpDataset(MlpDataset):
    def __init__(
        self, 
        root: Path, 
        split: str, 
        sample_sd: Dict[str, Tensor], 
        device: str
    ) -> None:
        super().__init__(root, split, sample_sd, device)

        self.feat_dim = nerf_cfg.TRIPLANE_FEAT_SIZE
        self.res = nerf_cfg.TRIPLANE_RES
        enc_dim = nerf_cfg.TRIPLANE_IN_SIZE_AFTER_ENC
        hid_dim = nerf_cfg.TRIPLANE_HID_UNITS
        out_dim = nerf_cfg.TRIPLANE_PADDED_OUT_SIZE
        num_hid_layers = nerf_cfg.TRIPLANE_HID_LAYERS - 1
        self.num_triplane_weights = (enc_dim+self.feat_dim)*hid_dim + num_hid_layers*hid_dim**2 + hid_dim*out_dim

        n_levels = nerf_cfg.HASH_LEVELS
        n_features_per_level = nerf_cfg.HASH_FEATURES_PER_ENTRY
        log2_hashmap_size = nerf_cfg.HASH_LOG2_TAB_SIZE
        self.num_hash_params = 2**log2_hashmap_size * n_features_per_level * n_levels

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict, str, str]:
        grid_sd, flat_sd, reshaped_sd, class_id, nerf_id = super().__getitem__(idx)

        mlp_weights = flat_sd["mlp_base.params"]
        num_mlp_weights = mlp_weights.shape[-1]
        padded_weights = torch.full([self.num_triplane_weights], float("nan"))
        padded_weights[:num_mlp_weights] = mlp_weights
        flat_sd["mlp_base.params"] = padded_weights

        flat_sd["encoding.params"] = torch.full([self.num_hash_params], float("nan"))
        flat_sd["triplane"] = torch.full([3, self.res, self.res, self.feat_dim], float("nan"))
        flat_sd["class_id"] = float("nan")
        flat_sd["is_triplane"] = False
        flat_sd["is_hash"] = False
    
        return grid_sd, flat_sd, reshaped_sd, class_id, nerf_id
    

class PaddedTriplaneDataset(TriplaneDataset):
    def __init__(
        self, 
        root: Path, 
        split: str, 
        sample_sd: Dict[str, Tensor], 
        device: str
    ) -> None:
        super().__init__(root, split, sample_sd, device)

        n_levels = nerf_cfg.HASH_LEVELS
        n_features_per_level = nerf_cfg.HASH_FEATURES_PER_ENTRY
        log2_hashmap_size = nerf_cfg.HASH_LOG2_TAB_SIZE
        self.num_hash_params = 2**log2_hashmap_size * n_features_per_level * n_levels

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict, str, str]:
        grid_sd, flat_sd, reshaped_sd, class_id, nerf_id = super().__getitem__(idx)

        flat_sd["encoding.params"] = torch.full([self.num_hash_params], float("nan"))
        flat_sd["is_hash"] = False
    
        return grid_sd, flat_sd, reshaped_sd, class_id, nerf_id
    

class PaddedHashGridDataset(HashGridDataset):
    def __init__(
        self, 
        root: Path, 
        split: str, 
        sample_sd: Dict[str, Tensor], 
        device: str
    ) -> None:
        super().__init__(root, split, sample_sd, device)

        self.feat_dim = nerf_cfg.TRIPLANE_FEAT_SIZE
        self.res = nerf_cfg.TRIPLANE_RES
        enc_dim = nerf_cfg.TRIPLANE_IN_SIZE_AFTER_ENC
        hid_dim = nerf_cfg.TRIPLANE_HID_UNITS
        out_dim = nerf_cfg.TRIPLANE_PADDED_OUT_SIZE
        num_hid_layers = nerf_cfg.TRIPLANE_HID_LAYERS - 1
        self.num_triplane_weights = (enc_dim+self.feat_dim)*hid_dim + num_hid_layers*hid_dim**2 + hid_dim*out_dim

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict, str, str]:
        grid_sd, flat_sd, reshaped_sd, class_id, nerf_id = super().__getitem__(idx)

        original_weights = flat_sd["mlp_base.params"]
        num_original_weights = original_weights.shape[-1]
        padded_weights = torch.full([self.num_triplane_weights], float("nan"))
        padded_weights[:num_original_weights] = original_weights
        flat_sd["mlp_base.params"] = padded_weights

        flat_sd["triplane"] = torch.full([3, self.res, self.res, self.feat_dim], float("nan"))
        flat_sd["class_id"] = float("nan")
        flat_sd["is_triplane"] = False
    
        return grid_sd, flat_sd, reshaped_sd, class_id, nerf_id
    

class BalancedSampler(Sampler):
    def __init__(
        self, 
        mlp_dset: ReconstructionDataset, 
        triplane_dset: ReconstructionDataset,
        hash_dset: ReconstructionDataset
    ) -> None:
        assert len(mlp_dset) == len(triplane_dset) == len(hash_dset)
        self.mlp_dset = mlp_dset
        self.triplane_dset = triplane_dset
        self.hash_dset = hash_dset

    def __iter__(self) -> Iterator[int]:
        indices = []
        num_samples = len(self.mlp_dset)
        mlp_indices = np.arange(num_samples)
        np.random.shuffle(mlp_indices)
        triplane_indices = [i + num_samples for i in mlp_indices]
        hash_indices = [i + num_samples * 2 for i in mlp_indices]
        for i in range(num_samples):
            if i % 4 == 0:
                indices.append(mlp_indices[i])
                indices.append(triplane_indices[i])
            elif i % 4 == 1:
                indices.append(triplane_indices[i])
                indices.append(hash_indices[i])
            elif i % 4 == 2:
                indices.append(mlp_indices[i])
                indices.append(hash_indices[i])
            else:  # random arch couple
                j = random.randint(0, 2)
                if j == 0:
                    indices.append(mlp_indices[i])
                    indices.append(triplane_indices[i])
                elif j == 1:
                    indices.append(triplane_indices[i])
                    indices.append(hash_indices[i])
                else:
                    indices.append(mlp_indices[i])
                    indices.append(hash_indices[i])
        
        return iter(indices)

    def __len__(self) -> int:
        return 2 * len(self.mlp_dset)
