import copy
import numpy as np
import random
import torch
import torch_geometric

from pathlib import Path
from torch import nn, Tensor
from torch_geometric.data import Data
from typing import Dict, List, Tuple

from gmn.graph_construct.model_arch_graph import sequential_to_arch, arch_to_graph

from nf2vec import config as nerf_cfg
from nf2vec.nerf.loader import NeRFLoader

from trainers.utils import progress_bar


CLASS_ID_TO_LABEL = {
    "02691156": 0,
    "02828884": 1,
    "02933112": 2,
    "02958343": 3,
    "03001627": 4,
    "03211117": 5,
    "03636649": 6,
    "03691459": 7,
    "04090263": 8,
    "04256520": 9,
    "04379243": 10,
    "04401088": 11,
    "04530566": 12
}

CLASS_ID_TO_NAME = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "speaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "phone",
    "04530566": "watercraft"
}


class NerfDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, sample_sd: Dict[str, Tensor]) -> None:
        super().__init__()
        
        grid_paths = []
        nerf_paths = []
        with open(root.parent / f"{split}.txt", "r") as f:
            for line in f:
                path = Path(line.strip())
                grid_path = root / path / "grid.pth"
                nerf_path = root / path / "nerf_weights.pth"
                grid_paths.append(grid_path)
                nerf_paths.append(nerf_path)
        self.grid_paths = grid_paths
        self.nerf_paths = nerf_paths
        
        self.sample_sd = sample_sd
        self.num_weights = [np.prod(weights.shape) for weights in sample_sd.values()]

    def __len__(self) -> int:
        return len(self.nerf_paths)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict, str, str]:
        raise NotImplementedError
    

class MlpDataset(NerfDataset):
    def __init__(
        self, 
        root: Path, 
        split: str, 
        sample_sd: Dict[str, Tensor], 
        device: str
    ) -> None:
        super().__init__(root, split, sample_sd)
        self.device = device

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict, str, str]:
        grid_sd = torch.load(self.grid_paths[idx], map_location=self.device)
        flat_sd = torch.load(self.nerf_paths[idx], map_location=self.device)
        flat_weights = flat_sd["mlp_base.params"]
        reshaped_sd = {key: torch.zeros(tensor.shape, device=self.device) for key, tensor in self.sample_sd.items()}
        
        start = 0
        for i, (key, sample_tensor) in enumerate(self.sample_sd.items()):
            flat_tensor = flat_weights[start : start + self.num_weights[i]]
            reshaped_tensor = flat_tensor.reshape(sample_tensor.shape)
            reshaped_sd[key] = reshaped_tensor
            start += self.num_weights[i]

        flat_sd["encoding.params"] = torch.tensor([])
            
        nerf_path = self.nerf_paths[idx]
        class_id = nerf_path.parent.parent.stem
        nerf_id = nerf_path.parent.stem
            
        return grid_sd, flat_sd, reshaped_sd, class_id, nerf_id


class TriplaneDataset(NerfDataset):
    def __init__(
        self, 
        root: Path, 
        split: str, 
        sample_sd: Dict[str, Tensor], 
        device: str
    ) -> None:
        super().__init__(root, split, sample_sd)
        self.device = device

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict, str, str]:
        grid_sd = torch.load(self.grid_paths[idx], map_location=self.device)
        flat_sd = torch.load(self.nerf_paths[idx], map_location=self.device)
        triplane = flat_sd["triplane"]
        flat_weights = flat_sd["mlp_base.params"]
        reshaped_sd = {key: torch.zeros(tensor.shape, device=self.device) for key, tensor in self.sample_sd.items()}
        
        start = 0
        for i, (key, sample_tensor) in enumerate(self.sample_sd.items()):
            if i == 0:
                reshaped_sd[key] = triplane.reshape(sample_tensor.shape)
            else:
                flat_tensor = flat_weights[start : start + self.num_weights[i]]
                reshaped_tensor = flat_tensor.reshape(sample_tensor.shape)
                reshaped_sd[key] = reshaped_tensor
                start += self.num_weights[i]

        flat_sd["is_triplane"] = True
            
        nerf_path = self.nerf_paths[idx]
        class_id = nerf_path.parent.parent.stem
        nerf_id = nerf_path.parent.stem
            
        return grid_sd, flat_sd, reshaped_sd, class_id, nerf_id
    

class HashGridDataset(NerfDataset):
    def __init__(
        self, 
        root: Path, 
        split: str, 
        sample_sd: Dict[str, Tensor], 
        device: str
    ) -> None:
        super().__init__(root, split, sample_sd)
        self.device = device

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict, str, str]:
        grid_sd = torch.load(self.grid_paths[idx], map_location=self.device)
        flat_sd = torch.load(self.nerf_paths[idx], map_location=self.device)
        flat_tables = flat_sd["encoding.params"]
        flat_weights = flat_sd["mlp_base.params"]
        reshaped_sd = {key: torch.zeros(tensor.shape, device=self.device) for key, tensor in self.sample_sd.items()}
        
        start_hash = 0
        start_mlp = 0
        for i, (key, sample_tensor) in enumerate(self.sample_sd.items()):
            if "levels" in key:
                flat_tensor = flat_tables[start_hash: start_hash + self.num_weights[i]]
                reshaped_tensor = flat_tensor.reshape(sample_tensor.shape)
                reshaped_sd[key] = reshaped_tensor
                start_hash += self.num_weights[i]
            else:
                flat_tensor = flat_weights[start_mlp : start_mlp + self.num_weights[i]]
                reshaped_tensor = flat_tensor.reshape(sample_tensor.shape)
                reshaped_sd[key] = reshaped_tensor
                start_mlp += self.num_weights[i]

        flat_sd["is_hash"] = True
            
        nerf_path = self.nerf_paths[idx]
        class_id = nerf_path.parent.parent.stem
        nerf_id = nerf_path.parent.stem
            
        return grid_sd, flat_sd, reshaped_sd, class_id, nerf_id


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, root: str, nerf_dset: NerfDataset, sample_nerf: nn.Sequential) -> None:
        self.nerf_dset = nerf_dset
        self.sample_nerf = copy.deepcopy(sample_nerf)
        super().__init__(root)
    
    @property
    def processed_file_names(self) -> List[str]:
        return [f"{idx}.pt" for idx in range(len(self.nerf_dset))]
    
    def process(self) -> None:
        for idx in progress_bar(range(len(self.nerf_dset)), desc="Dataset processing"):
            _, _, sd, class_id, nerf_id = self.nerf_dset[idx]
            self.sample_nerf.load_state_dict(sd)
            
            arch = sequential_to_arch(self.sample_nerf)
            x, edge_index, edge_attr = arch_to_graph(arch)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data = data.detach()
            data = {
                "data": data,
                "class_id": class_id,
                "nerf_id": nerf_id
            }
            torch.save(data, self.processed_paths[idx])
    
    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> Tuple[Dict, Dict, Data, str, str]:
        grid_sd, flat_sd, _, class_id, nerf_id = self.nerf_dset[idx]
        data = torch.load(self.processed_paths[idx])
        graph_data = data["data"]
        
        return grid_sd, flat_sd, graph_data, class_id, nerf_id


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dset: GraphDataset, device: str) -> None:
        super().__init__()
        self.graph_dset = graph_dset
        self.device = device

    def __len__(self) -> int:
        return len(self.graph_dset)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict, Data, Dict, Tensor, str, str]:
        nerf_dir = self.graph_dset.nerf_dset.nerf_paths[idx].parent

        nerf_loader = NeRFLoader(
            data_dir=nerf_dir,
            num_rays=nerf_cfg.NUM_RAYS,
            device=self.device)

        nerf_loader.training = True
        data = nerf_loader[0]  # NOTE: when training is True, __getitem__ returns
                               # a random element regardless of the index (0)
        color_bkgd = data["color_bkgd"]
        rays = data["rays"]
        train_dict = {
            "rays": rays,
            "color_bkgd": color_bkgd
        }

        nerf_loader.training = False
        test_data = nerf_loader[0]  
        test_color_bkgd = test_data["color_bkgd"]
        test_rays = test_data["rays"]
        test_dict = {
            "rays": test_rays,
            "color_bkgd": test_color_bkgd
        }

        grid_weights, nerf_weights, graph_data, class_id, nerf_id = self.graph_dset[idx]

        grid_weights["_binary"] = grid_weights["_binary"].to_dense()
        n_total_cells = nerf_cfg.GRID_NUMBER_OF_CELLS
        grid_weights["occs"] = torch.empty([n_total_cells], device=grid_weights["_binary"].device) 
        
        N = nerf_cfg.GRID_BACKGROUND_CELLS_TO_SAMPLE
        background_indices = self._sample_unoccupied_cells(N, grid_weights["_binary"], nerf_dir)

        return train_dict, test_dict, nerf_weights, graph_data, grid_weights, background_indices, class_id, nerf_id
    
    def _sample_unoccupied_cells(self, n: int, binary: Tensor, data_dir: Path) -> Tensor:
        zero_indices = torch.nonzero(binary.flatten() == 0)[:, 0]
        if len(zero_indices) < n:
            print(f"ERROR: {len(zero_indices)} - {data_dir}")

        randomized_indices = random.sample(range(0, len(zero_indices)), n)
        randomized_indices = zero_indices[randomized_indices]
        
        return randomized_indices
