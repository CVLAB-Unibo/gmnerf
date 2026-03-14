import h5py
import numpy as np
import torch

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, Tuple

from nf2vec import config as nerf_cfg

from datasets.single_arch import CLASS_ID_TO_LABEL
    

class ClassificationEmbeddingDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.emb_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.emb_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.emb_paths[idx], "r") as f:
            embedding = np.array(f["embedding"])
            embedding = torch.from_numpy(embedding)
            class_id = np.array(f["class_id"]).item().decode("utf-8")
            label = np.array(CLASS_ID_TO_LABEL[class_id])
            label = torch.from_numpy(label).long()

        return embedding, label
    

class RetrievalEmbeddingDataset(Dataset):
    def __init__(
        self, 
        emb_root: Path, 
        query_nerf_root: Path, 
        gallery_nerf_root: Path, 
        split: str,
        query_arch: str,
        gallery_arch: str
    ) -> None:
        super().__init__()

        query_emb_root = emb_root / query_arch / split
        gallery_emb_root = emb_root / gallery_arch / split
        query_emb_paths = sorted(query_emb_root.glob("*.h5"), key=lambda x: int(x.stem))
        gallery_emb_paths = sorted(gallery_emb_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.emb_paths = query_emb_paths + gallery_emb_paths

        grid_paths = []
        nerf_paths = []
        nerf_archs = []
        for nerf_root, nerf_arch in zip([query_nerf_root, gallery_nerf_root], [query_arch, gallery_arch]):
            with open(nerf_root.parent / f"{split}.txt", "r") as f:
                for line in f:
                    path = Path(line.strip())
                    grid_path = nerf_root / path / "grid.pth"
                    nerf_path = nerf_root / path / "nerf_weights.pth"
                    grid_paths.append(grid_path)
                    nerf_paths.append(nerf_path)
                    nerf_archs.append(nerf_arch)
        self.grid_paths = grid_paths
        self.nerf_paths = nerf_paths
        self.nerf_archs = nerf_archs

    def __len__(self) -> int:
        return len(self.emb_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict, Dict, Tensor, str, str]:
        with h5py.File(self.emb_paths[idx], "r") as f:
            class_id = np.array(f["class_id"]).item().decode("utf-8")
            nerf_id = np.array(f["nerf_id"]).item().decode("utf-8")

            grid_path = self.grid_paths[idx]
            nerf_path = self.nerf_paths[idx]
            nerf_arch = self.nerf_archs[idx]

            assert class_id == nerf_path.parent.parent.stem
            assert nerf_id == nerf_path.parent.stem

            embedding = np.array(f["embedding"])
            embedding = torch.from_numpy(embedding)

            grid_weights = torch.load(grid_path)
            nerf_weights = torch.load(nerf_path)

            grid_weights["_binary"] = grid_weights["_binary"].to_dense()
            n_total_cells = nerf_cfg.GRID_NUMBER_OF_CELLS
            grid_weights["occs"] = torch.empty([n_total_cells])
            
            nerf_weights["nerf_arch"] = nerf_arch
            is_trip = nerf_arch == "triplane"
            is_hash = nerf_arch == "hash"
            if "encoding.params" not in nerf_weights:
                nerf_weights["encoding.params"] = torch.tensor([])
            nerf_weights["is_triplane"] = is_trip
            nerf_weights["is_hash"] = is_hash

            label = np.array(CLASS_ID_TO_LABEL[class_id])
            label = torch.from_numpy(label).long()

        return embedding, grid_weights, nerf_weights, label, class_id, nerf_id
