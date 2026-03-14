import torch

from pathlib import Path
from torch_geometric.data import Data
from torch.utils.data import Dataset
from typing import Tuple


class ExportedGraphDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split / "processed"
        paths = list(self.root.glob("*.pt"))
        paths = [p for p in paths if p.stem.isdigit()]
        self.paths = sorted(paths, key=lambda p: int(p.stem))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Data, str, str]:
        data = torch.load(self.paths[idx])
        graph_data = data["data"]
        class_id = data["class_id"]
        nerf_id = data["nerf_id"]

        return graph_data, class_id, nerf_id
