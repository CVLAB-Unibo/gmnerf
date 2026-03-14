import argparse
import h5py
import torch

from pathlib import Path
from torch_geometric.loader import DataLoader

from nf2vec import config as nerf_cfg

from datasets.graph import ExportedGraphDataset
from models.gnn import GNN
from trainers.utils import progress_bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-name", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="shapenet", choices=["shapenet", "objaverse"])
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "triplane", "hash"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--num-gnn-layers", type=int, default=4)
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    graph_root = data_root / args.dataset / "graph" / args.arch
    graph_dset = ExportedGraphDataset(graph_root, args.split)
    loader = DataLoader(graph_dset, batch_size=1, shuffle=False)

    encoder = GNN(
        args.gnn_hidden_dim,
        nerf_cfg.ENCODER_EMBEDDING_DIM,
        args.num_gnn_layers
    )
    ckpt = torch.load(f"ckpts/{args.ckpt_name}/best.pt")
    ckpt = ckpt["encoder"]
    encoder.load_state_dict(ckpt)
    encoder.cuda()
    encoder.eval()

    for idx, sample in enumerate(progress_bar(loader, desc="Sample")):
        graph_data, class_id, nerf_id = sample
        graph_data = graph_data.cuda()
        graph_data.x = graph_data.x.float()

        with torch.no_grad():
            emb = encoder(
                graph_data.x, 
                graph_data.edge_index, 
                graph_data.edge_attr, 
                graph_data.batch
            )

        emb_root = data_root / "emb" / args.ckpt_name / args.dataset / args.arch
        emb_path = emb_root / args.split / f"{idx}.h5"
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(emb_path, "w") as f:
            f.create_dataset("embedding", data=emb[0].detach().cpu().numpy())
            f.create_dataset("class_id", data=class_id[0])
            f.create_dataset("nerf_id", data=nerf_id[0])
