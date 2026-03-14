import argparse
import os

from pathlib import Path
from torch import nn

from gmn.graph_construct.hash_grid import MultiResHashGrid
from gmn.graph_construct.layers import TriplanarGridWithInputEncoding

from nf2vec import config as nerf_cfg

from trainers.l_rec import RecontructionTrainer
from trainers.l_rec_con import RecontructionContrastiveTrainer
from trainers.l_con import ContrastiveTrainer

os.environ["WANDB_SILENT"] = "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", required=True, type=str, choices=["l_rec", "l_rec_con", "l_con"])
    parser.add_argument("--wandb-user", required=True, type=str)
    parser.add_argument("--wandb-project", required=True, type=str)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--num-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--weight-decay", type=int, default=1e-2)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--num-gnn-layers", type=int, default=4)
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    mlp_nerf_root = data_root / "nerf" / "shapenet" / "mlp"
    mlp_graph_root = str(data_root / "graph" / "shapenet" / "mlp")

    enc_dim = nerf_cfg.MLP_INPUT_SIZE_AFTER_ENCODING
    hid_dim = nerf_cfg.MLP_UNITS
    out_dim = nerf_cfg.MLP_PADDED_OUTPUT_SIZE

    mlp_nerf = nn.Sequential(
        nn.Linear(enc_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, out_dim, bias=False)  
    )

    triplane_nerf_root = data_root / "nerf" / "shapenet" / "triplane"
    triplane_graph_root = str(data_root / "graph" / "shapenet" / "triplane")

    res = nerf_cfg.TRIPLANE_RES
    feat_dim = nerf_cfg.TRIPLANE_FEAT_SIZE
    enc_dim = nerf_cfg.TRIPLANE_IN_SIZE_AFTER_ENC
    hid_dim = nerf_cfg.TRIPLANE_HID_UNITS
    out_dim = nerf_cfg.TRIPLANE_PADDED_OUT_SIZE

    triplane_nerf = nn.Sequential(
        TriplanarGridWithInputEncoding(res, feat_dim, enc_dim),
        nn.Linear(enc_dim + feat_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, out_dim, bias=False)
    )

    hash_nerf_root = data_root / "nerf" / "shapenet" / "hash"
    hash_graph_root = str(data_root / "graph" / "shapenet" / "hash")

    dim = nerf_cfg.HASH_IN_SIZE
    n_levels = nerf_cfg.HASH_LEVELS
    n_features_per_level = nerf_cfg.HASH_FEATURES_PER_ENTRY
    log2_hashmap_size = nerf_cfg.HASH_LOG2_TAB_SIZE
    base_resolution = nerf_cfg.HASH_MIN_RES
    finest_resolution = nerf_cfg.HASH_MAX_RES
    pad_in_dim = nerf_cfg.HASH_PADDED_IN_SIZE
    hid_dim = nerf_cfg.HASH_HID_UNITS
    pad_out_dim = nerf_cfg.HASH_PADDED_OUT_SIZE

    hash_nerf = nn.Sequential(
        MultiResHashGrid(
            dim,
            n_levels,
            n_features_per_level,
            log2_hashmap_size,
            base_resolution,
            finest_resolution
        ),
        nn.Linear(pad_in_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
        nn.Linear(hid_dim, pad_out_dim, bias=False)
    )
    
    if args.loss == "l_rec":
        Trainer = RecontructionTrainer
    elif args.loss == "l_rec_con":
        Trainer = RecontructionContrastiveTrainer
    else:
        Trainer = ContrastiveTrainer
    trainer = Trainer(
        mlp_nerf_root,
        mlp_graph_root,
        triplane_nerf_root,
        triplane_graph_root,
        hash_nerf_root,
        hash_graph_root,
        mlp_nerf,
        triplane_nerf,
        hash_nerf,
        args.num_epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.gnn_hidden_dim,
        args.num_gnn_layers,
        args.wandb_user,
        args.wandb_project,
        args.wandb_run_name,
    )
    trainer.train()
