import argparse

from pathlib import Path
from torch import nn

from gmn.graph_construct.hash_grid import MultiResHashGrid
from gmn.graph_construct.layers import TriplanarGridWithInputEncoding

from nf2vec import config as nerf_cfg

from datasets.single_arch import GraphDataset, HashGridDataset, MlpDataset, TriplaneDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="shapenet", choices=["shapenet", "objaverse"])
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "triplane", "hash"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    nerf_root = data_root / "nerf" / args.dataset / args.arch
    graph_root = str(data_root / "graph" / args.dataset / args.arch)

    if args.arch == "mlp":
        enc_dim = nerf_cfg.MLP_INPUT_SIZE_AFTER_ENCODING
        hid_dim = nerf_cfg.MLP_UNITS
        out_dim = nerf_cfg.MLP_PADDED_OUTPUT_SIZE
        
        nerf = nn.Sequential(
            nn.Linear(enc_dim, hid_dim, bias=False), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
            nn.Linear(hid_dim, out_dim, bias=False)  
        )
        NerfDataset = MlpDataset
        
    elif args.arch == "triplane":
        res = nerf_cfg.TRIPLANE_RES
        feat_dim = nerf_cfg.TRIPLANE_FEAT_SIZE
        enc_dim = nerf_cfg.TRIPLANE_IN_SIZE_AFTER_ENC
        hid_dim = nerf_cfg.TRIPLANE_HID_UNITS
        out_dim = nerf_cfg.TRIPLANE_PADDED_OUT_SIZE
        
        nerf = nn.Sequential(
            TriplanarGridWithInputEncoding(res, feat_dim, enc_dim),
            nn.Linear(enc_dim + feat_dim, hid_dim, bias=False), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(),
            nn.Linear(hid_dim, out_dim, bias=False)
        )
        NerfDataset = TriplaneDataset
        
    elif args.arch == "hash":
        dim = nerf_cfg.HASH_IN_SIZE
        n_levels = nerf_cfg.HASH_LEVELS
        n_features_per_level = nerf_cfg.HASH_FEATURES_PER_ENTRY
        log2_hashmap_size = nerf_cfg.HASH_LOG2_TAB_SIZE
        base_resolution = nerf_cfg.HASH_MIN_RES
        finest_resolution = nerf_cfg.HASH_MAX_RES
        pad_in_dim = nerf_cfg.HASH_PADDED_IN_SIZE
        hid_dim = nerf_cfg.HASH_HID_UNITS
        pad_out_dim = nerf_cfg.HASH_PADDED_OUT_SIZE
        
        nerf = nn.Sequential(
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
        NerfDataset = HashGridDataset

    nerf_dset = NerfDataset(nerf_root, args.split, nerf.state_dict(), device="cpu")
    graph_dset = GraphDataset(f"{graph_root}/{args.split}", nerf_dset, nerf)
