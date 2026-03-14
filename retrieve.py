import argparse
import os
import wandb

from pathlib import Path
from retrieval_utils import run_retrieval

os.environ["WANDB_SILENT"] = "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-name", type=str, required=True)
    parser.add_argument("--wandb-user", required=True, type=str)
    parser.add_argument("--wandb-project", required=True, type=str)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--query-arch", type=str, default="mlp", choices=["mlp", "triplane", "hash"])
    parser.add_argument("--gallery-arch", type=str, default="triplane", choices=["mlp", "triplane", "hash"])
    args = parser.parse_args()
    
    assert args.query_arch != args.gallery_arch
     
    data_root = Path(args.data_root)
    emb_root = data_root / "emb" / args.ckpt_name / "shapenet"
    query_nerf_root = data_root / "nerf" / "shapenet" / args.query_arch
    gallery_nerf_root = data_root / "nerf" / "shapenet" / args.gallery_arch

    run_name = f"{args.ckpt_name}_query_{args.query_arch}_gallery_{args.gallery_arch}"
    save_root = Path("retrieval") / run_name
    run_name = f"retrieval_{run_name}"
    
    split = "test"

    wandb.init(
        name=run_name,
        entity=args.wandb_user,
        project=args.wandb_project
    )
    run_retrieval(
        emb_root, 
        query_nerf_root, 
        gallery_nerf_root, 
        save_root, 
        split, 
        args.query_arch, 
        args.gallery_arch
    )
