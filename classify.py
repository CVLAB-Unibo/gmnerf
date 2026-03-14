import argparse
import os
from pathlib import Path

from trainers.classifier import EmbeddingClassifier

os.environ["WANDB_SILENT"] = "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-name", type=str, required=True)
    parser.add_argument("--wandb-user", required=True, type=str)
    parser.add_argument("--wandb-project", required=True, type=str)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "triplane", "hash"])
    args = parser.parse_args()
    
    emb_root = Path(args.data_root) / "emb" / args.ckpt_name / "shapenet" / args.arch
    run_name = f"classifier_{args.ckpt_name}"

    trainer = EmbeddingClassifier(emb_root, run_name, args.wandb_user, args.wandb_project)
    trainer.train()
