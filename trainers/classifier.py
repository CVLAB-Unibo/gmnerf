import copy
import torch
import torch.nn.functional as F
import wandb

from pathlib import Path
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.classification.accuracy import Accuracy
from typing import Any, Dict, Tuple

from nf2vec.models.fc_classifier import FcClassifier

from datasets.emb import ClassificationEmbeddingDataset
from datasets.single_arch import CLASS_ID_TO_LABEL
from trainers.utils import get_latest_ckpt_path, progress_bar


class EmbeddingClassifier:
    def __init__(
        self, 
        dset_root: Path, 
        wandb_run_name: str, 
        wandb_user: str,
        wandb_project: str
    ) -> None:
        train_dset = ClassificationEmbeddingDataset(dset_root, "train")
        self.train_loader = DataLoader(train_dset, batch_size=256, num_workers=8, shuffle=True)

        val_dset = ClassificationEmbeddingDataset(dset_root, "val")
        self.val_loader = DataLoader(val_dset, batch_size=256, num_workers=8)

        test_dset = ClassificationEmbeddingDataset(dset_root, "test")
        self.test_loader = DataLoader(test_dset, batch_size=256, num_workers=8)

        self.num_classes = len(CLASS_ID_TO_LABEL)
        net = FcClassifier([1024, 512, 256], self.num_classes)
        self.net = net.cuda()

        lr = 1e-4
        wd = 1e-2
        self.num_epochs = 150
        self.optimizer = AdamW(self.net.parameters(), lr, weight_decay=wd)
        num_steps = self.num_epochs * len(self.train_loader)
        self.scheduler = OneCycleLR(self.optimizer, lr, total_steps=num_steps)

        self.epoch = 0
        self.global_step = 0
        self.best_acc = 0.0
        
        self.run_name = wandb_run_name
        self.user = wandb_user
        self.project = wandb_project
        
        self.ckpts_dir = Path(f"ckpts") / wandb_run_name
        if self.ckpts_dir.exists():
            self.restore_from_last_ckpt()
        self.ckpts_dir.mkdir(parents=True, exist_ok=True)

    def interpolate(self, params: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        new_params = []
        new_labels = []

        for i, p1 in enumerate(params):
            same_class = labels == labels[i]
            num_same_class = torch.sum(same_class)

            if num_same_class > 2:
                indices = torch.where(same_class)[0]
                random_order = torch.randperm(len(indices))
                random_idx = indices[random_order][0]
                p2 = params[random_idx]

                random_uniform = torch.rand(len(p2))
                tsh = torch.rand(())
                from2 = random_uniform >= tsh
                p1_copy = p1.clone()
                p1_copy[from2] = p2[from2]
                new_params.append(p1_copy)
                new_labels.append(labels[i])

        if len(new_params) > 0:
            new_params = torch.stack(new_params)
            final_params = torch.cat([params, new_params], dim=0)
            new_labels = torch.stack(new_labels)
            final_labels = torch.cat([labels, new_labels], dim=0)
        else:
            final_params = params
            final_labels = labels

        return final_params, final_labels

    def train(self) -> None:
        self.wandb_init()
        start_epoch = self.epoch

        for epoch in progress_bar(range(start_epoch, self.num_epochs), desc="Epoch"):
            self.epoch = epoch
            self.net.train()

            for batch in progress_bar(self.train_loader, desc="Batch"):
                embeddings, labels = batch
                embeddings = embeddings.cuda()
                labels = labels.cuda()

                embeddings, labels = self.interpolate(embeddings, labels)

                pred = self.net(embeddings)
                loss = F.cross_entropy(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})

                self.global_step += 1

            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                self.val("train")
                self.val("val")
                self.save_ckpt()

            if epoch == self.num_epochs - 1:
                predictions, true_labels = self.val("test", best=True)
                self.log_confusion_matrix(predictions, true_labels)

    @torch.no_grad()
    def val(self, split: str, best: bool = False) -> Tuple[Tensor, Tensor]:
        acc = Accuracy("multiclass", num_classes=self.num_classes).cuda()
        predictions = []
        true_labels = []

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        if best:
            model = self.best_model
        else:
            model = self.net
        model = model.cuda()
        model.eval()

        losses = []
        for batch in progress_bar(loader, desc=f"Validating on {split} set"):
            params, labels = batch
            params = params.cuda()
            labels = labels.cuda()

            pred = model(params)
            loss = F.cross_entropy(pred, labels)
            losses.append(loss.item())

            pred_softmax = F.softmax(pred, dim=-1)
            acc(pred_softmax, labels)

            predictions.append(pred_softmax.clone())
            true_labels.append(labels.clone())

        accuracy = acc.compute()

        self.logfn({f"{split}/acc": accuracy})
        self.logfn({f"{split}/loss": torch.mean(torch.tensor(losses))})

        if accuracy > self.best_acc and split == "val":
            self.best_acc = accuracy
            self.save_ckpt(best=True)
            self.best_model = copy.deepcopy(self.net)

        return torch.cat(predictions, dim=0), torch.cat(true_labels, dim=0)

    def log_confusion_matrix(self, predictions: Tensor, labels: Tensor) -> None:
        conf_matrix = wandb.plot.confusion_matrix(
            probs=predictions.cpu().numpy(),
            y_true=labels.cpu().numpy(),
            class_names=[str(i) for i in range(self.num_classes)],
        )
        self.logfn({"conf_matrix": conf_matrix})

    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_acc": self.best_acc,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        for previous_ckpt_path in self.ckpts_dir.glob("*.pt"):
            if "best" not in previous_ckpt_path.name:
                previous_ckpt_path.unlink()

        ckpt_path = self.ckpts_dir / f"{self.epoch}.pt"
        torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_dir / "best.pt"
            torch.save(ckpt, ckpt_path)

    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_dir.exists():
            ckpt_path = get_latest_ckpt_path(self.ckpts_dir)
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_acc = ckpt["best_acc"]

            self.net.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            
    def wandb_init(self) -> None:
        wandb.init(
            name=self.run_name,
            entity=self.user,
            project=self.project,
        )
    
    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)
