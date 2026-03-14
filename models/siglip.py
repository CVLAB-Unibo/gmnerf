import torch
import torch.nn.functional as F

from torch import nn, Tensor


class SiglipLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.log(torch.tensor(10.0)))  # aka temperature
        self.shift = nn.Parameter(torch.tensor(-10.0))  # aka bias

    def forward(self, emb1: Tensor, emb2: Tensor) -> Tensor:
        batch_size = emb1.size(dim=0)

        emb1 = emb1 / emb1.norm(p=2, dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(p=2, dim=-1, keepdim=True)
        
        logits = torch.matmul(emb1, emb2.t()) * self.scale.exp() + self.shift
        eye = torch.eye(batch_size).to(logits.device)
        ones = torch.ones_like(logits).to(logits.device)
        labels = 2 * eye - ones
        loss = -torch.sum(F.logsigmoid(labels * logits)) / batch_size

        return loss
