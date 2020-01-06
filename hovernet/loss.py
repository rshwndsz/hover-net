# PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F


class DiceCoeff(nn.Module):
    # See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                smooth: float = 1e-7) -> torch.Tensor:
        if not (torch.max(inputs) == 1 and torch.min(inputs) >= 0):
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        iflat = probs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return ((2.0 * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class _NPBranchLoss(nn.Module):
    def __init__(self):
        super(_NPBranchLoss, self).__init__()
        self.dice_coeff = DiceCoeff()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        loss = (F.cross_entropy(logits, targets) +
                1 - self.dice_coeff(logits, targets))
        return loss


class _HVBranchLoss(nn.Module):
    def __init__(self):
        super(_HVBranchLoss, self).__init__()

    def forward(self,
                logits: torch.Tensor,
                h_grads: torch.Tensor,
                v_grads: torch.Tensor) -> torch.Tensor:
        # MSE of vertical and horizontal gradients with logits
        loss = F.mse_loss(logits, h_grads) + F.mse_loss(logits, v_grads)
        return loss


class HoverLoss(nn.Module):
    def __init__(self):
        super(HoverLoss, self).__init__()
        self.np_loss = _NPBranchLoss()
        self.hv_loss = _HVBranchLoss()

    def forward(self, np_logits, np_targets,
                hv_logits, h_grads, v_grads,
                weights=(1, 1)) -> torch.Tensor:
        loss = (self.np_loss(np_logits, np_targets) * weights[0] +
                self.hv_loss(hv_logits, h_grads, v_grads) * weights[1])
        return loss
