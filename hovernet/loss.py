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


class FocalLoss(nn.Module):
    # See: https://arxiv.org/abs/1708.02002
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        if not (targets.size() == logits.size()):
            raise ValueError(f"Target size ({targets.size()}) must be the same "
                             f"as input size ({logits.size()})")
        max_val = (-logits).clamp(min=0)
        loss = logits - logits * targets + max_val + \
            ((-max_val).exp() + (-logits - max_val).exp()).log()
        invprobs = F.logsigmoid(-logits * (targets * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


# Custom loss function combining Focal loss and Dice loss
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(gamma)
        self.dice_coeff = DiceCoeff()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        loss = (self.alpha * self.focal_loss(logits, targets) -
                torch.log(self.dice_coeff(logits, targets)))

        return loss.mean()


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
