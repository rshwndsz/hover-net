# Python STL
import logging
from typing import List
# PyTorch
import torch

# Local
from hovernet import utils

logger = logging.getLogger(__name__)


def dice_score(probs: torch.Tensor,
               targets: torch.Tensor,
               threshold: float = 0.5) -> torch.Tensor:
    """Calculate Sorenson-Dice coefficient

    Parameters
    ----------
    probs : torch.Tensor
        Probabilities
    targets : torch.Tensor
        Ground truths
    threshold : float
        probs > threshold => 1
        probs <= threshold => 0

    Returns
    -------
    dice : torch.Tensor
        Dice score

    See Also
    --------
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """

    batch_size: int = targets.shape[0]
    with torch.no_grad():
        # Shape: [N, C, H, W]targets
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        # Shape: [N, C*H*W]
        if not (probs.shape == targets.shape):
            raise ValueError(f"Shape of probs: {probs.shape} must be the same"
                             f"as that of targets: {targets.shape}.")
        # Only 1's and 0's in p & t
        p = utils.predict(probs, threshold)
        t = utils.predict(targets, 0.5)
        # Shape: [N, 1]
        dice = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

    return utils.nanmean(dice)


def true_positive(preds: torch.Tensor,
                  targets: torch.Tensor,
                  num_classes: int = 2) -> torch.Tensor:
    """Compute number of true positive predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    tp : torch.Tensor
        Tensor of number of true positives for each class
    """
    out: List[torch.Tensor] = []
    for i in range(num_classes):
        out.append(((preds == i) & (targets == i)).sum())

    return torch.tensor(out)


def true_negative(preds: torch.Tensor,
                  targets: torch.Tensor,
                  num_classes: int) -> torch.Tensor:
    """Computes number of true negative predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    tn : torch.Tensor
        Tensor of true negatives for each class
    """
    out: List[torch.Tensor] = []
    for i in range(num_classes):
        out.append(((preds != i) & (targets != i)).sum())

    return torch.tensor(out)


def false_positive(preds: torch.Tensor,
                   targets: torch.Tensor,
                   num_classes: int) -> torch.Tensor:
    """Computes number of false positive predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    fp : torch.Tensor
        Tensor of false positives for each class
    """
    out: List[torch.Tensor] = []
    for i in range(num_classes):
        out.append(((preds == i) & (targets != i)).sum())

    return torch.tensor(out)


def false_negative(preds: torch.Tensor,
                   targets: torch.Tensor,
                   num_classes: int) -> torch.Tensor:
    """Computes number of false negative predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    fn : torch.Tensor
        Tensor of false negatives for each class
    """
    out: List[torch.Tensor] = []
    for i in range(num_classes):
        out.append(((preds != i) & (targets == i)).sum())

    return torch.tensor(out)


def precision_score(preds: torch.Tensor,
                    targets: torch.Tensor,
                    num_classes: int = 2) -> torch.Tensor:
    """Computes precision score

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    precision : Tuple[torch.Tensor, ...]
        List of precision scores for each class
    """
    tp = true_positive(preds, targets, num_classes).to(torch.float)
    fp = false_positive(preds, targets, num_classes).to(torch.float)
    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def accuracy_score(preds: torch.Tensor,
                   targets: torch.Tensor,
                   smooth: float = 1e-10) -> torch.Tensor:
    """Compute accuracy score

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    smooth: float
        Smoothing for numerical stability
        1e-10 by default

    Returns
    -------
    acc : torch.Tensor
        Average accuracy score
    """
    valids = (targets >= 0)
    acc_sum = (valids * (preds == targets)).sum().float()
    valid_sum = valids.sum().float()
    return acc_sum / (valid_sum + smooth)


def iou_score(preds: torch.Tensor,
              targets: torch.Tensor,
              smooth: float = 1e-7) -> torch.Tensor:
    """Computes IoU or Jaccard index

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    smooth: float
        Smoothing for numerical stability
        1e-10 by default

    Returns
    -------
    iou : torch.Tensor
        IoU score or Jaccard index
    """
    intersection = torch.sum(targets * preds)
    union = torch.sum(targets) + torch.sum(preds) - intersection + smooth
    score = (intersection + smooth) / union

    return score
