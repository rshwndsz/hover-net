# Python STL
from typing import Tuple
# PyTorch
import torch
from torch.nn import functional as F


def nanmean(v: torch.Tensor,
            *args,
            inplace: bool = False,
            **kwargs):
    """Compute the arithmetic mean along the specified axis, ignoring NaNs

    Parameters
    ----------
    v: torch.Tensor
    args
    inplace: bool
    kwargs

    Returns
    -------
    torch.Tensor
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def predict(probs: torch.Tensor,
            threshold: float) -> torch.Tensor:
    """Thresholding probabilities

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities i.e. values from 0 to 1
    threshold: float
        probs > threshold => 1
        probs <= threshold => 0

    Returns
    -------
    predictions: torch.Tensor
        Thresholded probabilities
        Contains only 0, 1 and has the same shape as `probs`
    """
    return (probs > threshold).float()


def get_sobel_filter(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get sobel filter of particular size

    Parameters
    ----------
    size : int
        Size of sobel filter

    Returns
    -------
    kernel_h : torch.Tensor
        Horizontal sobel filter of size `size`
    kernel_v : torch.Tensor
        Vertical sobel filter of size `size`
    """
    assert size % 2 == 1, "Size must be odd"

    h_range = torch.arange(-size//2 + 1, size//2 + 1, dtype=torch.float32)
    v_range = torch.arange(-size//2 + 1, size//2 + 1, dtype=torch.float32)
    h, v = torch.meshgrid([h_range, v_range])
    h, v = h.transpose(0, 1), v.transpose(0, 1)

    kernel_h = h / (h*h + v*v + 1e-15)
    kernel_v = v / (h*h + v*v + 1e-15)

    return kernel_h, kernel_v


def get_gradient_hv(logits: torch.Tensor,
                    h_ch: int = 1,
                    v_ch: int = 0) -> torch.Tensor:
    """Get horizontal & vertical gradients

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits from HV branch
    h_ch : int
        Number of horizontal channels
    v_ch : int
        Number of vertical channels

    Returns
    -------
    gradients : torch.Tensor
        concatenated output [dh, dv]
    """
    mh, mv = get_sobel_filter(size=5)
    mh = mh.reshape(shape=(1, 1, 5, 5))
    mv = mv.reshape(shape=(1, 1, 5, 5))

    hl = logits[..., h_ch].unsqueeze(dim=-1)
    vl = logits[..., v_ch].unsqueeze(dim=-1)

    dh = F.conv2d(hl, mh, stride=1, padding=2)
    dv = F.conv2d(vl, mv, stride=1, padding=2)

    out = torch.cat([dh, dv])
    return out
