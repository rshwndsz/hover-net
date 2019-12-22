import torch


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
