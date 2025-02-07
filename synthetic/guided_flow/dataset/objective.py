
import numpy as np
import torch

"""
This is step
"""

def J_8gaussian(x1):
    """
    Args: x1 (B, 2)
    Returns: J (B,)
    """
    # theta in [-pi, pi]
    theta = torch.atan2(x1[..., 0], x1[..., 1])
    theta_out = torch.zeros_like(theta)
    intervals = torch.linspace(-torch.pi, torch.pi, 9) + torch.pi / 8
    for i in range(len(intervals) - 1):
        mask = (theta >= intervals[i]) & (theta < intervals[i+1])
        theta_out[mask] = i
    theta_out[theta < intervals[0]] = len(intervals) - 2

    assert theta_out.shape == x1.shape[:-1]
    return theta_out    



def in_dist_8gaussian(x1, device='cpu'):
    centers = torch.tensor([
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ], device=device) * 5
    x1 = x1.to(device)
    dist = (x1[:, None, :] - centers[None, :, :]).norm(dim=-1).min(dim=-1).values
    assert dist.dim() == 1
    return dist


def metric_8gaussian(x1, device='cpu'):
    return J_8gaussian(x1.to(device)).mean() + in_dist_8gaussian(x1, device).mean()
