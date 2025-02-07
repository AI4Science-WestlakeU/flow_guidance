import torch
import ot
import numpy as np

def compute_w2(x0, x1):
    """
    Compute the Wasserstein-2 distance between two distributions
    Args:
        x0: Tensor, shape (bs, dim)
            represents the source minibatch
        x1: Tensor, shape (bs, dim)
            represents the target minibatch
    Returns:
        w2: float
            the Wasserstein-2 distance between x0 and x1
    """
   # Compute the cost matrix (squared Euclidean distance)
    M = ot.dist(x0.detach().cpu().numpy(), x1.detach().cpu().numpy(), metric='sqeuclidean')

    # Compute the W2 distance
    W2_distance = np.sqrt(ot.emd2([], [], M)) 
    return W2_distance