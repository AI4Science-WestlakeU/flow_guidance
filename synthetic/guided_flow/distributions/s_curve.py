from .base import BaseDistribution, EPS
import torch
import numpy as np
import sklearn

class SCurveDistribution(BaseDistribution):
    def __init__(self, noise: float = 0.05):
        super().__init__()
        self.noise = noise

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        X, _ = sklearn.datasets.make_s_curve(n_samples=batch_size, noise=self.noise) # (B, 2)
        x = torch.tensor(X[:, [0]], device=device, dtype=torch.float32)
        y = torch.tensor(X[:, [2]], device=device, dtype=torch.float32)
        return torch.cat([x, y / 2], dim=-1)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for s-curve is complicated, not implemented yet")
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for s-curve is complicated, not implemented yet")

    def get_J(self, x) -> torch.Tensor:
        """
        Allows grad.
        """
        return (x[:, 0] - x[:, 1]).abs() * 5

    def __name__(self):
        return f'S-Curve'
