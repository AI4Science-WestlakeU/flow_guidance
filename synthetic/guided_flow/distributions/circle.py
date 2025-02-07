import math
from .base import BaseDistribution, EPS
import torch
import sklearn

class CircleDistribution(BaseDistribution):
    def __init__(self, r: float = 1, sigma: float = 0.05):
        super().__init__()
        self.r = r
        self.sigma = sigma

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        theta = torch.rand(batch_size, device=device) * 2 * torch.pi
        r = self.r + torch.randn(batch_size, device=device) * self.sigma
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=-1)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        print('here')
        l = 2 * torch.pi * self.r
        r_x = (x[:, 0] ** 2 + x[:, 1] ** 2).sqrt() # (B,)
        log_p_gaussian_along_r = -0.5 * torch.log(2 * torch.pi * self.sigma ** 2) - (r_x - self.r) ** 2 / 2 / self.sigma ** 2 # (B,)
        return log_p_gaussian_along_r - math.log(l)
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        l = 2 * torch.pi * self.r
        r_x = (x[:, 0] ** 2 + x[:, 1] ** 2).sqrt() # (B,)
        p_gaussian_along_r = 1 / math.sqrt(2 * torch.pi * self.sigma ** 2) * torch.exp(-(r_x - self.r) ** 2 / 2 / self.sigma ** 2) # (B,)
        return p_gaussian_along_r / l

    def get_J(self, x) -> torch.Tensor:
        """
        Allows Gradient
        """
        return (((x[:, 0] - 1).square() + (x[:, 1] - 1).square() - 1).abs() * 5).clamp(0, 10) # (B,)
    
    def __name__(self):
        return f'Circle'
        

class ConcentricCircleDistribution(BaseDistribution):
    def __init__(self, r: float = 0.5, sigma: float = 0.05):
        super().__init__()
        self.r = r
        self.sigma = sigma

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        samples, _ = sklearn.datasets.make_circles(n_samples=batch_size, noise=self.sigma, factor=self.r) # (B, 2)
        return torch.tensor(samples, device=device, dtype=torch.float32)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for concentric circles is complicated, not implemented yet")
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for concentric circles is complicated, not implemented yet")

    def get_J(self, x) -> torch.Tensor:
        return ((x[:, 0] ** 2 + x[:, 1] ** 2) - 1).abs().clamp(0)
