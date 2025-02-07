from .base import BaseDistribution, EPS
import torch

class UniformDistribution(BaseDistribution):
    def __init__(self, low: float = -1, high: float = 1, device: str = 'cuda'):
        super().__init__()
        self.low = low
        self.high = high
        self.dim = 2
        self.device = device

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        return torch.rand(batch_size, self.dim, device=device) * (self.high - self.low) + self.low

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.prob(x).log()
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype) + EPS
        zeros[(x[:, 0] >= self.low) & (x[:, 0] <= self.high) & (x[:, 1] >= self.low) & (x[:, 1] <= self.high)] = 1
        return zeros * 1 / (self.high - self.low) ** self.dim
    
    def get_J(self) -> torch.Tensor:
        raise NotImplementedError("J for uniform is not implemented yet")

    def __name__(self):
        return f'Uniform'
