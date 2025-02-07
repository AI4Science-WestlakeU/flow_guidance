from .base import BaseDistribution
import torch

class SpiralDistribution(BaseDistribution):
    def __init__(self, noise: float = 0.1):
        super().__init__()
        self.noise = noise

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        theta = torch.linspace(0, 4 * torch.pi, batch_size, device=device)
        r = torch.linspace(0, 4, batch_size, device=device)
        x1 = r * torch.cos(theta) + self.noise * torch.rand(batch_size, device=device)
        x2 = r * torch.sin(theta) + self.noise * torch.rand(batch_size, device=device)
        return torch.stack([x1, x2], dim=-1) / 3

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Log probability for spiral is not implemented yet")

    def get_J(self, x: torch.Tensor) -> torch.Tensor:
        return x.square().sum(dim=-1) * 2
