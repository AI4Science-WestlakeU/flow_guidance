from .base import BaseDistribution
import torch
import math

class GaussianDistribution(BaseDistribution):
    def __init__(
        self, 
        mean: torch.Tensor = torch.zeros(2), 
        std: torch.Tensor = torch.ones(2)
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        eps = torch.randn(batch_size, 2, device=device)
        return eps * self.std.to(device) + self.mean.to(device)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((x - self.mean.to(x.device)) / self.std.to(x.device)).pow(2).sum(-1) - torch.log(self.std.to(x.device)).sum(-1) - 0.5 * math.log(2 * torch.pi)

    def get_J(self) -> torch.Tensor:
        raise NotImplementedError("J for gaussian is not implemented yet")

    def __name__(self):
        return f'Gaussian'

class EightGaussiansDistribution(BaseDistribution):
    def __init__(
        self, 
        centers: torch.Tensor = torch.tensor(
            [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
                (1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
                (-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
                (-1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
            ], dtype=torch.float32
        ), 
        std: torch.Tensor = torch.ones(8, 2) * 0.1
    ):
        super().__init__()
        self.centers = centers
        self.std = std

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        multi = torch.multinomial(torch.ones(8, device=device), batch_size, replacement=True) # size (batch_size,)
        eps = torch.randn(batch_size, 2, device=device)
        centers = self.centers.to(device).repeat(batch_size, 1, 1)[torch.arange(batch_size, device=device), multi]
        stds = self.std.to(device).repeat(batch_size, 1, 1)[torch.arange(batch_size, device=device), multi]
        return centers + eps * stds
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.prob(x).clamp(1e-10).log()

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate with the PDF of 8 non-overlapping gaussians.
        """
        one_log_prob = -0.5 * ((x.unsqueeze(1) - self.centers.to(x.device).unsqueeze(0)) / self.std.to(x.device).unsqueeze(0)).pow(2).sum(-1) # (B, 8, 2) -> (B, 8)
        one_log_prob_z = torch.log(self.std.to(x.device)).sum(-1) - 0.5 * math.log(2 * torch.pi) # (B, 8)
        prob = (torch.exp(one_log_prob) / torch.exp(one_log_prob_z)).mean(-1)
        return prob

    def get_J(self, x1: torch.Tensor) -> torch.Tensor:
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

    def __name__(self):
        return f'8 Gaussians'
