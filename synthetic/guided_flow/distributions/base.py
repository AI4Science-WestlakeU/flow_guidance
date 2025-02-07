import torch
from abc import ABC, abstractmethod



EPS = 1e-10


class BaseDistribution(ABC):
    """
    Base class for all distributions.
    """
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        """
        Returns: (B, 2)
        """
        pass

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_J(self, x1) -> torch.Tensor:
        pass

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(x))


    def __str__(self):
        return self.__class__.__name__
    

def get_distribution(dist_name: str) -> BaseDistribution:
    # import distributions here to avoid circular imports
    from guided_flow.distributions.uniform import UniformDistribution
    from guided_flow.distributions.laplace import LaplaceDistribution
    from guided_flow.distributions.gaussian import GaussianDistribution, EightGaussiansDistribution
    from guided_flow.distributions.circle import CircleDistribution, ConcentricCircleDistribution
    from guided_flow.distributions.moon import MoonDistribution
    from guided_flow.distributions.s_curve import SCurveDistribution
    from guided_flow.distributions.checkerboard import CheckerboardDistribution
    from guided_flow.distributions.spiral import SpiralDistribution

    if dist_name == 'gaussian':
        return GaussianDistribution()
    elif dist_name.startswith('gaussian_std'): # e.g. gaussian_std_0.1
        std = float(dist_name.split('_')[2])
        return GaussianDistribution(std=torch.ones(2) * std)
    elif dist_name == '8gaussian':
        return EightGaussiansDistribution()
    elif dist_name == 'uniform':
        return UniformDistribution()
    elif 'uniform_w' in dist_name:
        w = float(dist_name.split('_')[2])
        return UniformDistribution(low=-w, high=w)
    elif dist_name == 'laplace':
        return LaplaceDistribution()
    elif dist_name == 'circle':
        return CircleDistribution()
    elif dist_name == 'concentric_circle':
        return ConcentricCircleDistribution()
    elif dist_name == 'moon':
        return MoonDistribution()
    elif dist_name == 's_curve':
        return SCurveDistribution()
    elif dist_name == 'checkerboard':
        return CheckerboardDistribution()
    elif dist_name == 'laplace':
        return LaplaceDistribution()
    elif dist_name == 'spiral':
        return SpiralDistribution()
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    
