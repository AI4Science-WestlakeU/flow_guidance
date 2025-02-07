from dataclasses import dataclass

@dataclass
class GuidanceTrainingConfig:
    """
    Config for training flow matching models.
    """
    seed: int = 0
    device: str = 'cuda:0'
    # 
    x0_dist: str = 'gaussian' # can be 'gaussian', '8gaussian', 'uniform', 'circle'
    x1_dist: str = '8gaussian' # can be 'gaussian', '8gaussian', 'uniform', 'circle'

    cfm: str = 'cfm' # can be 'cfm', 'vp_cfm', 'ot_cfm', 'sb_cfm'
    which_model: str = 'z-g-g2-g3-rwft' # can be 'g', 'g2', 'g3', 'rwft'

    # Guidance
    scale: float = 1.0 # scale of the guidance
    # Below are fixed parameters by default
    batch_size: int = 65536
    ot_batch_size: int = 256
    lr: float = 1e-4
    num_steps: int = 1000
    sigma: float = 0.0 # minimum variance in CFM
    width: int = 256 # width of the MLP

    def __post_init__(self):
        if self.cfm == 'ot_cfm': # If batch size is too large, OT will be too expensive
            self.batch_size = min(self.batch_size, 1024)

        slow_dist = ['moon', 's_curve']
        if self.x0_dist in slow_dist or self.x1_dist in slow_dist: # sklearn-based datasets are slow to sample from
            self.batch_size = min(self.batch_size, 2048)

