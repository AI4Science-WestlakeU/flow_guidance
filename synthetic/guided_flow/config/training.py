from dataclasses import dataclass

@dataclass
class FlowMatchingTrainingConfig:
    """
    Config for training flow matching models.
    """
    seed: int = 0
    device: str = 'cuda:0'

    cfm: str = 'cfm' # can be 'cfm', 'vp_cfm', 'ot_cfm', 'sb_cfm'
    
    # source and target distributions
    x0_dist: str = 'gaussian' # can be 'gaussian', '8gaussian', 'uniform', 'circle'
    x1_dist: str = 'gaussian' # can be 'gaussian', '8gaussian', 'uniform', 'circle'

    ### Below are fixed parameters by default
    batch_size: int = 65536
    lr: float = 1e-4
    num_steps: int = 100000
    sigma: float = 0.0 # minimum variance in CFM
    width: int = 256 # width of the MLP

    overwrite: bool = False

    def __post_init__(self):
        if self.cfm == 'ot_cfm': # If batch size is too large, OT will be too expensive
            self.batch_size = min(self.batch_size, 256)

        slow_dist = ['moon', 's_curve', 'concentric_circle']
        if self.x0_dist in slow_dist or self.x1_dist in slow_dist: # sklearn-based datasets are slow to sample from
            self.batch_size = min(self.batch_size, 2048)

