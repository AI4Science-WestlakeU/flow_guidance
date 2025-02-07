from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class ODEConfig:
    seed: int = 0
    device: str = 'cuda:1'
    batch_size: int = 1024
    num_steps: int = 200
    solver: str = 'euler'
    t_end: float = 1


@dataclass
class GuideFnConfig:
    # general
    cfm: str = 'cfm' # 'cfm' or 'ot_cfm'
    dist_pair: Tuple[str, str] = None
    ode_cfg: ODEConfig = field(default_factory=ODEConfig)
    scale: float = 1.0 # guidance intensity. Default to 1

    guide_type: str = 'mc' # 'mc' or 'ceg' or 'learned' or 'g_cov_A' or 'g_cov_G'
    # MC
    mc_batch_size: int = 1024
    ot_std: float = 0.1
    ep: float = 1e-2
    ot_plan_batch_size: int = 256

    # learned
    gm_type: str = 'g1' # 'g1' or 'g2' or 'g3' or 'rwft'

    # gradient
    guide_scale: float = 1.0 #
    guide_schedule: str = 'linear_decay' # 'linear' or 'exp'

    # simple MC
    sim_mc_std: float = 0.1
    sim_mc_n: int = 100

    # visualization
    disp_traj_batch: int = 1024
    xlim: float = 2
    ylim: float = 2



