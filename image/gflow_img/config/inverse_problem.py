from dataclasses import dataclass
import os


@dataclass
class InverseProblemConfig:
    seed: int = 0
    device: str = 'cuda:1'
    batch_size: int = 24  # 64:256*2*2 256:32*2*2 128:64*2

    num_channels: int = 3
    dim_image: int = 256  # 64, 128, 256
    model_data: str = 'punet256_celeba256'  # celeba , imagenet64 , imagenet128
    data_cache_dir: str = 'data_cache'
    use_dataset: bool = True  # ?

    # flow matching sampling
    flow_type: str = 'cfm'  # ot or cfm
    sigma: float = 0.01  # flow matching noise
    steps: int = 100
    ode_method: str = 'dopri5'  # euler, rk4, dopri5. Does not apply to PiGDM which uses Euler

    problem: str = 'inpainting'  # superresolution, deblurring, inpainting
    noise_type: str = 'gaussian'
    J_method: str = 'L2'  # L2, L1, exp
    
    # guidance, general settings
    clamp_x: bool = True # Whether to clamp x to [-1, 1]
    guide_method: str = 'PiGDM' # nabla_x1_J_x1,nabla_xt_J_xt, nabla_xt_J_x1, MC, PiGDM, PiGDM+
    schedule: str = 'const'  # Overall schedule. pigdm, const, linear_decay, cosine_decay, exp_decay
    schedule_ratio: bool = True # Whether to use as_score time schedule for ratio

    rt_schedule: str = 'const' # Time schedule r_t
    rt_scale: float = 1.0 # Time schedule r_t scale
    guide_scale: float = 1.0 # guidance scale. 1.0 for default PiGDM

    # PiGDM related
    start_time: float = 0.08 # 0.08 for default inpainting, 0.2 for otherss
    time_ratio_eps: float = 1e-1

    # MC related
    mc_ep: float = 1e-2
    mc_batch_size: int = 256
    mc_scale: float = 1.0
    mc_self_normalize: bool = True

    def make_path(self):
        # join strings with '-'
        exp_name = '-'.join([
            self.flow_type, self.model_data, self.problem, self.guide_method,
            self.J_method, self.ode_method, 
            # Guidance
            self.schedule, str(self.schedule_ratio), str(self.guide_scale), 
            self.rt_schedule, str(self.rt_scale), 
            str(self.start_time), str(self.time_ratio_eps), 
            str(self.clamp_x),
        ])
        return os.path.join(
            'infer', exp_name
        )