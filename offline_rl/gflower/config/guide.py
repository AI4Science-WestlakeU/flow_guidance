from dataclasses import dataclass, field
from gflower.config.flow_matching import TransformerConfig

@dataclass
class GuideTrainingConfig:
    seed: int = 0
    device: str = 'cuda:0'

    log_folder: str = 'logs'
    exp_name: str = 'value'

    # environment
    env: str = 'hopper-medium-replay'
    horizon: int = 20
    inf_horizon: bool = True # whether calculate value for infinite horizon
    normalizer: str = 'GaussianNormalizer'
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 100000
    max_n_episodes: int = 100000
    termination_penalty: float = 0
    termination_penalty: float = -100

    state_dim: int = 17
    action_dim: int = 6
  
    # model
    train_z: bool = False # need to train z first
    transformer_config: TransformerConfig = field(default=TransformerConfig(depth=8, num_heads=8, hidden_size=256))
    scale: float = 1.0 # return-to-go scale for guidance
    discount: float = 0.99
    normed: bool = True # whether to normalize the value function
    guidance_matching_type: str = 'direct' # 'direct', 'use_learned_v', 'rw_use_learned_z' 'rw', 'grad_z

    # z model
    z_transformer_config: TransformerConfig = field(default=TransformerConfig(depth=4, num_heads=4, hidden_size=64))
    z_exp_name: str = ''
    z_cp: str = '2'

    # flow model
    flow_transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    flow_exp_name: str = 'H20_1e6steps'
    flow_cp: str = '19'
    flow_matching_type: str = 'cfm' # Must specify: this is necessary for MC guidance! 'cfm', 'ot_cfm', 'vp_cfm', 'sb_cfm'

    # training
    n_train_steps: int = 100001
    save_freq: int = 10000
    batch_size: int = 64
    learning_rate: float = 1e-4

    eps: float = 0

    def __post_init__(self):
        if self.z_exp_name == '':
            self.z_exp_name = self.exp_name

