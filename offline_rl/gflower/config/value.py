from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

@dataclass
class ValueTrainingConfig:
    seed: int = 0
    device: str = 'cuda:0'

    log_folder: str = 'logs'
    exp_name: str = 'value'

    # environment
    env: str = 'hopper-medium-replay'
    horizon: int = 20
    inf_horizon: bool = False # whether calculate value for infinite horizon
    normalizer: str = 'GaussianNormalizer'
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 100000
    max_n_episodes: int = 100000
    termination_penalty: float = 0
    termination_penalty: float = -100

    state_dim: int = 4
    action_dim: int = 4

    # model
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    discount: float = 0.99
    normed: bool = False

    # training
    n_train_steps: int = 10001
    save_freq: int = 5000
    batch_size: int = 64
    learning_rate: float = 2e-4
