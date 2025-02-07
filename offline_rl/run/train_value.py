from itertools import cycle
import os
import torch
import tqdm
import tyro

from gflower.config.value import ValueTrainingConfig
from gflower.datasets.sequence import ValueDataset
from gflower.models_value.transformer import Transformer
from gflower.utils.arrays import batch_to_device
from run.utils import deterministic, set_cuda_visible_device, save_config
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

def train(cfg: ValueTrainingConfig, log_subfolder: str):

    dataset = ValueDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        max_n_episodes=cfg.max_n_episodes,
        termination_penalty=0,
        seed=cfg.seed,
        discount=cfg.discount,
        normed=cfg.normed,
        inf_horizon=cfg.inf_horizon,
    )

    transformer = Transformer(
        input_dim=cfg.state_dim + cfg.action_dim,
        output_dim=1,
        model_dim=cfg.transformer_config.model_dim,
        num_heads=cfg.transformer_config.num_heads,
        num_layers=cfg.transformer_config.num_layers,
        dropout=cfg.transformer_config.dropout,
    ).to(cfg.device)

    train_loader = cycle(DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True
    ))

    optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg.learning_rate)
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_subfolder, 'tensorboard_logs'))

    for i in tqdm.tqdm(range(cfg.n_train_steps)):
        batch = batch_to_device(next(train_loader), cfg.device)
        value_pred = transformer(batch.trajectories).squeeze(-1)[:, -1] # (B, T, 1) -> (B,)

        loss = torch.nn.MSELoss()(value_pred, batch.values.squeeze(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % cfg.save_freq == 0:
            # Save both regular model and EMA model
            torch.save(
                transformer.state_dict(), 
                os.path.join(log_subfolder, f'model_{i // cfg.save_freq}.pth')
            )
        
        writer.add_scalar('loss', loss, i)


if __name__ == "__main__":
    cfg = tyro.cli(ValueTrainingConfig)
    
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed) # seed everything

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, 'value', cfg.exp_name)
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)



