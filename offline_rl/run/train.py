from itertools import cycle
import os
import torch
import tyro

from gflower.config.flow_matching import FlowMatchingTrainingConfig
from gflower.datasets.sequence import SequenceDataset
from gflower.models_flow.flow_matcher import FlowMatcher
from gflower.models_flow.transformer import TransformerFlow
from gflower.utils.arrays import batch_to_device
from run.utils import deterministic, set_cuda_visible_device, save_config
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import tqdm


def train(cfg: FlowMatchingTrainingConfig, log_subfolder: str):
    dataset = SequenceDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        max_n_episodes=cfg.max_n_episodes,
        termination_penalty=0,
        seed=cfg.seed,
    )

    transformer = TransformerFlow(
        seq_len=cfg.horizon,
        in_channels=cfg.action_dim + cfg.state_dim,
        out_channels=cfg.action_dim + cfg.state_dim,
        hidden_size=cfg.transformer_config.hidden_size,
        depth=cfg.transformer_config.depth,
        num_heads=cfg.transformer_config.num_heads,
        mlp_ratio=cfg.transformer_config.mlp_ratio,
        x_emb_proj=cfg.transformer_config.x_emb_proj,
        x_emb_proj_conv_k=cfg.transformer_config.x_emb_proj_conv_k,
    ).to(cfg.device)

    flow_matcher = FlowMatcher(
        action_dim=cfg.action_dim,
        model=transformer,
        flow_matching_type=cfg.flow_matching_type,
    )

    train_loader = cycle(DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True
    ))

    optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.save_freq * 2)
    ema = torch.optim.swa_utils.AveragedModel(
        transformer, 
        avg_fn=lambda avg, new, num: cfg.ema_decay * avg + (1 - cfg.ema_decay) * new
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_subfolder, 'tensorboard_logs'))

    for i in tqdm.tqdm(range(cfg.n_train_steps)):
        batch = batch_to_device(next(train_loader), cfg.device) # (B, T, C)
        loss, infos = flow_matcher.loss(*batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update EMA
        ema.update_parameters(transformer)

        if i % cfg.save_freq == 0:
            # Save both regular model and EMA model
            torch.save(
                transformer.state_dict(), 
                os.path.join(log_subfolder, f'model_{i // cfg.save_freq}.pth')
            )
            torch.save(
                ema.module.state_dict(), 
                os.path.join(log_subfolder, f'model_ema_{i // cfg.save_freq}.pth')
            )
        writer.add_scalar('loss', loss, i)


if __name__ == "__main__":
    cfg = tyro.cli(FlowMatchingTrainingConfig)
    
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed) # seed everything

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, 'flow', cfg.exp_name)
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)



