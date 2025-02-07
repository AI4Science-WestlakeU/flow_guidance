from itertools import cycle
import os
from gflower.models_flow.flow_matcher import GuidanceMatcher
from gflower.models_flow.transformer import TransformerFlow
import torch
import tqdm
import tyro
from gflower.config.guide import GuideTrainingConfig
from gflower.datasets.sequence import ValueDataset
from gflower.utils.arrays import batch_to_device
from run.utils import deterministic, set_cuda_visible_device, save_config
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader


def get_model_g_z_v(cfg: GuideTrainingConfig):
    if cfg.train_z:
        model_g = None
    else:
        model_g = TransformerFlow(
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

    if cfg.guidance_matching_type in ['direct', 'use_learned_v', 'rw_use_learned_z'] or cfg.train_z:
        model_z = TransformerFlow(
            seq_len=cfg.horizon,
            in_channels=cfg.action_dim + cfg.state_dim,
            out_channels=1,
            hidden_size=cfg.z_transformer_config.hidden_size,
            depth=cfg.z_transformer_config.depth,
            num_heads=cfg.z_transformer_config.num_heads,
            mlp_ratio=cfg.z_transformer_config.mlp_ratio,
            x_emb_proj=cfg.z_transformer_config.x_emb_proj,
            x_emb_proj_conv_k=cfg.z_transformer_config.x_emb_proj_conv_k,
        ).to(cfg.device)
        if not cfg.train_z: # training model_g, need to load pretrained model_z
            model_z.load_state_dict(
                torch.load(os.path.join(cfg.log_folder, cfg.env, "guidance", cfg.z_exp_name, f'model_z_{cfg.z_cp}.pth'))
            )
    elif cfg.guidance_matching_type == 'rw':
        model_z = None # training model_g with rw loss, no need to use pretrained model_z
    else:
        raise ValueError(f"Guidance matching type {cfg.guidance_matching_type} + train_z={cfg.train_z} not supported")

    if cfg.guidance_matching_type in ['use_learned_v', 'rw_use_learned_z', 'rw']:
        model_v = TransformerFlow(
            seq_len=cfg.horizon,
            in_channels=cfg.action_dim + cfg.state_dim,
            out_channels=cfg.action_dim + cfg.state_dim,
            hidden_size=cfg.flow_transformer_config.hidden_size,
            depth=cfg.flow_transformer_config.depth,
            num_heads=cfg.flow_transformer_config.num_heads,
            mlp_ratio=cfg.flow_transformer_config.mlp_ratio,
            x_emb_proj=cfg.flow_transformer_config.x_emb_proj,
            x_emb_proj_conv_k=cfg.flow_transformer_config.x_emb_proj_conv_k,
        ).to(cfg.device)
        model_v.load_state_dict(
            torch.load(os.path.join(cfg.log_folder, cfg.env, "flow", cfg.flow_exp_name, f'model_{cfg.flow_cp}.pth'))
        )
    elif cfg.guidance_matching_type == 'direct' or cfg.train_z:
        model_v = None
    else:
        raise ValueError(f"Guidance matching type {cfg.guidance_matching_type} + train_z={cfg.train_z} not supported")

    return model_g, model_z, model_v


def train(cfg: GuideTrainingConfig, log_subfolder: str):

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

    model_g, model_z, model_v = get_model_g_z_v(cfg)

    guidance_matcher = GuidanceMatcher(
        action_dim=cfg.action_dim,
        model=model_g,
        model_z=model_z, 
        model_v=model_v,
        flow_matching_type=cfg.flow_matching_type,
        train_z=cfg.train_z,
        scale=cfg.scale,
        guidance_matching_type=cfg.guidance_matching_type,
    )

    train_loader = cycle(DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True
    ))

    optimizer = torch.optim.Adam(
        model_z.parameters() if cfg.train_z else model_g.parameters(), lr=cfg.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_train_steps)
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_subfolder, 'tensorboard_logs'))
    pbar = tqdm.tqdm(range(cfg.n_train_steps))
    for i in pbar:
        batch = batch_to_device(next(train_loader), cfg.device)
        loss, infos = guidance_matcher.loss(*batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        pbar.set_description(f"Loss: {loss.item():.4f}")
        if i % cfg.save_freq == 0:
            # Save both regular model and EMA model
            torch.save(
                model_z.state_dict() if cfg.train_z else model_g.state_dict(), 
                os.path.join(log_subfolder, f'model_{"z" if cfg.train_z else cfg.guidance_matching_type}_{i // cfg.save_freq}.pth')
            )
        
        writer.add_scalar('loss', loss, i)


if __name__ == "__main__":
    cfg = tyro.cli(GuideTrainingConfig)
    
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed) # seed everything

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, 'guidance', cfg.exp_name)
    
    if os.path.exists(os.path.join(log_subfolder, f'model_{"z" if cfg.train_z else cfg.guidance_matching_type}_1.pth')) or cfg.guidance_matching_type == 'grad_z':
        print(f"[ Skip ] Model {cfg.exp_name} has been trained, skipping training")
        exit(0)

    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)



