import os
import torch
from guided_flow.config.training import FlowMatchingTrainingConfig
from guided_flow.distributions.base import get_distribution
import tyro
from matplotlib import pyplot as plt

from guided_flow.backbone.mlp import MLP
from guided_flow.flow.conditional_flow_matching import get_cfm
from guided_flow.utils.misc import deterministic, save_config, set_cuda_visible_device
import tqdm

def train_unconditional(
        x0_sampler,
        x1_sampler,
        cfm = 'cfm', 
        width = 64,
        sigma = 0.0,
        device='cuda',
        batch_size=1024,
        lr=1e-4,
        num_steps=20000,
):
    # Definition
    model = MLP(dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)

    for k in tqdm.tqdm(range(num_steps)):
        optimizer.zero_grad()

        x0 = x0_sampler(batch_size=batch_size, device=device)
        x1 = x1_sampler(batch_size=batch_size, device=device)

        t = torch.rand(x0.shape[0]).to(device)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

    return model


if __name__ == "__main__":
    cfg = tyro.cli(FlowMatchingTrainingConfig)

    # decide whether to skip
    log_subfolder = os.path.join(
        "logs", f"{cfg.x0_dist}-{cfg.x1_dist}", f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}"
    )
    if not cfg.overwrite and os.path.exists(os.path.join(log_subfolder, f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}.pth")):
        print(f"[ Skip ] Model {cfg.cfm} for {log_subfolder} already exists, skipping")
        exit(0)

    # set seed and cuda
    deterministic(cfg.seed)
    set_cuda_visible_device(cfg)

    # save the config
    save_config(cfg, log_subfolder)

    # training
    x0_sampler = get_distribution(cfg.x0_dist).sample
    x1_sampler = get_distribution(cfg.x1_dist).sample

    model = train_unconditional(
        x0_sampler, 
        x1_sampler, 
        cfg.cfm, 
        cfg.width, 
        cfg.sigma, 
        cfg.device, 
        cfg.batch_size, 
        cfg.lr, 
        cfg.num_steps
    )

    model_path = os.path.join(log_subfolder, f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}.pth")
    torch.save(model.state_dict(), model_path)
