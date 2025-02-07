import os
import torch
from guided_flow.config.guidance_training import GuidanceTrainingConfig
from guided_flow.distributions.base import get_distribution
import tyro
from matplotlib import pyplot as plt

from guided_flow.backbone.mlp import MLP, EnergyMLP
from guided_flow.flow.conditional_flow_matching import get_cfm
from guided_flow.utils.misc import deterministic, save_config, set_cuda_visible_device
import tqdm

def train_ceg(
        x0_sampler,
        x1_sampler,
        # guidance related
        J,
        scale=1.0, 
        # training related
        cfm = 'cfm', 
        width = 64,
        sigma = 0.0,
        device='cuda',
        batch_size=1024,
        lr=1e-4,
        num_steps=20000,
):
    # Definition
    model = MLP(dim=2, out_dim=1, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)

    for k in tqdm.tqdm(range(num_steps)):
        optimizer.zero_grad()

        x0 = x0_sampler(batch_size=batch_size, device=device)
        x1 = x1_sampler(batch_size=batch_size, device=device)

        t = torch.rand(x0.shape[0]).to(device)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

        loss = -torch.sum(
            torch.softmax(-scale * J(x1), dim=0) # (B, )
            * torch.softmax(
                model(torch.cat([xt, t[:, None]], dim=-1)).squeeze(-1), # (B, 1) -> (B, )
                dim=0
            ).log() # (B, )
        ).mean()

        loss.backward()
        optimizer.step()

    return model



if __name__ == "__main__":
    cfg = tyro.cli(GuidanceTrainingConfig)

    deterministic(cfg.seed)
    set_cuda_visible_device(cfg)
    # save the config
    log_subfolder = os.path.join(
        "logs", f"{cfg.x0_dist}-{cfg.x1_dist}", f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}"
    )
    save_config(cfg, log_subfolder, config_name=f'ceg_scale_{cfg.scale}_config.yaml')

    model = train_ceg(
        get_distribution(cfg.x0_dist).sample, 
        get_distribution(cfg.x1_dist).sample, 
        get_distribution(cfg.x1_dist).get_J, 
        cfg.scale, 
        cfg.cfm, 
        cfg.width, 
        cfg.sigma, 
        cfg.device, 
        cfg.batch_size, 
        cfg.lr, 
        cfg.num_steps
    )

    model_path = os.path.join(log_subfolder, f"ceg_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")
    torch.save(model.state_dict(), model_path)
