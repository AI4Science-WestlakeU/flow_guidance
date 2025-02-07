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

def train_z(
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
    model = MLP(dim=2, out_dim=1, w=width, time_varying=True, exp_final=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)

    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            loss = torch.nn.functional.mse_loss(
                model(torch.cat([xt, t[:, None]], dim=-1)).squeeze(-1),
                torch.exp(-scale * J(x1))
            )

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

    return model


def train_g(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_z,
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
    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.nn.functional.mse_loss(
                g, # (B, 2)
                ((torch.exp(-scale * J(x1)).unsqueeze(-1) / (z.abs() + 1e-8)) - 1) * ut  # (B, 2)
            )

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

    return model


def train_g2(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_v,
        model_z,
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
    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            v = model_v(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.nn.functional.mse_loss(
                g, # (B, 2)
                (torch.exp(-scale * J(x1)).unsqueeze(-1) / (z.abs() + 1e-8)) * ut  # (B, 2)
                - v
            )

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

    return model


def train_g3(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_v,
        model_z,
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
    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            v = model_v(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = (
                torch.square(
                    g + v - ut  # (B, 2)
                ) * (torch.exp(-scale * J(x1).unsqueeze(-1)) / (z.abs() + 1e-8))
            ).mean()

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

    return model


def train_rwft(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_v,
        model_z,
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
    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            v = model_v(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = (
                torch.square(
                    g + v - ut  # (B, 2)
                ) * (torch.exp(-scale * J(x1).unsqueeze(-1)))
            ).mean()

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

    return model



if __name__ == "__main__":
    cfg = tyro.cli(GuidanceTrainingConfig)

    deterministic(cfg.seed)
    set_cuda_visible_device(cfg)

    # save the config
    log_subfolder = os.path.join(
        "logs", f"{cfg.x0_dist}-{cfg.x1_dist}", f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}"
    )
    save_config(cfg, log_subfolder, config_name=f'gm_scale_{cfg.scale}_config.yaml')

    # train z
    if 'z' in cfg.which_model.split('-'):
        model_z = train_z(
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

        model_path = os.path.join(log_subfolder, f"guidance_matching_z_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")
        torch.save(model_z.state_dict(), model_path)

    # load model_z in case not trained from scratch
    model_z = MLP(dim=2, out_dim=1, w=cfg.width, time_varying=True, exp_final=True).to(cfg.device)
    try:
        model_z.load_state_dict(torch.load(os.path.join(log_subfolder, f"guidance_matching_z_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")))
    except:
        raise FileNotFoundError("No model_z found!")

    # train g
    if 'g' in cfg.which_model.split('-'):
        model_g = train_g(
            get_distribution(cfg.x0_dist).sample, 
            get_distribution(cfg.x1_dist).sample, 
            model_z, 
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

        model_path = os.path.join(log_subfolder, f"guidance_matching_g_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")
        torch.save(model_g.state_dict(), model_path)


    # load model_v
    if 'g2' in cfg.which_model.split('-') or 'g3' in cfg.which_model.split('-') or 'rwft' in cfg.which_model.split('-'):
        model_v = MLP(dim=2, out_dim=2, w=cfg.width, time_varying=True).to(cfg.device)
        try:
            model_v.load_state_dict(torch.load(os.path.join(log_subfolder, f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}.pth")))
        except:
            raise FileNotFoundError("No model_v found, training from scratch")  

    # train g2
    if 'g2' in cfg.which_model.split('-'):
        model_g2 = train_g2(
            get_distribution(cfg.x0_dist).sample, 
            get_distribution(cfg.x1_dist).sample, 
            model_v, 
            model_z, 
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

        model_path = os.path.join(log_subfolder, f"guidance_matching_g2_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")
        torch.save(model_g2.state_dict(), model_path)

    # train g3
    if 'g3' in cfg.which_model.split('-'):
        model_g3 = train_g3(
            get_distribution(cfg.x0_dist).sample, 
            get_distribution(cfg.x1_dist).sample, 
            model_v, 
            model_z, 
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

        model_path = os.path.join(log_subfolder, f"guidance_matching_g3_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")
        torch.save(model_g3.state_dict(), model_path)

    # train rwft
    if 'rwft' in cfg.which_model.split('-'):
        model_rwft = train_rwft(
            get_distribution(cfg.x0_dist).sample, 
            get_distribution(cfg.x1_dist).sample, 
            model_v, 
            model_z, 
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

        model_path = os.path.join(log_subfolder, f"guidance_matching_rwft_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")
        torch.save(model_rwft.state_dict(), model_path)
