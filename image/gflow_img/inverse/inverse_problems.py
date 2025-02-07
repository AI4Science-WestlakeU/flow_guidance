# Standard library imports
from functools import partial
import os
import numpy as np
import torch
import torchdiffeq
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.distributions import Independent, Normal
# Third-party imports
from backbone.unet_configs import instantiate_model
from cfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from config.inverse_problem import InverseProblemConfig
from inverse.degradations import BoxInpainting, GaussianDeblurring, Superresolution
from gflow_img.utils.grad_fn import get_schedule, wrap_grad_fn


def plt_save(x, path, data_scoure='generated', index=0):
    for i in range(x.shape[0]):
        plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')
        plt.savefig(os.path.join(path, data_scoure,
                    f'{i + index}.jpg'), bbox_inches='tight', pad_inches=0)
        plt.close()


class InverseProblem:
    def __init__(self, cfg: InverseProblemConfig):
        self.cfg = cfg

        # prepare model
        self.model_name = cfg.model_data.split('_')[0]
        self.data_name = cfg.model_data.split('_')[1]
        self.model = instantiate_model(self.model_name)
        self.path = cfg.make_path()

        if cfg.flow_type == 'ot':
            self.FM = ExactOptimalTransportConditionalFlowMatcher(cfg.sigma)
        elif cfg.flow_type == 'cfm':
            self.FM = ConditionalFlowMatcher(cfg.sigma)
        else:
            raise NotImplementedError('flow type not implemented')

        # prepare pretrained model
        if self.data_name == 'imagenet64':
            self.load_pretrained_file: str = f'results/{cfg.flow_type}_punet64_imagenet64/model_499.pt'
        elif self.data_name == 'celeba128':
            self.load_pretrained_file: str = f'results/{cfg.flow_type}_punet128_celeba128/model_499.pt'
        elif self.data_name == 'celeba256':
            self.load_pretrained_file: str = f'results/{cfg.flow_type}_punet256_celeba256/model_499.pt'
        else:
            raise ValueError(f"Data {self.data_name} not supported")

        if os.path.exists(self.load_pretrained_file):
            self.model.load_state_dict(torch.load(self.load_pretrained_file))
            self.model = self.model.to(cfg.device)
            self.model.eval()
        else:
            raise ValueError(
                f"Pretrained model not found at {self.load_pretrained_file}")

        # prepare inverse problem
        if cfg.problem == 'inpainting':
            if cfg.noise_type == 'laplace':
                self.sigma_noise = 0.3
            elif cfg.noise_type == 'gaussian':
                self.sigma_noise = 0.05
            if cfg.dim_image == 128:
                half_size_mask = 20
            elif cfg.dim_image == 256:
                half_size_mask = 40
            self.degradation = BoxInpainting(half_size_mask)

        elif cfg.problem == 'deblurring':
            if cfg.dim_image == 128:
                sigma_blur = 1.0
            elif cfg.dim_image == 256:
                sigma_blur = 3.0

            if cfg.noise_type == 'laplace':
                self.sigma_noise = 0.3
            elif cfg.noise_type == 'gaussian':
                self.sigma_noise = 0.05
            kernel_size = 61
            self.degradation = GaussianDeblurring(
                sigma_blur, kernel_size, "fft", cfg.num_channels, cfg.dim_image, cfg.device
            )

        elif cfg.problem == 'superresolution':
            if cfg.dim_image == 128:
                print('Superresolution with scale factor 2')
                self.sf = 2
            elif cfg.dim_image == 256:
                print('Superresolution with scale factor 4')
                self.sf = 4
            if cfg.noise_type == 'laplace':
                self.sigma_noise = 0.3
            elif cfg.noise_type == 'gaussian':
                self.sigma_noise = 0.05
            self.degradation = Superresolution(self.sf, cfg.dim_image)
        else:
            raise ValueError(f"Problem {cfg.problem} not supported")

        self.index = 0
        self.cached_x1 = None
        self.cached_Jx1 = None

    def reset_counter(self):
        self.index = 0

    def get_J(self, y, hat_x):

        if self.cfg.J_method == 'L2':
            return (self.degradation.H(hat_x) - y)**2
        elif self.cfg.J_method == 'L1':
            return torch.abs(self.degradation.H(hat_x) - y)
        elif self.cfg.J_method == 'exp':
            return torch.exp(-torch.abs(self.degradation.H(hat_x) - y))
        else:
            raise ValueError(f"J method {self.cfg.J_method} not supported")

    # 1. Gradient
    @torch.enable_grad()
    def guide_fn(self, t, x_t, dx_dt, y):

        if self.cfg.guide_method == 'nabla_xt_J_xt':
            try:
                with torch.enable_grad():
                    x_t = x_t.requires_grad_(True)
                    J = self.get_J(y, x_t)
                    grad = -torch.autograd.grad(J.sum(), x_t, create_graph=True)[0]
                    return grad
            except Exception as e:
                raise ValueError(
                    f"Failed to compute gradient for {self.cfg.guide_method}: {e}")

        elif self.cfg.guide_method == 'nabla_x1_J_x1':

            x1_pred = x_t + dx_dt * (1 - t)
            try:
                with torch.enable_grad():
                    x1_pred = x1_pred.requires_grad_(True)
                    J = self.get_J(y, x1_pred)
                    grad = - torch.autograd.grad(J.sum(), x1_pred, create_graph=True)[0]
                    return grad
            except Exception as e:
                raise ValueError(
                    f"Failed to compute gradient for {self.cfg.guide_method}: {e}")

        elif self.cfg.guide_method == 'nabla_xt_J_x1':
            with torch.enable_grad():
                x_t = x_t.requires_grad_(True)
                x1_pred = x_t + self.model(t.repeat(x_t.shape[0]), x_t) * (1 - t)

                J = self.get_J(y, x1_pred)
                try:
                    grad = - torch.autograd.grad(J.sum(), x_t, create_graph=True)[0]
                    return grad
                except Exception as e:
                    raise ValueError(
                        f"Failed to compute gradient for {self.cfg.guide_method}: {e}")

        else:
            raise ValueError(
                f"Unknown guide function: {self.cfg.guide_method}")

    def get_grad_fn(self):
        return wrap_grad_fn(self.cfg.guide_scale, self.cfg.schedule, self.guide_fn)

    def wrap_grad_model(self, t, x_t, x1, grad_fn):
        dx_dt = self.model(t, x_t)
        grad = grad_fn(t, x_t, dx_dt, x1)
        if self.cfg.problem == 'superresolution':
            grad = grad * (self.sf**2)
        return dx_dt + grad

    @torch.no_grad()
    def solve_ip_grad(self, data, return_all=False):
        grad_fn = self.get_grad_fn()
        index = 0
        
        x1 = data.to(self.cfg.device)
        d_x1 = self.degradation.H(x1)
        d_x1 = d_x1 + torch.randn_like(d_x1) * self.sigma_noise
        traj = torchdiffeq.odeint(
            lambda t, x: self.wrap_grad_model(t, x, d_x1, grad_fn),
            torch.randn_like(x1, device=self.cfg.device),
            torch.linspace(0, 1, self.cfg.steps, device=self.cfg.device),
            atol=1e-4,
            rtol=1e-4,
            method=self.cfg.ode_method
        )

        plt_save(traj[-1], self.path, 'generated', index)
        plt_save(x1, self.path, 'groundtruth', index)
        plt_save(d_x1, self.path, 'd_groundtruth', index)
        index += x1.shape[0]

    # 2. PiGDM
    def solve_ip_pgdm(self, data):
        H = self.degradation.H
        H_adj = self.degradation.H_adj
        steps, delta = self.cfg.steps, 1 / self.cfg.steps
        scale = self.cfg.guide_scale
    
        noisy_img = H(data.clone().to(self.cfg.device))
        noisy_img += torch.randn_like(noisy_img) * self.sigma_noise
        x = H_adj(noisy_img.clone())
        start_time = self.cfg.start_time
        x = start_time * x.clone() + (1 - start_time) * torch.randn_like(x)

        pbar_ode = tqdm(range(int(steps * start_time), int(steps)))
        for iteration in pbar_ode:

            t = delta * iteration
            t_t = torch.tensor(t, device=self.cfg.device)
            t_b = torch.ones(len(x), device=self.cfg.device) * t  # (B, )
            vt = self.model(t_b, x)
            
            if self.cfg.guide_method in ['PiGDM', 'PiGDM+']: # If PiGDM, use shedule name pigdm
                rt_squared = get_schedule(self.cfg.rt_schedule, t_t).view(-1, 1, 1, 1) * self.cfg.rt_scale
            else:
                raise ValueError(f"Unknown guide function: {self.cfg.guide_method}")
            
            x1_hat = x + (1 - t_b.view(-1, 1, 1, 1)) * vt

            # sovle linear problem Cx=d
            d = noisy_img - H(x1_hat)

            sol = torch.zeros_like(d) # To compute. sol should be: (y-Hx)^T(I + H^TH)^{-1}

            if self.cfg.problem == "inpainting":
                for i in range(d.shape[0]):
                    h_h_adj = H(torch.ones_like(x)) # (B, C, H, W)? Why is this H^TH though?
                    sol_tmp = 1 / (
                        h_h_adj[i] * rt_squared + self.sigma_noise ** 2
                    ) * d[i]
                    sol[i] = sol_tmp.reshape(d[i].shape)

            elif self.cfg.problem == "superresolution":
                # rt_squared = (1 - t) ** 2 / ((1 - t) ** 2 + delta * iteration ** 2) # Old shedule. This is wrong!

                diag = self.degradation.downsampling_matrix
                h_h_adj = torch.matmul(diag, diag.T)
                sol_tmp = 1 / (
                    torch.diag(h_h_adj) * rt_squared + self.sigma_noise ** 2
                )

                sol_tmp = sol_tmp[None, None, :] * d.flatten(start_dim=2)
                sol = sol_tmp.reshape(d.shape)

            elif self.cfg.problem == "deblurring":
                fft_d = torch.fft.fft2(d)
                kernel = self.degradation.filter
                kernel_size = kernel.shape[2]
                kernel_id = torch.zeros_like(kernel)
                kernel_id[:, :, kernel_size // 2, kernel_size // 2] = 1
                fft_kernel = torch.fft.fft2(kernel)
                inv = rt_squared * fft_kernel * torch.conj(fft_kernel) + self.sigma_noise ** 2
                sol = torch.fft.ifft2(fft_d / inv)
            else:
                raise ValueError(
                    f"Problem {self.cfg.problem} not supported")

            
            # do vector jacobian product
            t_pad = t_b.view(-1, 1, 1, 1)
            
            # gamma seems like a manual adjustment to the time schedule
            gamma = get_schedule(self.cfg.schedule, t_pad)
            

            vec = H_adj(sol) # (y-Hx)^T(I + H^TH)^{-1} H
            
            if self.cfg.guide_method == 'PiGDM':
                with torch.enable_grad():
                    g = torch.autograd.functional.vjp(
                        lambda z: self.model(t_b, z), inputs=x, v=vec
                    )[1] # (y-Hx)^T(I + H^TH)^{-1} H (d_{x_t} v_t)
                
                g = vec + (1 - t_pad) * g # Since d_{x_t} \hat{x}_1 = 1 + (1 - t) d_{x_t} v_t
            elif self.cfg.guide_method == 'PiGDM+':
                g = vec
            else:
                raise ValueError(f"Unknown guide function: {self.cfg.guide_method}")
            
            if self.cfg.schedule_ratio:
                ratio = (1 - t_pad) / (t_pad + self.cfg.time_ratio_eps) # Time schedule
            else:
                ratio = 1.0
            v_adapted = vt + ratio * gamma * g * scale
            x_new = x + delta * v_adapted

            x = x_new
            if self.cfg.clamp_x:
                x = torch.clamp(x, -1, 1)

        plt_save(x, self.path, 'x_gen', self.index)
        plt_save(data, self.path, 'x_gt', self.index) # groundtruth x
        plt_save(noisy_img, self.path, 'y', self.index) # noisy measurement
        self.index += data.shape[0]

    # 3. MC
    def cfm_log_p_t1(self, x1, xt, t, epsilon):
        # xt = t x1 + (1 - t) x0 -> x0 = xt / (1 - t) - t / (1 - t) x1
        x1 = x1.flatten(1) # (B, C * H * W)
        xt = xt.flatten(1) # (B, C * H * W)
        mu_t = t * x1 # (B, C * H * W)
        sigma_t = (1 - t + epsilon)
        log_p1t = Independent(
            Normal(mu_t, torch.ones_like(mu_t) * sigma_t), 1
        ).log_prob(xt) # (B, C * H * W). Changed implementation for efficiency. (Multivairiate requires a large cov matrix)
        return log_p1t
    
    def mc_guide_fn(self, t, x_t, x1=None, Jx1=None):
        """
        Args:
            t: float
            x_t: tensor (B, C, H, W)
            x1: tensor (B, C, H, W)
            Jx1: tensor (B_MC, B)
        """
        # estimate E (e^{-J} / Z - 1) * u
        MC_EP = self.cfg.mc_ep
        MC_B = self.cfg.mc_batch_size
        assert MC_B == x1.shape[0], "MC_B must be the same as the number of samples in x1"
        SCALE = self.cfg.mc_scale
        b = x_t.shape[0]
        x_ = x_t.repeat(MC_B, 1, 1, 1) # (MC_B * b, C, H, W)
        x1_ = x1.unsqueeze(0).repeat(b, 1, 1, 1, 1).permute(1, 0, 2, 3, 4).reshape(-1, *x1.shape[1:]) # (MC_B * b, C, H, W)
        
        log_p_t1_x = self.cfm_log_p_t1(x1_, x_, t, epsilon=MC_EP) # (MC_B * b)
        
        # self normalize
        assert Jx1.dim() == 2 and Jx1.shape[-1] == b
        Jx1 = Jx1.reshape(-1)
        if self.cfg.mc_self_normalize:
            Jx1 = (Jx1 - Jx1.mean()) / (Jx1.std() + 1e-8)
        J_exp = torch.exp(-SCALE * Jx1) # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
        
        log_p_t1_x = log_p_t1_x.reshape(MC_B, b, 1, 1, 1) 
        log_p_t_x = log_p_t1_x.logsumexp(0) - torch.log(torch.tensor(MC_B)) # (MC_B, B, 1, 1, 1) -> (B, 1, 1, 1)
        log_p_t1_x_times_J_exp = log_p_t1_x + torch.log(J_exp).reshape(MC_B, b, 1, 1, 1) # (MC_B, b, 1, 1, 1)
        log_Z = log_p_t1_x_times_J_exp.logsumexp(0) - torch.log(torch.tensor(MC_B)) - log_p_t_x # (B, 1, 1, 1)
        Z = torch.exp(log_Z) # (B, 1, 1, 1)

        u = (x1_ - x_) / (1 - t + MC_EP) # (MC_B * b, C, H, W)
        g = torch.exp(log_p_t1_x - log_p_t_x) * (J_exp.reshape(MC_B, b, 1, 1, 1) / (Z + 1e-8) - 1) * u.reshape(MC_B, b, *x_.shape[1:]) # (MC_B, b, C, H, W)
        return g.mean(0) # (MC_B, B, C, H, W) -> (B, C, H, W)

    # cached x1 and Jx1 support set
    def get_cached_for_mc(self, data, data_x1):
        """
        Args:
            data: tensor (B, C, H, W)
            data_x1: tensor (MC_B, C, H, W)
        """
        if self.cached_x1 is None:
            self.cached_x1 = data_x1.to(self.cfg.device).clone().detach()
        if self.cached_Jx1 is None:
            y = self.degradation.H(data.to(self.cfg.device).clone())
            y = y + torch.randn_like(y) * self.sigma_noise # (MC_B, C, H', W')
            # import pdb; pdb.set_trace()
            self.cached_Jx1 = self.get_J(
                y.unsqueeze(1).repeat(1, self.cached_x1.shape[0], 1, 1, 1).reshape(-1, *y.shape[1:]), # (MC_B, B, C, H, W) -> (B * B_MC, C, H, W)
                self.cached_x1.unsqueeze(0).repeat(y.shape[0], 1, 1, 1, 1).reshape(-1, *self.cached_x1.shape[1:]) # (MC_B, B, C, H, W) -> (B * B_MC, C, H, W)
            ).mean((1, 2, 3)).reshape(self.cfg.mc_batch_size, y.shape[0])
    
    def remove_cached_x1_for_mc(self):
        self.cached_x1 = None
    
    def remove_cached_j_for_mc(self):
        self.cached_Jx1 = None

    def solve_ip_mc(self, data, data_x1):
        self.get_cached_for_mc(data, data_x1)
        guide_fn = partial(self.mc_guide_fn, x1=self.cached_x1, Jx1=self.cached_Jx1)
        
        x1 = data.to(self.cfg.device)
        y = self.degradation.H(x1)
        y = y + torch.randn_like(y) * self.sigma_noise
        traj = torchdiffeq.odeint(
            lambda t, x: guide_fn(t, x) + self.model(t, x),
            torch.randn_like(x1, device=self.cfg.device),
            torch.linspace(self.cfg.start_time, 1, self.cfg.steps, device=self.cfg.device),
            atol=1e-4,
            rtol=1e-4,
            method=self.cfg.ode_method
        )
        x = traj[-1]

        plt_save(x, self.path, 'x_gen', self.index)
        plt_save(data, self.path, 'x_gt', self.index) # groundtruth x
        plt_save(y, self.path, 'y', self.index) # noisy measurement
        self.index += data.shape[0]
