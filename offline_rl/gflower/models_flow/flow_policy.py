from collections import namedtuple
import math
import os
from torch import nn
import torch
from gflower.config.flow_matching import FlowMatchingEvaluationConfig
from gflower import utils
from gflower.models_flow.flow_matcher import apply_conditioning, apply_conditioning_from_conditioned_x
from gflower.models_flow.optimal_transport import OTPlanSampler
from gflower.sampling.guides import ValueGuide
from gflower.utils.arrays import to_device, to_torch

Trajectories = namedtuple('Trajectories', 'actions observations values')

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class ConditionedODESolver(nn.Module):
    def __init__(self, model, conditions, action_dim, guide_fn=None, ode_method='euler'):
        super().__init__()
        self.model = model
        self.conditions = conditions
        self.guide_fn = guide_fn
        self.action_dim = action_dim
        assert ode_method in ['euler'], "Only Euler is supported for now"

    def forward(self, x, t_span, *args, **kwargs):
        """
        Args:
            x (B, C, T), t (T)
        """
        assert len(t_span) > 1, "t_span must have at least 2 elements"
        x0 = x.clone()
        dt = t_span[1] - t_span[0]
        for t in t_span:
            if self.guide_fn is None:
                # model forward pass
                dx_dt = self.model(x, t)
            # add gradient guidance
            else:
                x = x.requires_grad_()
                dx_dt = self.model(x, t)
                dx_dt = dx_dt + self.guide_fn(x, t, dx_dt, self.model)
            # fill in the condition
            dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), self.conditions, self.action_dim
            )
            x = x + dx_dt * dt
            x = x.detach()
        return x


class FlowPolicy(nn.Module):
    """
    This class is a wrapper around a flow model that generates actions from ONE step of 
    the observed state. 

    The generation is guided with the value model using different guidance methods.

    Normalization:
        Input observation and output action are denormalized; Models' input and output 
        are normalized.
    """
    def __init__(
        self, 
        flow_model, value_model, normalizer, action_dim, state_dim, horizon, 
        cfg: FlowMatchingEvaluationConfig,
        guide_model=None,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon

        self.cfg = cfg
        self.value_model = value_model # we need this to return value
        self.guide_model = guide_model

    def __call__(self, conditions, batch_size=1, verbose=True):
        # assert batch_size == 1, "batch_size must be 1 for now"

        if self.cfg.guidance_method in ['ss']:
            return self.ss_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['gradient']:
            return self.gradient_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['mc']:
            return self.mc_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['no']:
            pass
        elif self.cfg.guidance_method in ['guidance_matching']:
            return self.learned_guidance_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['sim_mc']:
            return self.sim_mc_guidance_forward(conditions, batch_size)
        else:
            raise ValueError(f"Unsupported guidance method: {self.cfg.guidance_method}")

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=None, 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(
            x, 
            t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            ),
        ) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        if self.cfg.guidance_method != 'no':
            values = self.value_model(normed_observations, normed_actions)
        else:
            values = None

        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories


    ### Sample and Selection MPC ###

    def ss_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for now"
        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=None, 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )


        x = torch.randn(self.cfg.ss_batch * batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        conditions = to_torch(conditions, device=x.device) # {'0': tensor (B, C)}
        conditions = utils.apply_dict(lambda x: x.repeat(self.cfg.ss_batch, 1), conditions)
        x = apply_conditioning(x, conditions, self.action_dim)

        x = solver(
            x, 
            t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            ),
        ) # (B_ss * B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = torch.tensor(self.normalizer.unnormalize(normed_actions, 'actions'), device=self.cfg.device)

        normed_observations = x[:, :, self.action_dim:]
        observations = torch.tensor(self.normalizer.unnormalize(normed_observations, 'observations'), device=self.cfg.device)

        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1)) # (B_ss * B, T, 1)
        values = values[:, -1, 0] # (B_ss * B)
        values = values.reshape(self.cfg.ss_batch, batch_size) # (B_ss, B)
        best_idx = values.argmax(dim=0).to(self.cfg.device) # (B,)
        
        # to construct trajectories
        best_values = values[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B,)
        best_values = best_values[0] # (1,), select the first sample in the batch
        best_observations = observations.reshape(self.cfg.ss_batch, batch_size, self.horizon, self.state_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        best_observations = best_observations[0] # (T, C), select the first sample in the batch
        best_actions = actions.reshape(self.cfg.ss_batch, batch_size, self.horizon, self.action_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        best_actions = best_actions[0] # (T, C), select the first sample in the batch
        
        trajectories = Trajectories(to_np(best_actions), to_np(best_observations), to_np(best_values))

        # output actions
        actions = actions.reshape(self.cfg.ss_batch, batch_size, self.horizon, self.action_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        actions = to_np(actions[0, 0]) # (C,), simply get the first action in the first sample in the batch

        return actions, trajectories


    ### Taylor Expansion Approximate Gradient Guidance ###

    def get_gradient_guidance_model(self, value_model, schedule_fn, scale, grad_at='x_1', grad_to='x_1'):
        """
        Return the guidance model for the flow model.
        """
        assert self.cfg.guidance_method in ['gradient'], f"Unsupported guidance method: {self.cfg.guidance_method}"

        def guide_fn(x, t, dx_dt, flow_model):
            if grad_at == 'x_t':
                value = value_model(x)[:, -1, 0] # (B, T, 1) -> (B,)
            elif grad_at == 'x_1':
                x1_pred = x + (1 - t) * dx_dt
                value = value_model(x1_pred)[:, -1, 0] # (B, T, 1) -> (B,)
            else:
                raise ValueError(f"Unsupported gradient compute at: {grad_at}")
            if grad_to == 'x_t':
                grad = torch.autograd.grad([value.sum()], [x])[0]
            elif grad_to == 'x_1':
                assert grad_at == 'x_1', "cannot compute gradient wrt x_1 when grad_at is x_t"
                grad = torch.autograd.grad([value.sum()], [x1_pred])[0]
            else:
                raise ValueError(f"Unsupported gradient compute at: {grad_to}")
            return grad * scale * schedule_fn(t)
        return guide_fn

    def get_scheduler(self, schedule_fn):
        """
        Return the scheduler for the gradient guidance.
        """
        if schedule_fn == 'const':
            return lambda x: x
        elif schedule_fn == 'linear_decay':
            return lambda x: 1 - x
        elif schedule_fn == 'cosine_decay':
            return lambda x: 0.5 * (1 + torch.cos(x * math.pi))
        elif schedule_fn == 'exp_decay':
            return lambda x: (torch.exp(-x) - math.exp(-1)) / (1 - math.exp(-1))
        else:
            raise ValueError(f"Unsupported gradient schedule: {schedule_fn}")

    def gradient_forward(self, conditions, batch_size=1):
        """
        Use gradient guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"
        assert self.cfg.guidance_method == 'gradient', f"guidance_method must be gradient, but got {self.cfg.guidance_method}"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_gradient_guidance_model(
                self.value_model, 
                schedule_fn=self.get_scheduler(self.cfg.grad_schedule), 
                scale=self.cfg.grad_scale, 
                grad_at=self.cfg.grad_compute_at, 
                grad_to=self.cfg.grad_wrt
            ), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories
    

    ### Monte-Carlo Approximate Guidance ###

    def _get_cached_ot_cfm_plan(self):
        if self.cached_ot_cfm_plan is None:
            raise ValueError("No cached OT-CFM plan found")
        return self.cached_ot_cfm_plan

    def _save_cached_ot_cfm_plan(self, x0_, x1_):
        self.cached_ot_cfm_plan = (x0_, x1_)

    def _remove_cached_ot_cfm_plan(self):
        self.cached_ot_cfm_plan = None

    def gaussian_prob(self, x, mean=0, std=1):
        """
        x: (B, T, C)
        """
        return torch.exp(-(x - mean).square().sum((1, 2)) / 2 / std.pow(x.shape[1] * x.shape[2])) / (2 * math.pi * std.pow(2)).pow(x.shape[1] * x.shape[2] / 2)

    def get_mc_guide_fn(self, x1, cached_v=None):
        """
        Compute the gradient guidance for the flow model.
        I think we only need to implement CFM and OT-CFM with Gaussian paths

        Args:
            x1_: Tensor, shape (B, T, C)
        """

        def cfm_log_p_t1(x1, xt, t, epsilon):
            # xt = t x1 + (1 - t) x0 -> x0 = xt / (1 - t) - t / (1 - t) x1
            x1 = x1.flatten(1) # (B, T * C)
            xt = xt.flatten(1) # (B, T * C)
            mu_t = t * x1 # (B, T * C)
            sigma_t = (1 - t + epsilon)
            log_p1t = torch.distributions.MultivariateNormal(
                mu_t, torch.eye(mu_t.shape[1], device=mu_t.device) * sigma_t
            ).log_prob(xt) # (B, T * C)
            return log_p1t
        
        def ot_cfm_log_p_tz(x0, x1, xt, t, std):
            """ 
            Args:
                std: float, g.t.: 0. Too small: requires large mc_batch_size; Too large: inaccurate
            """
            # xt = t x1 + (1 - t) x0 -> x0 = xt / (1 - t) - t / (1 - t) x1
            x0 = x0.flatten(1) # (B, T * C)
            x1 = x1.flatten(1) # (B, T * C)
            xt = xt.flatten(1) # (B, T * C)
            mean = t * x1 + (1 - t) * x0 # (B, T * C)
            log_p1t = torch.distributions.MultivariateNormal(
                mean, torch.eye(mean.shape[1], device=mean.device) * std
            ).log_prob(xt)
            return log_p1t

        def guide_fn(x, t, dx_dt, model):
            """
            Args:
                t: float
                x: Tensor, shape (b, T, C)
                dx_dt: Tensor, shape (b, T, C)
            """
            # estimate E (e^{-J} / Z - 1) * u
            MC_EP = self.cfg.mc_ep
            MC_B = self.cfg.mc_batch_size
            assert MC_B == x1.shape[0], "MC_B must be the same as the number of samples in x1"
            SCALE = self.cfg.mc_scale
            OT_STD = self.cfg.mc_ot_std
            b = x.shape[0]
            x_ = x.repeat(MC_B, 1, 1) # (MC_B * b, T, C)
            x1_ = x1.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x1.shape[1:]) # (MC_B * b, T, C)
            
            if self.cfg.flow_matching_type == 'cfm':
                log_p_t1_x = cfm_log_p_t1(x1_, x_, t, epsilon=MC_EP) # (MC_B * b)
                
                if cached_v is None:
                    v_ = self.value_model(x1_)[:, -1, 0]
                else:
                    v_ = cached_v.clone()
                
                if self.cfg.mc_linear_J:
                    J_ = SCALE * v_ # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                    if self.cfg.mc_self_normalize:
                        J_ = ((J_ - J_.mean()) / (J_.std() + 1e-8)).clamp(0)
                else:
                    # self normalize
                    if self.cfg.mc_self_normalize:
                        v_ = (v_ - v_.mean()) / (v_.std() + 1e-8)
                    J_ = torch.exp(SCALE * v_) # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                
                log_p_t1_x = log_p_t1_x.reshape(MC_B, b, 1, 1)
                log_p_t_x = log_p_t1_x.logsumexp(0) - torch.log(torch.tensor(MC_B)) # (MC_B, B, 1, 1) -> (B, 1, 1)
                # Z = (p_t1_x * J_).reshape(MC_B, b, 1, 1).mean(0) / (p_t_x + 1e-8) # (MC_B, B, 1, 1) -> (B, 1, 1)
                log_p_t1_x_times_J_ = log_p_t1_x + torch.log(J_).reshape(MC_B, b, 1, 1) # (MC_B, b, 1, 1)
                log_Z = log_p_t1_x_times_J_.logsumexp(0) - torch.log(torch.tensor(MC_B)) - log_p_t_x # (B, 1, 1)
                Z = torch.exp(log_Z) # (B, 1, 1)

                u = (x1_ - x_) / (1 - t + MC_EP) # (MC_B * b, T, C)
                g = torch.exp(log_p_t1_x - log_p_t_x) \
                    * (J_.reshape(MC_B, b, 1, 1) / (Z + 1e-8) - 1) \
                    * u.reshape(MC_B, b, *x_.shape[1:]) # (MC_B, b, T, C)
                return g.mean(0) # (MC_B, B, T, C) -> (B, T, C)

            elif self.cfg.flow_matching_type == 'ot_cfm':
                try:
                    x0_, x1_ = self._get_cached_ot_cfm_plan()
                except:
                    x0_ = torch.randn(MC_B, *x.shape[1:], device=x.device) # (MC_B, T, C)
                    x0_, x1_ = OTPlanSampler(method='exact').sample_plan(x0_, x1_)
                    x0_ = x0_.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x.shape[1:]) # (MC_B * b, T, C)
                    x1_ = x1_.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x.shape[1:]) # (MC_B * b, T, C)
                    self._save_cached_ot_cfm_plan(x0_, x1_)
                log_p_t1_x = ot_cfm_log_p_tz(x0_, x1_, x_, t, std=OT_STD) # (MC_B * b)
                
                if cached_v is None:
                    v_ = self.value_model(x1_)[:, -1, 0]
                else:
                    v_ = cached_v.clone()
                
                if self.cfg.mc_linear_J:
                    J_ = SCALE * v_ # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                    if self.cfg.mc_self_normalize:
                        J_ = ((J_ - J_.mean()) / (J_.std() + 1e-8)).clamp(0)
                else:
                    # self normalize
                    if self.cfg.mc_self_normalize:
                        v_ = (v_ - v_.mean()) / (v_.std() + 1e-8)
                    J_ = torch.exp(SCALE * v_) # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                
                log_p_t1_x = log_p_t1_x.reshape(MC_B, b, 1, 1)
                log_p_t_x = log_p_t1_x.logsumexp(0) - torch.log(torch.tensor(MC_B)) # (MC_B, B) -> (B, 1, 1)
                # Z = (p_t1_x * J_).reshape(MC_B, b, 1, 1).mean(0) / (p_t_x + 1e-8) # (MC_B, B) -> (B, 1, 1)
                log_p_t1_x_times_J_ = log_p_t1_x + torch.log(J_).reshape(MC_B, b, 1, 1) # (MC_B, b, 1, 1)
                log_Z = log_p_t1_x_times_J_.logsumexp(0) - torch.log(torch.tensor(MC_B)) - log_p_t_x # (B, 1, 1)
                Z = torch.exp(log_Z) # (B, 1, 1)

                u = x1_ - x0_ # (MC_B * b, T, C)
                g = torch.exp(log_p_t1_x - log_p_t_x) \
                    * (J_.reshape(MC_B, b, 1, 1) / (Z + 1e-8) - 1) \
                    * u.reshape(MC_B, b, *x_.shape[1:]) # (MC_B, b, T, C)
                return g.mean(0) # (MC_B, B, T, C) -> (B, T, C)
            else:
                raise ValueError(f"Unsupported flow matching type: {self.cfg.flow_matching_type}")
        return guide_fn
    
    def mc_forward(self, conditions, batch_size=1):
        # assert batch_size == 1, "env batch_size must be 1 for now" # but SS_B can be > 1
        if batch_size > 1:
            print("WARNING: batch_size > 1 for MC, this is not tested")
        assert self.cfg.guidance_method == 'mc', f"guidance_method must be mc, but got {self.cfg.guidance_method}"
        
        b = self.cfg.mc_ss

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # first, sample support set x1 ~ p_1(x)
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=None, 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
        x = torch.randn(self.cfg.mc_batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)
        with torch.no_grad():
            x1_support = solver(x, t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            )) # (MC_B, T, C)
        
        # Then sample guided x1 ~ p_1(x) e^{R(x)} / Z 
        # precompute the value model output for the support set
        x1_support_rep = x1_support.unsqueeze(0).repeat(b * batch_size, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x1_support.shape[1:]) # (MC_B * b, T, C)
        v_support = self.value_model(x1_support_rep)[:, -1, 0].detach() # (MC_B * b)
        
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_mc_guide_fn(x1_support, cached_v=v_support), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
        x = torch.randn(b * batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)
        with torch.no_grad():
            x = solver(x, t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            )) # (B, T, C)

        self._remove_cached_ot_cfm_plan()

        normed_actions = x[:, :, :self.action_dim]
        actions = torch.tensor(self.normalizer.unnormalize(normed_actions, 'actions'), device=self.cfg.device)

        normed_observations = x[:, :, self.action_dim:]
        observations = torch.tensor(self.normalizer.unnormalize(normed_observations, 'observations'), device=self.cfg.device) # NOTE: we do need to make torch tensor, otherwise the indexing later will be wrong
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1)) # (B_ss * B, T, 1)
        values = values[:, -1, 0] # (B_ss * B)
        values = values.reshape(b, batch_size) # (B_ss, B)
        best_idx = values.argmax(dim=0).to(self.cfg.device) # (B,)
        
        # to construct trajectories
        best_values = values[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B,)
        best_observations = observations.reshape(b, batch_size, self.horizon, self.state_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        best_actions = actions.reshape(b, batch_size, self.horizon, self.action_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        
        trajectories = Trajectories(to_np(best_actions), to_np(best_observations), to_np(best_values))

        # output actions
        actions = actions.reshape(b, batch_size, self.horizon, self.action_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        actions = to_np(actions[0, 0]) # (C,), simply get the first action in the first sample in the 
        
        return actions, trajectories


    ### Learned Guidance ###

    def get_learned_guidance_model(self, conditions):
        """
        Return the guidance model for the flow model.
        """
        def guide_fn(x, t, dx_dt, flow_model):
            if self.cfg.guide_matching_type != 'grad_z':
                with torch.no_grad():
                    guidance = self.guide_model(x, t) # input like flow model. x(B, T, C), t (,) or (B, )
            else:
                logz = self.guide_model(x, t)[:, -1, 0] # output is model_z output
                guidance = torch.autograd.grad([logz.sum()], [x])[0].detach()

            return guidance * self.cfg.guide_inference_scale
        return guide_fn

    def learned_guidance_forward(self, conditions, batch_size=1):
        """
        Use learned guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"
        assert self.cfg.guidance_method == 'guidance_matching', f"guidance_method must be learned, but got {self.cfg.guidance_method}"
        assert self.guide_model is not None, "guide_model is not provided"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_learned_guidance_model(conditions), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories
    
    ### Simple p(z|x_1) MC guidance ###
    
    def get_sim_mc_guidance_model(self, value_model, schedule_fn, scale):
        """
        Return the guidance model for the flow model.
        """

        def guide_fn(x, t, dx_dt, flow_model):
            """
            Implements guidance following Eq. 12
            Args:
                t: flow time. float
                x: current sample x_t. Tensor, shape (b, dim)
                dx_dt: current predicted VF. Tensor, shape (b, dim)
                model: flow model. MLP
            """
            x1_pred = x + dx_dt * (1 - t) # (B, 2)

            x1 = torch.randn_like(
                x1_pred.unsqueeze(0).repeat(self.cfg.sim_mc_n, 1, 1, 1)
            ) * self.cfg.sim_mc_std + x1_pred # (cfg.sim_mc_n, B, C, T)
            values = value_model(x1.reshape(-1, *x1.shape[2:]))[:, -1, 0] # (cfg.sim_mc_n * B)
            if self.cfg.sim_mc_self_normalize:
                values = (values - values.mean()) / (values.std() + 1e-8) # (cfg.sim_mc_n * B)
            Jx1_ = torch.exp(
                self.cfg.sim_mc_J_scale * values
            ).reshape(self.cfg.sim_mc_n, -1) # (cfg.sim_mc_n, B)
            v = (x1 - x) / (1 - t + self.cfg.sim_mc_eps)  # Conditional VF v_{t|z} in Eq. 12 (cfg.sim_mc_n, B, C, T)
            Z = Jx1_.mean(0) + 1e-8  # Z in Eq. 12 (B,)
            g = (Jx1_ / Z - 1).reshape(self.cfg.sim_mc_n, -1, 1, 1) * v  # g in Eq. 12 (cfg.sim_mc_n, B, C, T)
            g = g.mean(0) # (B, C, T)
            return g * scale * schedule_fn(t)
        return guide_fn
    
    def sim_mc_guidance_forward(self, conditions, batch_size):
        """
        Use g^{\text{sim-MC}} guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_sim_mc_guidance_model(
                self.value_model, 
                schedule_fn=self.get_scheduler(self.cfg.sim_mc_schedule), 
                scale=self.cfg.sim_mc_scale
            ), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories