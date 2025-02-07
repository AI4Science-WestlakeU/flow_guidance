import torch
import torch.nn as nn
from guided_flow.backbone.transformer import Transformer


class TransformerWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transformer = Transformer(*args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        b, c, n = x.shape
        while t.dim() > 1: # remove tailing dimensions
            t = t[:, 0]
        if t.dim() == 0:
            t = t.repeat(b)
        # Now, t (B,); x (B, C, T)

        x = self.transformer(x, t)
        return x


class GuidedTransformerWrapper(nn.Module):
    """Wrap the transformer model in ODE solver for model that takes input with condition
    and returns output with condition
    E.g., input: x (B, C_action + C_state, T); output x (B, C_action + C_state, T)
    In CFM from Gaussian, this corresponds to sample (B, C_action, T) from Gaussian and
    set (B, C_state, T) as the initial condition of the ODE.
    """

    def __init__(self, model, get_condition=None, set_condition=None, guide_fn=None):
        super().__init__()
        self.model = model
        self.get_condition = get_condition
        self.set_condition = set_condition
        self.guide_fn = guide_fn

    def forward(self, t, x, *args, **kwargs):
        # x (B, C, T), t ()
        dx_dt = self.model(t, x)
        # add gradient guidance
        if self.guide_fn is not None:
            dx_dt = dx_dt + self.guide_fn(t, x, dx_dt, self.model)
        # fill in the condition
        zeros_x = torch.zeros_like(x)
        self.set_condition(dx_dt, self.get_condition(zeros_x))
        return dx_dt    

class MLPWrapper(torch.nn.Module):
    def __init__(self, model, scheduler=lambda t: 1, clamp=0.):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.clamp = clamp

    def forward(self, t, x, *args, **kwargs):
        model_out = self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
        if self.clamp > 0:
            model_out = model_out.clamp(-self.clamp, self.clamp)
        return self.scheduler(t) * model_out

class GuidedMLPWrapper(nn.Module):
    def __init__(self, model, guide_fn, scheduler=lambda t: 1):
        super().__init__()
        self.model = model
        self.guide_fn = guide_fn
        self.scheduler = scheduler
        
    def forward(self, t, x, *args, **kwargs):
        dx_dt = self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
        if self.guide_fn is not None:
            dx_dt = dx_dt + self.scheduler(t) * self.guide_fn(t, x, dx_dt, self.model)
        return dx_dt
    
# e_t = \log E_{x1\sim p(x1|xt)} e^{J(x1)}
class ExpEnergyMLPWrapper(nn.Module):
    def __init__(self, model, scheduler=lambda t: 1, clamp=0.):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.clamp = clamp

    def forward(self, t, x, *args, **kwargs):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            e = (self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1)).clamp(0)).log()
            grad = torch.autograd.grad(e.sum(), x, create_graph=True)[0]
            if self.clamp > 0:
                grad = grad.clamp(-self.clamp, self.clamp)
        return self.scheduler(t) * grad

class EnergyMLPWrapper(nn.Module):
    def __init__(self, model, scheduler=lambda t: 1):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def forward(self, t, x, *args, **kwargs):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            e = self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
            grad = -torch.autograd.grad(e.sum(), x, create_graph=True)[0]
        return self.scheduler(t) * grad

class GradEnergyMLPWrapper(nn.Module):
    def __init__(self, model, scheduler=lambda t: 1):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def forward(self, t, x, *args, **kwargs):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            grad = self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
            
        return self.scheduler(t) * grad