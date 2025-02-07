# Define gradient guidance function
# Using estimate \nabla_{E_{x1\sim p(x1|xt)} [x1]} J(E_{x1\sim p(x1|xt)} [x1])
import math
import torch

def get_scheduler(name, eps=1e-1):
    if name == "const":
        return lambda t: 1 + 0 * t # compatible for both float and tensor
    elif name == "linear_decay":
        return lambda t: 1 - t
    elif name == "cosine_decay":
        return lambda t: (torch.cos(t * torch.pi / 2))
    elif name == "exp_decay":
        return lambda t: (torch.exp(-t) - math.exp(-1)) / (1 - math.exp(-1))
    elif name == "linear_ramp": 
        return lambda t: t
    elif name == "cosine_ramp":
        return lambda t: (torch.cos(t * torch.pi / 2) + 1) / 2
    elif name == "exp_ramp":
        return lambda t: (torch.exp(t) - 1) / (math.exp(1) - 1)
    elif name == "linear_ramp_rt": 
        return lambda t: t + 0.4 # When ramping in rt, zero schedule results in div zero
    elif name == "cosine_ramp_rt":
        return lambda t: (torch.cos(t * torch.pi / 2) + 1) / 2 + 0.4  # When ramping in rt, zero schedule results in div zero
    elif name == "exp_ramp_rt":
        return lambda t: (torch.exp(t) - 1) / (math.exp(1) - 1) + 0.4  # When ramping in rt, zero schedule results in div zero
    elif name == "as_score":
        return lambda t: 1 / (t + eps) - 1
    elif name == "as_var":
        return lambda t: (1 / (t + eps) - 1).square()
    elif name == "pigdm":
        return lambda t: ((1 - t) ** 2 / ((1 - t) ** 2 + t ** 2) + 1e-8)
    elif name == "pigdm_gamma":
        return lambda t: torch.sqrt(t / (t ** 2 + (1 - t) ** 2 + 1e-8)) 
    elif name == "manual":
        return lambda t: ((1 - t - 0.1).exp() - 1).clamp(0) # first, counter the effect of an as_score schedule, then add scheduler
    else:
        raise ValueError(f"Schedule {name} not supported")

def get_schedule(name, t, eps=1e-1):
    return get_scheduler(name, eps)(t)


def wrap_grad_fn(scale, schedule_name, grad_fn, eps=1e-1):
    schedule = get_scheduler(schedule_name, eps)
    
    def wrapped_grad_fn(t, x, dx_dt, model):
        return scale * schedule(t) * grad_fn(t, x, dx_dt, model)
    return wrapped_grad_fn