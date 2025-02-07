
# Define gradient guidance function
# Using estimate \nabla_{E_{x1\sim p(x1|xt)} [x1]} J(E_{x1\sim p(x1|xt)} [x1])
import math
import torch

def wrap_grad_fn(scale, schedule, grad_fn, eps=1e-1):
    if schedule == "const":
        schedule = lambda t: 1
    elif schedule == "linear_decay":
        schedule = lambda t: 1 - t
    elif schedule == "cosine_decay":
        schedule = lambda t: (torch.cos(t * torch.pi / 2))
    elif schedule == "exp_decay":
        schedule = lambda t: (torch.exp(-t) - math.exp(-1)) / (1 - math.exp(-1))
    elif schedule == "as_score":
        schedule = lambda t: 1 / (t + eps) - 1
    elif schedule == "as_var":
        schedule = lambda t: (1 / (t + eps) - 1).square()
    else:
        raise ValueError(f"Schedule {schedule} not supported")
    
    def wrapped_grad_fn(t, x, dx_dt, model):
        return scale * schedule(t) * grad_fn(t, x, dx_dt, model)
    return wrapped_grad_fn
