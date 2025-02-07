import torch


class MLP(torch.nn.Module):
    def __init__(
            self, 
            dim, 
            out_dim=None, 
            w=64, 
            time_varying=False, 
            time_embedding=False,
            clamp_to=None, 
            soft_clamp=False,
            exp_final=False
    ):
        super().__init__()
        self.time_varying = time_varying
        self.time_embedding = time_embedding
        if out_dim is None:
            out_dim = dim

        if time_varying:
            if not time_embedding:
                maybe_time_dim = 1
            else:
                self.half_t_embed_dim = dim
                maybe_time_dim = self.half_t_embed_dim * 2
        else:
            maybe_time_dim = 0

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + maybe_time_dim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )
        if clamp_to is not None:
            self.clamp_output_to = clamp_to
            self.soft_clamp = soft_clamp
        else:
            self.clamp_output_to = None
        self.exp_final = exp_final

    def time_embedder(self, t):
        assert self.time_embedding
        freq = torch.arange(self.half_t_embed_dim, device=t.device).float()
        t = t * torch.exp(-freq * torch.log(torch.tensor(10000.0)) / self.half_t_embed_dim)
        t = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        return t
        
    def forward(self, x):
        if self.time_embedding:
            t = x[..., -1:]
            t = self.time_embedder(t)
            x = torch.cat([x[..., :-1], t], dim=-1)

        out = self.net(x)
        if self.clamp_output_to is not None:
            if self.soft_clamp:
                out = torch.nn.functional.softplus(out, beta=1 / self.clamp_output_to)
            else:
                out = out.clamp(min=self.clamp_output_to) # for energy function
        if self.exp_final:
            out = torch.exp(out)
        return out

class EnergyMLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, time_embedding=False):
        super().__init__()
        self.net = MLP(dim, out_dim, w, time_varying, time_embedding)

    def forward(self, x):
        return self.net(x).sum(dim=-1)

class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]

