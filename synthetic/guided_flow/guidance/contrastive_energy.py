
import torch


def contrastive_energy_loss(model, beta, J, xt, t):
    # sum e^-J / \sum e^-J * log(e^-model(xt, t) / \sum e^-model(xt, t))
    return -torch.sum(torch.softmax(-beta * J(xt), dim=0) * torch.softmax(-model(torch.cat([xt, t[:, None]], dim=-1)), dim=0).log())