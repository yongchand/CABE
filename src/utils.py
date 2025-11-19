import torch
import numpy as np


def moe_nig(u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
    """
    Mixture of Normal-Inverse Gamma (MoNIG) aggregation
    Equation 9 from the paper
    """
    u = (la1 * u1 + u2 * la2) / (la1 + la2)
    la = (la1 + la2)
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
    return u, la, alpha, beta


def criterion_nig(u, la, alpha, beta, y, hyp_params):
    """
    NIG loss function for evidential regression
    """
    om = 2 * beta * (1 + la)
    loss = sum(
        0.5 * torch.log(np.pi / la) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)) / len(u)
    lossr = hyp_params.risk * sum(torch.abs(u - y) * (2 * la + alpha)) / len(u)
    loss = loss + lossr
    return loss
