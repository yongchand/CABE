import torch
import torch.nn.functional as F
import numpy as np


def moe_nig(u1, la1, alpha1, beta1, u2, la2, alpha2, beta2,
            eps=1e-8, max_la=1e6, max_alpha=1e6, max_beta=1e6):
    """
    Mixture of Normal-Inverse Gamma (MoNIG) aggregation
    (Equation 9 style) with basic numerical stabilizers.
    """
    # Standard aggregation
    u = (la1 * u1 + la2 * u2) / (la1 + la2 + eps)
    la = la1 + la2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)

    # Numerical stabilizers
    la = la.clamp(min=eps, max=max_la)
    alpha = alpha.clamp(min=1.0 + eps, max=max_alpha)
    beta = beta.clamp(min=eps, max=max_beta)

    return u, la, alpha, beta


def nig_nll_loss(mu, gamma, alpha, beta, y, eps=1e-8):
    """
    Correct Negative Log-Likelihood for Normal-Inverse-Gamma
    Implements Equation (5) from the MoNIG paper.

    Args:
        mu:    predicted mean (δ)
        gamma: predicted precision (λ > 0)
        alpha: shape    (α > 1)
        beta:  scale    (β > 0)
        y:     true labels
    """
    # Ensure numerical stability
    gamma = gamma.clamp(min=eps)
    alpha = alpha.clamp(min=1.0 + eps)
    beta  = beta.clamp(min=eps)

    # Ω = 2β(1 + γ)
    Omega = 2.0 * beta * (1.0 + gamma)

    # (y - μ)^2 γ + Ω
    S = (y - mu)**2 * gamma + Omega

    # log Ψ = log Γ(α) - log Γ(α + 1/2)
    log_Psi = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    # NLL = 1/2 log(π/γ)
    #     - α log Ω
    #     + (α + 1/2) log(S)
    #     + log(Ψ)
    nll = (
        0.5 * (torch.log(torch.tensor(torch.pi, device=gamma.device)) - torch.log(gamma))
        - alpha * torch.log(Omega)
        + (alpha + 0.5) * torch.log(S)
        + log_Psi
    )

    return torch.mean(nll)


def evidential_regularizer(mu, gamma, alpha, y):
    return torch.mean(torch.abs(y - mu) * (2 * gamma + alpha))


def criterion_nig(mu, gamma, alpha, beta, y, risk_weight=1e-3):
    nll = nig_nll_loss(mu, gamma, alpha, beta, y)
    reg = evidential_regularizer(mu, gamma, alpha, y)
    return nll + risk_weight * reg