import torch
import numpy as np

# -------------------------------------------------
#  Mixture of Experts — Normal-Inverse-Gamma
#  (This should already be here; keep as-is)
# -------------------------------------------------
def moe_nig(u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
    """
    Eq. 9 from the MoNIG paper (same as NIG repo)
    """
    u = (la1 * u1 + la2 * u2) / (la1 + la2)
    la = la1 + la2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (la1 * (u1 - u)**2 + la2 * (u2 - u)**2)
    return u, la, alpha, beta


def moe_nig_balanced(nigs, balance_factor: float = 0.5):
    """
    Balanced MoNIG aggregation that prevents single expert dominance.
    
    Args:
        nigs: List of (mu, v, alpha, beta) tuples for each expert
        balance_factor: Controls balance (0.0 = standard moe_nig, 1.0 = fully balanced)
    
    Returns:
        Aggregated (mu, v, alpha, beta)
    
    The balance_factor interpolates between:
    - Standard MoNIG (balance_factor=0): Direct precision-weighted aggregation
    - Balanced MoNIG (balance_factor=1): Equal-weight aggregation (democratic)
    """
    if len(nigs) == 0:
        raise ValueError("Cannot aggregate empty NIG list")
    if len(nigs) == 1:
        return nigs[0]
    
    # Extract all parameters
    mus = torch.stack([mu for mu, _, _, _ in nigs], dim=0)  # [E, B, 1]
    vs = torch.stack([v for _, v, _, _ in nigs], dim=0)  # [E, B, 1]
    alphas = torch.stack([alpha for _, _, alpha, _ in nigs], dim=0)  # [E, B, 1]
    betas = torch.stack([beta for _, _, _, beta in nigs], dim=0)  # [E, B, 1]
    
    # For balance_factor >= 0.8, use equal-weight aggregation (democratic)
    if balance_factor >= 0.8:
        # Equal weights: all experts contribute equally
        num_experts = len(nigs)
        vs_equal = vs.mean(dim=0, keepdim=True).expand_as(vs)  # All vs become mean
        vs_balanced = vs_equal
    else:
        # Compute normalized vs for balanced aggregation
        vs_mean = vs.mean(dim=0, keepdim=True)  # [1, B, 1]
        
        # Aggressive normalization: compress differences toward mean
        # Use power-law compression for extreme ratios
        vs_max = vs.max(dim=0, keepdim=True)[0]
        vs_min = vs.min(dim=0, keepdim=True)[0]
        vs_range = vs_max - vs_min + 1e-6
        
        # Normalize vs to reduce variance: compress differences toward mean
        compression_factor = balance_factor
        vs_normalized = vs_mean + (vs - vs_mean) * (1.0 - compression_factor)
        
        # For high balance factors, apply log-space compression for extreme ratios
        if balance_factor > 0.3:
            # Use log-space normalization to handle extreme ratios better
            vs_log = torch.log(vs + 1e-8)
            vs_log_mean = vs_log.mean(dim=0, keepdim=True)
            # More aggressive log compression
            log_compression = min(1.0, (balance_factor - 0.3) / 0.5 * 1.5)
            vs_log_normalized = vs_log_mean + (vs_log - vs_log_mean) * (1.0 - log_compression)
            vs_log_normalized_exp = torch.exp(vs_log_normalized)
            # Blend: use log-space normalization when balance_factor is high
            blend_weight = max(0.0, (balance_factor - 0.3) / 0.5)
            vs_normalized = (1.0 - blend_weight) * vs_normalized + blend_weight * vs_log_normalized_exp
        
        # Ensure vs_normalized are positive
        vs_normalized = torch.clamp(vs_normalized, min=1e-6)
        
        # Interpolate between original and normalized vs
        vs_balanced = (1.0 - balance_factor) * vs + balance_factor * vs_normalized
    
    # Aggregate using sequential moe_nig with balanced v values
    # Start with first expert
    mu_final, v_final, alpha_final, beta_final = nigs[0]
    v_final_balanced = vs_balanced[0]
    
    for i in range(1, len(nigs)):
        mu_i, _, alpha_i, beta_i = nigs[i]
        v_i_balanced = vs_balanced[i]
        # Use balanced v values for aggregation
        mu_final, v_final, alpha_final, beta_final = moe_nig(
            mu_final, v_final_balanced, alpha_final, beta_final,
            mu_i, v_i_balanced, alpha_i, beta_i
        )
        # Update v_final_balanced for next iteration (use aggregated v)
        v_final_balanced = v_final
    
    return mu_final, v_final, alpha_final, beta_final


# -------------------------------------------------
#  NIG loss (from NIG repo)
# -------------------------------------------------
def criterion_nig(u, la, alpha, beta, y, hyp_params):
    """
    Normal-Inverse-Gamma evidential loss function.
    Matches the original NIG implementation.
    """
    om = 2 * beta * (1 + la)
    n = len(u)

    loss = (
        0.5 * torch.log(np.pi / la)
        - alpha * torch.log(om)
        + (alpha + 0.5) * torch.log(la * (u - y)**2 + om)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    loss = torch.sum(loss) / n

    # risk regularizer
    loss_r = hyp_params.risk * torch.sum(torch.abs(u - y) * (2 * la + alpha)) / n

    return loss + loss_r


# -------------------------------------------------
#  Legacy-facing aliases expected by your code
# -------------------------------------------------
def nig_loss(u, la, alpha, beta, y):
    """
    Backwards-compatible wrapper for training code.
    Matches call signature: nig_loss(mu, v, alpha, beta, labels)
    """
    class Hyp:
        pass
    hyp = Hyp()
    hyp.risk = 0.01  # default risk to match your training script

    return criterion_nig(u, la, alpha, beta, y, hyp)

def nig_to_prediction(nigs):
    """
    Convert model outputs to a single NIG (mu, la, alpha, beta).

    Your training code calls:
        mu, v, alpha, beta = nig_to_prediction(nigs)

    So this function MUST:
      - accept `nigs` (either a single (mu, la, alpha, beta) tuple
        or a list/tuple of such tuples for multiple experts),
      - return exactly 4 tensors: (mu, la, alpha, beta).
    """
    # Case 1: already a single NIG tuple
    if isinstance(nigs, tuple) and len(nigs) == 4:
        return nigs  # (mu, la, alpha, beta)

    # Case 2: list/tuple of NIGs from multiple experts (MoNIG)
    if isinstance(nigs, (list, tuple)):
        if len(nigs) == 0:
            raise ValueError("nig_to_prediction received an empty list of NIGs.")

        # Start from the first expert
        mu, la, alpha, beta = nigs[0]

        # Sequentially fuse others with moe_nig
        for (u2, la2, alpha2, beta2) in nigs[1:]:
            mu, la, alpha, beta = moe_nig(mu, la, alpha, beta, u2, la2, alpha2, beta2)

        return mu, la, alpha, beta

    # Anything else is a bug
    raise TypeError(
        f"nig_to_prediction expected a tuple or list of tuples, "
        f"but got object of type {type(nigs)}"
    )