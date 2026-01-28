"""
Student-t distribution metrics for MoNIG models.

MoNIG outputs a Normal-Inverse-Gamma (NIG) distribution, which when marginalized
over the uncertainty parameters yields a Student-t predictive distribution.

This module provides proper CRPS and NLL calculations for Student-t distributions,
which are more accurate than Gaussian approximations for MoNIG models.
"""

import numpy as np
from scipy.stats import t
from scipy.special import gammaln
import warnings


def compute_student_t_nll(y_pred, y_std, y_true, alpha, nu=None):
    """
    Compute Negative Log-Likelihood (NLL) for Student-t distribution from NIG parameters.
    
    For a NIG distribution (μ, ν, α, β), the predictive distribution is Student-t with:
    - Location: μ
    - Scale: sqrt(β * (ν + 1) / (ν * (α - 1)))
    - Degrees of freedom: 2α
    
    Args:
        y_pred: predicted mean (μ) [n_samples]
        y_std: predicted std (sqrt of total variance) [n_samples]
        y_true: true values [n_samples]
        alpha: α parameter from NIG [n_samples] or scalar
        nu: ν parameter from NIG [n_samples] or scalar. If None, will be inferred.
    
    Returns:
        Mean NLL across all samples
    """
    y_pred = np.asarray(y_pred)
    y_std = np.asarray(y_std)
    y_true = np.asarray(y_true)
    alpha = np.asarray(alpha)
    
    # Ensure alpha > 1 (required for NIG)
    alpha = np.maximum(alpha, 1.01)
    
    # Degrees of freedom for Student-t
    df = 2 * alpha
    
    # If nu is not provided, infer it from epistemic/aleatoric ratio
    # This is a fallback - ideally nu should be saved during inference
    if nu is None:
        # We can't perfectly recover nu without additional info, but we can estimate
        # For now, use a reasonable default (this is a limitation)
        warnings.warn("nu not provided, using default estimation. Save nu during inference for accurate metrics.")
        nu = np.ones_like(alpha) * 10.0  # Default reasonable value
    
    nu = np.asarray(nu)
    
    # Total variance from NIG: β*(ν+1)/(ν*(α-1))
    # We have y_std^2 = total variance
    total_var = y_std ** 2
    
    # Compute scale parameter for Student-t
    # For NIG: Var = β*(ν+1)/(ν*(α-1))
    # For Student-t: Var = scale^2 * df/(df-2) for df > 2
    # Equating: scale^2 * df/(df-2) = β*(ν+1)/(ν*(α-1))
    # So: scale^2 = β*(ν+1)/(ν*(α-1)) * (df-2)/df
    # But we don't have β directly. However, we can use:
    # scale^2 = Var * (df-2)/df where Var = total_var
    
    # Ensure df > 2 for valid variance
    df_safe = np.maximum(df, 2.01)
    scale_sq = total_var * (df_safe - 2) / df_safe
    scale = np.sqrt(np.maximum(scale_sq, 1e-8))
    
    # Compute NLL for each sample
    nll_values = []
    for i in range(len(y_true)):
        try:
            # Student-t log PDF
            # log_pdf = log(gamma((df+1)/2)) - log(gamma(df/2)) - 0.5*log(π*df*scale^2)
            #          - (df+1)/2 * log(1 + ((y - μ)/scale)^2 / df)
            
            z = (y_true[i] - y_pred[i]) / scale[i]
            log_pdf = (gammaln((df[i] + 1) / 2) - 
                      gammaln(df[i] / 2) - 
                      0.5 * np.log(np.pi * df[i] * scale[i]**2) -
                      (df[i] + 1) / 2 * np.log(1 + z**2 / df[i]))
            
            nll = -log_pdf
            nll_values.append(nll)
        except Exception as e:
            # Fallback to scipy if manual calculation fails
            try:
                nll = -t.logpdf(y_true[i], df=df[i], loc=y_pred[i], scale=scale[i])
                nll_values.append(nll)
            except:
                # If still fails, use Gaussian approximation
                nll = 0.5 * (np.log(2 * np.pi * total_var[i]) + 
                            (y_true[i] - y_pred[i])**2 / total_var[i])
                nll_values.append(nll)
    
    return np.mean(nll_values)


def compute_student_t_crps(y_pred, y_std, y_true, alpha, nu=None, n_samples=1000):
    """
    Compute Continuous Ranked Probability Score (CRPS) for Student-t distribution.
    
    CRPS measures the accuracy of probabilistic forecasts by comparing the predicted
    CDF to the observed value.
    
    For Student-t, we use sampling-based CRPS estimation:
    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent samples from the Student-t distribution.
    
    Args:
        y_pred: predicted mean (μ) [n_samples]
        y_std: predicted std (sqrt of total variance) [n_samples]
        y_true: true values [n_samples]
        alpha: α parameter from NIG [n_samples] or scalar
        nu: ν parameter from NIG [n_samples] or scalar. If None, will be inferred.
        n_samples: number of samples for Monte Carlo estimation
    
    Returns:
        Mean CRPS across all samples
    """
    y_pred = np.asarray(y_pred)
    y_std = np.asarray(y_std)
    y_true = np.asarray(y_true)
    alpha = np.asarray(alpha)
    
    # Ensure alpha > 1
    alpha = np.maximum(alpha, 1.01)
    df = 2 * alpha
    
    if nu is None:
        warnings.warn("nu not provided, using default estimation. Save nu during inference for accurate metrics.")
        nu = np.ones_like(alpha) * 10.0
    
    nu = np.asarray(nu)
    
    # Compute scale parameter
    total_var = y_std ** 2
    df_safe = np.maximum(df, 2.01)
    scale_sq = total_var * (df_safe - 2) / df_safe
    scale = np.sqrt(np.maximum(scale_sq, 1e-8))
    
    # Compute CRPS for each sample
    crps_values = []
    
    for i in range(len(y_true)):
        try:
            # Sample from Student-t distribution
            samples = t.rvs(df=df[i], loc=y_pred[i], scale=scale[i], size=n_samples, random_state=42+i)
            
            # CRPS = E|X - y| - 0.5 * E|X - X'|
            # First term: mean absolute deviation from true value
            term1 = np.mean(np.abs(samples - y_true[i]))
            
            # Second term: mean pairwise absolute difference (half of it)
            # Sample pairs for efficiency
            n_pairs = min(500, n_samples * (n_samples - 1) // 2)
            if n_pairs < n_samples * (n_samples - 1) // 2:
                # Use random pairs for efficiency
                idx1 = np.random.choice(n_samples, size=n_pairs, replace=True)
                idx2 = np.random.choice(n_samples, size=n_pairs, replace=True)
                term2 = 0.5 * np.mean(np.abs(samples[idx1] - samples[idx2]))
            else:
                # Full pairwise (only for small n_samples)
                term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
            
            crps = term1 - term2
            crps_values.append(crps)
            
        except Exception as e:
            # Fallback to Gaussian CRPS approximation
            # Gaussian CRPS = σ * [z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π]
            # where z = (y - μ)/σ
            from scipy.stats import norm
            z = (y_true[i] - y_pred[i]) / y_std[i]
            crps = y_std[i] * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
            crps_values.append(crps)
    
    return np.mean(crps_values)


def compute_student_t_metrics_from_nig(y_pred, epistemic, aleatoric, y_true, alpha, nu):
    """
    Compute Student-t metrics directly from NIG parameters.
    
    This is the preferred method when you have full NIG parameters.
    
    Args:
        y_pred: predicted mean (μ) [n_samples]
        epistemic: epistemic uncertainty (β/(ν*(α-1))) [n_samples]
        aleatoric: aleatoric uncertainty (β/(α-1)) [n_samples]
        y_true: true values [n_samples]
        alpha: α parameter from NIG [n_samples]
        nu: ν parameter from NIG [n_samples]
    
    Returns:
        dict with 'nll' and 'crps' keys
    """
    y_pred = np.asarray(y_pred)
    epistemic = np.asarray(epistemic)
    aleatoric = np.asarray(aleatoric)
    y_true = np.asarray(y_true)
    alpha = np.asarray(alpha)
    nu = np.asarray(nu)
    
    # Ensure alpha > 1
    alpha = np.maximum(alpha, 1.01)
    
    # Degrees of freedom for Student-t
    df = 2 * alpha
    
    # Compute β from aleatoric: aleatoric = β/(α-1), so β = aleatoric*(α-1)
    beta = aleatoric * (alpha - 1)
    
    # Compute scale parameter for Student-t
    # From NIG: Var = β*(ν+1)/(ν*(α-1))
    # For Student-t: Var = scale^2 * df/(df-2) for df > 2
    # So: scale^2 = β*(ν+1)/(ν*(α-1)) * (df-2)/df
    
    # Ensure df > 2
    df_safe = np.maximum(df, 2.01)
    
    # Compute scale directly from NIG parameters
    # scale^2 = β*(ν+1)/(ν*(α-1)) * (df-2)/df
    # But we can simplify: since β = aleatoric*(α-1)
    # scale^2 = aleatoric*(α-1)*(ν+1)/(ν*(α-1)) * (df-2)/df
    # scale^2 = aleatoric*(ν+1)/ν * (df-2)/df
    
    scale_sq = aleatoric * (nu + 1) / nu * (df_safe - 2) / df_safe
    scale = np.sqrt(np.maximum(scale_sq, 1e-8))
    
    # Compute NLL for each sample
    nll_values = []
    for i in range(len(y_true)):
        try:
            z = (y_true[i] - y_pred[i]) / scale[i]
            log_pdf = (gammaln((df[i] + 1) / 2) - 
                      gammaln(df[i] / 2) - 
                      0.5 * np.log(np.pi * df[i] * scale[i]**2) -
                      (df[i] + 1) / 2 * np.log(1 + z**2 / df[i]))
            nll = -log_pdf
            nll_values.append(nll)
        except Exception:
            # Fallback to scipy
            try:
                nll = -t.logpdf(y_true[i], df=df[i], loc=y_pred[i], scale=scale[i])
                nll_values.append(nll)
            except:
                # Final fallback: Gaussian approximation
                total_var = epistemic[i] + aleatoric[i]
                nll = 0.5 * (np.log(2 * np.pi * total_var) + 
                            (y_true[i] - y_pred[i])**2 / total_var)
                nll_values.append(nll)
    
    nll = np.mean(nll_values)
    
    # Compute CRPS for each sample
    crps_values = []
    for i in range(len(y_true)):
        try:
            # Sample from Student-t distribution
            samples = t.rvs(df=df[i], loc=y_pred[i], scale=scale[i], size=1000, random_state=42+i)
            
            # CRPS = E|X - y| - 0.5 * E|X - X'|
            term1 = np.mean(np.abs(samples - y_true[i]))
            
            # Use random pairs for efficiency
            idx1 = np.random.choice(1000, size=500, replace=True)
            idx2 = np.random.choice(1000, size=500, replace=True)
            term2 = 0.5 * np.mean(np.abs(samples[idx1] - samples[idx2]))
            
            crps = term1 - term2
            crps_values.append(crps)
        except Exception:
            # Fallback to Gaussian CRPS
            from scipy.stats import norm
            total_var = epistemic[i] + aleatoric[i]
            y_std_i = np.sqrt(total_var)
            z = (y_true[i] - y_pred[i]) / y_std_i
            crps = y_std_i * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
            crps_values.append(crps)
    
    crps = np.mean(crps_values)
    
    return {'nll': nll, 'crps': crps}

