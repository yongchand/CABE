import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from src.utils import flatten, unflatten_like, moe_nig


class DrugDiscoveryMoNIGEmb(nn.Module):
    """
    Drug Discovery model using Mixture of Normal-Inverse Gamma (MoNIG) with reliability network
    
    Architecture:
    - Expert Scores → Engine Score MLP → Per-expert NIG params (μ_j, α_j, β_j, v_j)
    - 703D Embeddings → Reliability Net (MLP) → r_j ∈ (0,1) per engine
    - Scale α_j, β_j, v_j with r_j
    - MoNIG Expert Fusion → Final Aggregated NIG
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryMoNIGEmb, self).__init__()
        
        self.num_experts = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        
        # Engine Score MLP: Expert scores → NIG parameters
        # Per-expert evidential heads (produce NIG params from expert scores)
        self.evidential_heads = nn.ModuleList([
            self._build_evidential_head() for _ in range(self.num_experts)
        ])
        
        # Reliability Net: Embeddings → reliability scores r_j ∈ (0,1) per engine
        # Very simple architecture to reduce overfitting and encourage learning
        self.reliability_net = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),  # Further reduced
            nn.ReLU(),
            nn.Dropout(self.dropout * 2.0),  # Higher dropout (0.4)
            nn.Linear(128, self.num_experts),  # Direct to output
            nn.Sigmoid()  # Ensure r_j ∈ (0,1)
        )
    
    def _build_evidential_head(self):
        """
        Evidential head that produces NIG parameters from expert score
        Input: expert score (1)
        Output: μ, ν, α, β (NIG parameters)
        """
        return nn.ModuleDict({
            'fusion': nn.Sequential(
                nn.Linear(1, self.hidden_dim),  # Calibrated score (1)
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            ),
            'mu_head': nn.Linear(64, 1),      # Mean
            'v_head': nn.Linear(64, 1),       # Precision parameter
            'alpha_head': nn.Linear(64, 1),   # Shape parameter
            'beta_head': nn.Linear(64, 1),    # Scale parameter
        })
    
    def evidence(self, x):
        """Apply softplus to ensure positive values"""
        return F.softplus(x)
    
    def forward(self, expert_scores, embeddings):
        """
        Forward pass
        
        Args:
            expert_scores: [batch, num_experts] - Pre-computed expert predictions
            embeddings: [batch, embedding_dim] - 703D protein/ligand embeddings
        
        Returns:
            List of (mu, v, alpha, beta) tuples, one per expert (after reliability scaling)
        """
        # Step 1: Engine Score MLP → Per-expert NIG params
        nigs = []
        for i in range(self.num_experts):
            # Get expert score directly
            expert_i = expert_scores[:, i:i+1]  # [batch, 1]
            
            # Process through evidential head
            head = self.evidential_heads[i]
            fused = head['fusion'](expert_i)  # [batch, 64]
            
            # Predict NIG parameters
            mu = head['mu_head'](fused)          # [batch, 1]
            logv = head['v_head'](fused)         # [batch, 1]
            logalpha = head['alpha_head'](fused) # [batch, 1]
            logbeta = head['beta_head'](fused)   # [batch, 1]
            
            # Apply evidence function to ensure proper parameter ranges
            v = self.evidence(logv)              # ν > 0
            alpha = self.evidence(logalpha) + 1  # α > 1
            beta = self.evidence(logbeta)        # β > 0
            
            nigs.append((mu, v, alpha, beta))
        
        # Step 2: Reliability Net → reliability scores r_j ∈ (0,1) per engine
        reliability_scores = self.reliability_net(embeddings)  # [batch, num_experts]
        
        # Step 3: Scale α_j, β_j, v_j with r_j (μ_j is NOT scaled)
        scaled_nigs = []
        for i in range(self.num_experts):
            mu, v, alpha, beta = nigs[i]
            r_j = reliability_scores[:, i:i+1]  # [batch, 1] reliability for expert i
            
            # Scale v, alpha, beta with reliability (but keep mu unchanged)
            # For alpha: ensure α > 1 after scaling by scaling the excess above 1
            v_scaled = v * r_j
            alpha_scaled = 1.0 + (alpha - 1.0) * r_j + 1e-6  # Ensure α > 1
            beta_scaled = beta * r_j
            
            scaled_nigs.append((mu, v_scaled, alpha_scaled, beta_scaled))
        
        return scaled_nigs


# ============================================================================
# Ablation Model Variants for CABE
# ============================================================================

class DrugDiscoveryMoNIG_NoReliabilityScaling(DrugDiscoveryMoNIGEmb):
    """
    Ablation 1: Remove reliability scaling (set r_j = 1.0 for all experts)
    This should show that MoNIG collapses without reliability scaling.
    """
    def forward(self, expert_scores, embeddings):
        # Get per-expert NIG params (same as base)
        nigs = []
        for i in range(self.num_experts):
            expert_i = expert_scores[:, i:i+1]
            head = self.evidential_heads[i]
            fused = head['fusion'](expert_i)
            mu = head['mu_head'](fused)
            logv = head['v_head'](fused)
            logalpha = head['alpha_head'](fused)
            logbeta = head['beta_head'](fused)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
            nigs.append((mu, v, alpha, beta))
        
        # ABLATION: Set r_j = 1.0 (no scaling)
        batch_size = expert_scores.shape[0]
        reliability_scores = torch.ones(batch_size, self.num_experts, device=expert_scores.device)
        
        # Apply "scaling" with r_j = 1.0 (effectively no scaling)
        scaled_nigs = []
        for i in range(self.num_experts):
            mu, v, alpha, beta = nigs[i]
            r_j = reliability_scores[:, i:i+1]
            v_scaled = v * r_j  # v * 1.0 = v
            alpha_scaled = 1.0 + (alpha - 1.0) * r_j + 1e-6  # alpha (no change)
            beta_scaled = beta * r_j  # beta * 1.0 = beta
            scaled_nigs.append((mu, v_scaled, alpha_scaled, beta_scaled))
        
        return scaled_nigs


class DrugDiscoveryMoNIG_UniformReliability(DrugDiscoveryMoNIGEmb):
    """
    Ablation 2: Use uniform reliability (r_j = 1/num_experts for all experts)
    This tests if learned reliability matters vs uniform priors.
    """
    def forward(self, expert_scores, embeddings):
        # Get per-expert NIG params (same as base)
        nigs = []
        for i in range(self.num_experts):
            expert_i = expert_scores[:, i:i+1]
            head = self.evidential_heads[i]
            fused = head['fusion'](expert_i)
            mu = head['mu_head'](fused)
            logv = head['v_head'](fused)
            logalpha = head['alpha_head'](fused)
            logbeta = head['beta_head'](fused)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
            nigs.append((mu, v, alpha, beta))
        
        # ABLATION: Set uniform reliability r_j = 1/num_experts
        batch_size = expert_scores.shape[0]
        uniform_r = 1.0 / self.num_experts
        reliability_scores = torch.full(
            (batch_size, self.num_experts), 
            uniform_r, 
            device=expert_scores.device
        )
        
        # Apply scaling with uniform reliability
        scaled_nigs = []
        for i in range(self.num_experts):
            mu, v, alpha, beta = nigs[i]
            r_j = reliability_scores[:, i:i+1]
            v_scaled = v * r_j
            alpha_scaled = 1.0 + (alpha - 1.0) * r_j + 1e-6
            beta_scaled = beta * r_j
            scaled_nigs.append((mu, v_scaled, alpha_scaled, beta_scaled))
        
        return scaled_nigs


class DrugDiscoveryMoNIG_NoContextReliability(DrugDiscoveryMoNIGEmb):
    """
    Ablation 3: Use per-expert learned reliability (no context dependence)
    Each expert has its own learned reliability parameter r_j that is constant across samples.
    This tests if context-dependent reliability matters vs per-expert constant reliability.
    """
    def __init__(self, hyp_params):
        super().__init__(hyp_params)
        # Learn per-expert reliability parameters (not context-dependent)
        # Initialize with sigmoid to ensure r_j ∈ (0,1)
        self.reliability_params = nn.Parameter(torch.ones(self.num_experts) * 0.5)
    
    def forward(self, expert_scores, embeddings):
        # Get per-expert NIG params (same as base)
        nigs = []
        for i in range(self.num_experts):
            expert_i = expert_scores[:, i:i+1]
            head = self.evidential_heads[i]
            fused = head['fusion'](expert_i)
            mu = head['mu_head'](fused)
            logv = head['v_head'](fused)
            logalpha = head['alpha_head'](fused)
            logbeta = head['beta_head'](fused)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
            nigs.append((mu, v, alpha, beta))
        
        # ABLATION: Use per-expert learned reliability (ignore embeddings, constant per expert)
        # Apply sigmoid to ensure r_j ∈ (0,1)
        reliability_values = torch.sigmoid(self.reliability_params)  # [num_experts]
        batch_size = expert_scores.shape[0]
        # Expand to [batch, num_experts] - same reliability for all samples per expert
        reliability_scores = reliability_values.unsqueeze(0).expand(batch_size, -1)
        
        # Apply scaling with per-expert reliability
        scaled_nigs = []
        for i in range(self.num_experts):
            mu, v, alpha, beta = nigs[i]
            r_j = reliability_scores[:, i:i+1]  # [batch, 1] - same value for all samples
            v_scaled = v * r_j
            alpha_scaled = 1.0 + (alpha - 1.0) * r_j + 1e-6
            beta_scaled = beta * r_j
            scaled_nigs.append((mu, v_scaled, alpha_scaled, beta_scaled))
        
        return scaled_nigs

class DrugDiscoveryConsensusScoring(nn.Module):
    """
    Exponential Consensus Ranking (ECR) adapted for regression.

    Based on: Palacio-Rodriguez et al., "Exponential consensus ranking improves
    the outcome in docking and receptor ensemble docking", Sci Rep 9, 5142 (2019)
    https://www.nature.com/articles/s41598-019-41594-3

    Original ECR formula (for ranking): P(i) = (1/σ) * Σ_j exp(-r_ij / σ)

    Adapted for regression:
    - For each sample, compute median as robust consensus estimate
    - Weight each expert by exp(-|s_j - consensus| / σ)
    - Experts closer to consensus get exponentially higher weights
    - Final prediction = weighted average
    - Uncertainty = weighted standard deviation

    This is a heuristic baseline with NO learnable parameters.
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryConsensusScoring, self).__init__()
        self.num_experts = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        # σ parameter controls how quickly weights decay with disagreement
        # Typical pKd range is ~2-3 units, so σ=1.0 is reasonable
        self.sigma = 1.0

    def forward(self, expert_scores, embeddings):
        """
        Forward pass with ECR-style exponential consensus weighting.

        Args:
            expert_scores: [batch, num_experts] - Docking engine predictions (pKd scale)
            embeddings: [batch, embedding_dim] - Not used (for interface compatibility)

        Returns:
            mean: [batch, 1] - Consensus-weighted prediction
            std: [batch, 1] - Uncertainty based on weighted disagreement
        """
        # Compute robust consensus (median across experts for each sample)
        consensus = expert_scores.median(dim=1, keepdim=True)[0]  # [batch, 1]

        # ECR-style exponential weighting: exp(-|deviation| / sigma)
        # Experts closer to consensus get higher weights
        deviation = torch.abs(expert_scores - consensus)  # [batch, num_experts]
        weights = torch.exp(-deviation / self.sigma)  # [batch, num_experts]

        # Normalize weights to sum to 1
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted mean prediction
        mean = (weights * expert_scores).sum(dim=1, keepdim=True)

        # Weighted standard deviation as uncertainty
        squared_diff = (expert_scores - mean) ** 2
        weighted_var = (weights * squared_diff).sum(dim=1, keepdim=True)
        std = torch.sqrt(weighted_var + 1e-8)

        return mean, std


class DrugDiscoveryEnsembleScoring(nn.Module):
    """
    ENS_Score: Simple Ensemble Scoring for binding affinity prediction.

    Based on: Rayka et al., ENS_Score ensemble method
    https://github.com/miladrayka/ENS_Score

    Method:
    - Simple arithmetic mean of all expert predictions
    - Standard deviation as uncertainty measure
    - ENS_Score = (1/N) × Σ(Expert_i prediction)

    This is the most common ensemble approach in drug discovery and
    achieved Rp=0.842 on CASF-2016 benchmark.

    This is a heuristic baseline with NO learnable parameters.
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryEnsembleScoring, self).__init__()
        self.num_experts = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim

    def forward(self, expert_scores, embeddings):
        """
        Forward pass with simple averaging (ENS_Score style).

        Args:
            expert_scores: [batch, num_experts] - Docking engine predictions (pKd scale)
            embeddings: [batch, embedding_dim] - Not used (for interface compatibility)

        Returns:
            mean: [batch, 1] - Simple average of engine predictions
            std: [batch, 1] - Standard deviation across engines (uncertainty)
        """
        # ENS_Score: Simple arithmetic mean
        mean = expert_scores.mean(dim=1, keepdim=True)

        # Standard deviation as uncertainty (prediction confidence)
        std = expert_scores.std(dim=1, keepdim=True)

        # Ensure non-zero std for numerical stability
        std = torch.clamp(std, min=1e-6)

        return mean, std


class DrugDiscoveryMoNIG_ScoresOnlyReliability(DrugDiscoveryMoNIGEmb):
    """
    Ablation: Reliability network takes only engine scores, not h(x) embeddings.
    
    r_i(x) = g_phi([s1, s2, s3, s4])_i
    
    This tests whether the embeddings h(x) contribute to reliability estimation,
    or if the engine scores alone are sufficient.
    """
    def __init__(self, hyp_params):
        super().__init__(hyp_params)
        # Override reliability net to take only engine scores (not embeddings)
        self.reliability_net = nn.Sequential(
            nn.Linear(self.num_experts, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts),
            nn.Sigmoid()
        )
    
    def forward(self, expert_scores, embeddings):
        # Get per-expert NIG params (same as base)
        nigs = []
        for i in range(self.num_experts):
            expert_i = expert_scores[:, i:i+1]
            head = self.evidential_heads[i]
            fused = head['fusion'](expert_i)
            mu = head['mu_head'](fused)
            logv = head['v_head'](fused)
            logalpha = head['alpha_head'](fused)
            logbeta = head['beta_head'](fused)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
            nigs.append((mu, v, alpha, beta))
        
        # ABLATION: Reliability from scores only (ignore embeddings)
        reliability_scores = self.reliability_net(expert_scores)  # [batch, num_experts]
        
        # Apply scaling with scores-only reliability
        scaled_nigs = []
        for i in range(self.num_experts):
            mu, v, alpha, beta = nigs[i]
            r_j = reliability_scores[:, i:i+1]
            v_scaled = v * r_j
            alpha_scaled = 1.0 + (alpha - 1.0) * r_j + 1e-6
            beta_scaled = beta * r_j
            scaled_nigs.append((mu, v_scaled, alpha_scaled, beta_scaled))
        
        return scaled_nigs


class DrugDiscoveryMoNIG_UniformWeightAggregation(DrugDiscoveryMoNIGEmb):
    """
    Ablation: Uniform weight aggregation (no learned reliability).

    Tests the importance of learned reliability by using r_j = 1 for all experts
    (i.e., all experts are equally "reliable") and then applying proper MoNIG aggregation.

    This is different from the full MoNIG which learns context-dependent reliability.
    """
    def forward(self, expert_scores, embeddings):
        # Get per-expert NIG params (without reliability scaling)
        nigs = []
        for i in range(self.num_experts):
            expert_i = expert_scores[:, i:i+1]
            head = self.evidential_heads[i]
            fused = head['fusion'](expert_i)
            mu = head['mu_head'](fused)
            logv = head['v_head'](fused)
            logalpha = head['alpha_head'](fused)
            logbeta = head['beta_head'](fused)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
            nigs.append((mu, v, alpha, beta))

        # Apply proper MoNIG aggregation (Equation 9) with uniform weights
        # This is mathematically correct aggregation of NIG distributions
        mu_final, v_final, alpha_final, beta_final = nigs[0]
        for i in range(1, len(nigs)):
            mu_i, v_i, alpha_i, beta_i = nigs[i]
            # MoNIG aggregation formula
            mu_new = (v_final * mu_final + v_i * mu_i) / (v_final + v_i + 1e-8)
            v_new = v_final + v_i
            alpha_new = alpha_final + alpha_i + 0.5
            beta_new = (beta_final + beta_i +
                       0.5 * (v_final * (mu_final - mu_new) ** 2 +
                              v_i * (mu_i - mu_new) ** 2))
            mu_final, v_final, alpha_final, beta_final = mu_new, v_new, alpha_new, beta_new

        # Return as list for compatibility with training code
        return [(mu_final, v_final, alpha_final, beta_final)]

class DrugDiscoveryNIGEmb(nn.Module):
    """
    Baseline: Single NIG model without mixture of experts (with embeddings)
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryNIGEmb, self).__init__()
        
        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.expert_dim + self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # NIG parameter heads
        self.mu_head = nn.Linear(64, 1)
        self.v_head = nn.Linear(64, 1)
        self.alpha_head = nn.Linear(64, 1)
        self.beta_head = nn.Linear(64, 1)
    
    def evidence(self, x):
        return F.softplus(x)
    
    def forward(self, expert_scores, embeddings):
        """
        Forward pass
        Returns single NIG distribution
        """
        # Concatenate all inputs
        x = torch.cat([expert_scores, embeddings], dim=1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Predict NIG parameters
        mu = self.mu_head(encoded)
        logv = self.v_head(encoded)
        logalpha = self.alpha_head(encoded)
        logbeta = self.beta_head(encoded)
        
        # Apply evidence function
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        
        return mu, v, alpha, beta


class DrugDiscoveryGaussianEmb(nn.Module):
    """
    Gaussian model: Predicts mean and variance (simpler than NIG)
    Output: μ (mean) and σ² (variance)
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryGaussianEmb, self).__init__()
        
        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.expert_dim + self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Output heads
        self.mu_head = nn.Linear(64, 1)      # Mean
        self.sigma_head = nn.Linear(64, 1)  # Variance
    
    def evidence(self, x):
        """Ensure positive variance"""
        return F.softplus(x)
    
    def forward(self, expert_scores, embeddings):
        """
        Forward pass
        
        Returns:
            mu: predicted mean [batch, 1]
            sigma: predicted variance [batch, 1] (must be positive)
        """
        # Concatenate all inputs
        x = torch.cat([expert_scores, embeddings], dim=1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Predict mean and variance
        mu = self.mu_head(encoded)
        log_sigma = self.sigma_head(encoded)
        sigma = self.evidence(log_sigma)  # Ensure σ² > 0
        
        return mu, sigma


class DrugDiscoveryBaselineEmb(nn.Module):
    """
    Simple baseline: MLP regression without uncertainty (with embeddings)
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryBaselineEmb, self).__init__()
        
        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        
        self.model = nn.Sequential(
            nn.Linear(self.expert_dim + self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, expert_scores, embeddings):
        x = torch.cat([expert_scores, embeddings], dim=1)
        return self.model(x)


class DrugDiscoveryDeepEnsemble(nn.Module):
    """
    Deep Ensemble: Ensemble of multiple baseline models for uncertainty estimation
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryDeepEnsemble, self).__init__()
        
        self.num_models = hyp_params.num_models if hasattr(hyp_params, 'num_models') else 5
        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            DrugDiscoveryBaselineEmb(hyp_params) for _ in range(self.num_models)
        ])
    
    def forward(self, expert_scores, embeddings):
        """
        Forward pass: average predictions from all models
        
        Returns:
            mean: averaged prediction [batch, 1]
            std: standard deviation across models [batch, 1]
        """
        predictions = []
        for model in self.models:
            pred = model(expert_scores, embeddings)  # [batch, 1]
            predictions.append(pred)
        
        # Stack and compute statistics
        pred_stack = torch.stack(predictions, dim=0)  # [num_models, batch, 1]
        mean = pred_stack.mean(dim=0)  # [batch, 1]
        std = pred_stack.std(dim=0)  # [batch, 1]
        
        return mean, std


class DrugDiscoveryMCDropout(nn.Module):
    """
    Monte Carlo Dropout with Heteroscedastic Output

    Combines MC Dropout for epistemic uncertainty with learned aleatoric uncertainty.
    The model predicts both mean and variance, and uses dropout sampling at inference.

    Total uncertainty = aleatoric (learned) + epistemic (from MC sampling)

    Reference: Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
               for Computer Vision?", NeurIPS 2017
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryMCDropout, self).__init__()

        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        self.num_samples = hyp_params.num_mc_samples if hasattr(hyp_params, 'num_mc_samples') else 50

        # Shared encoder with dropout
        self.encoder = nn.Sequential(
            nn.Linear(self.expert_dim + self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Separate heads for mean and variance (heteroscedastic)
        self.mu_head = nn.Linear(64, 1)
        self.logvar_head = nn.Linear(64, 1)  # Log variance for numerical stability

    def forward_single(self, x):
        """Single forward pass returning mean and variance."""
        encoded = self.encoder(x)
        mu = self.mu_head(encoded)
        log_var = self.logvar_head(encoded)
        var = F.softplus(log_var) + 1e-6  # Ensure positive variance
        return mu, var

    def forward(self, expert_scores, embeddings, num_samples=None):
        """
        Forward pass with Monte Carlo sampling

        Args:
            expert_scores: [batch, num_experts]
            embeddings: [batch, embedding_dim]
            num_samples: Number of MC samples (default: self.num_samples)

        Returns:
            mean: mean prediction [batch, 1]
            std: total uncertainty (aleatoric + epistemic) [batch, 1]
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Enable dropout for MC sampling
        was_training = self.training
        self.train()

        x = torch.cat([expert_scores, embeddings], dim=1)
        means = []
        variances = []

        for _ in range(num_samples):
            mu, var = self.forward_single(x)
            means.append(mu)
            variances.append(var)

        # Restore original training state
        if not was_training:
            self.eval()

        # Stack predictions
        means_stack = torch.stack(means, dim=0)  # [num_samples, batch, 1]
        variances_stack = torch.stack(variances, dim=0)  # [num_samples, batch, 1]

        # Compute final mean
        mean = means_stack.mean(dim=0)  # [batch, 1]

        # Total uncertainty using law of total variance:
        # Var[Y] = E[Var[Y|θ]] + Var[E[Y|θ]]
        # aleatoric = E[σ²(θ)] - average predicted variance
        # epistemic = Var[μ(θ)] - variance of predicted means (from dropout)
        aleatoric = variances_stack.mean(dim=0)  # [batch, 1]
        epistemic = means_stack.var(dim=0)  # [batch, 1]

        total_std = torch.sqrt(aleatoric + epistemic + 1e-8)

        return mean, total_std

    def train_forward(self, expert_scores, embeddings):
        """
        Forward pass for training (single pass, returns mean and variance).

        Use this during training with Gaussian NLL loss.
        """
        x = torch.cat([expert_scores, embeddings], dim=1)
        return self.forward_single(x)


class DrugDiscoverySoftmaxMoE(nn.Module):
    """
    Softmax Mixture-of-Experts baseline: Learn attention weights using softmax
    over experts and aggregate their predictions.
    
    This is simpler than MoNIG as it doesn't use evidential deep learning,
    but still learns expert importance dynamically.
    """
    def __init__(self, hyp_params):
        super(DrugDiscoverySoftmaxMoE, self).__init__()
        
        self.num_experts = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        
        # Per-expert prediction heads (transform expert scores to refined predictions)
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(self.num_experts)
        ])
        
        # Attention network: Embeddings → attention weights
        self.attention_net = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_experts)  # Output attention logits
        )
        
        # Uncertainty estimation network (predict heteroscedastic uncertainty)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_experts, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, expert_scores, embeddings):
        """
        Forward pass
        
        Args:
            expert_scores: [batch, num_experts] - Expert predictions
            embeddings: [batch, embedding_dim] - Protein/ligand embeddings
        
        Returns:
            mean: Weighted prediction [batch, 1]
            std: Predicted uncertainty [batch, 1]
        """
        # Step 1: Compute attention weights using softmax
        attention_logits = self.attention_net(embeddings)  # [batch, num_experts]
        attention_weights = F.softmax(attention_logits, dim=1)  # [batch, num_experts]
        
        # Step 2: Refine expert predictions
        refined_predictions = []
        for i in range(self.num_experts):
            expert_i = expert_scores[:, i:i+1]  # [batch, 1]
            refined = self.expert_heads[i](expert_i)  # [batch, 1]
            refined_predictions.append(refined)
        
        # Stack predictions [batch, num_experts]
        refined_predictions = torch.cat(refined_predictions, dim=1)
        
        # Step 3: Weighted aggregation
        # mean = Σ w_j * pred_j
        mean = torch.sum(attention_weights * refined_predictions, dim=1, keepdim=True)  # [batch, 1]
        
        # Step 4: Predict uncertainty
        # Concatenate embeddings and attention weights as context
        uncertainty_input = torch.cat([embeddings, attention_weights], dim=1)
        log_std = self.uncertainty_net(uncertainty_input)
        std = F.softplus(log_std)  # Ensure positive
        
        return mean, std


class DrugDiscoveryDeepEnsembleMVE(nn.Module):
    """
    Deep Ensemble with Mean-Variance Estimation (MVE)
    
    Each model in the ensemble predicts both mean and variance.
    Final prediction uses proper uncertainty aggregation:
    - Mean = average of individual means
    - Variance = average of (variances + squared means) - squared(mean of means)
    
    This captures both aleatoric (from individual variances) and 
    epistemic (from model disagreement) uncertainty.
    
    Reference: Lakshminarayanan et al., "Simple and Scalable Predictive 
    Uncertainty Estimation using Deep Ensembles", NIPS 2017
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryDeepEnsembleMVE, self).__init__()
        
        self.num_models = hyp_params.num_models if hasattr(hyp_params, 'num_models') else 5
        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        
        # Create ensemble of Gaussian models (predict mean + variance)
        self.models = nn.ModuleList([
            DrugDiscoveryGaussianEmb(hyp_params) for _ in range(self.num_models)
        ])
    
    def forward(self, expert_scores, embeddings):
        """
        Forward pass with proper MVE uncertainty aggregation
        
        Args:
            expert_scores: [batch, num_experts]
            embeddings: [batch, embedding_dim]
        
        Returns:
            mean: Aggregated mean [batch, 1]
            std: Total uncertainty (aleatoric + epistemic) [batch, 1]
        """
        means = []
        variances = []
        
        for model in self.models:
            mu, sigma_sq = model(expert_scores, embeddings)  # mu: [batch, 1], sigma_sq: [batch, 1]
            means.append(mu)
            variances.append(sigma_sq)
        
        # Stack predictions
        means_stack = torch.stack(means, dim=0)  # [num_models, batch, 1]
        variances_stack = torch.stack(variances, dim=0)  # [num_models, batch, 1]
        
        # Compute ensemble mean
        mean = means_stack.mean(dim=0)  # [batch, 1]
        
        # Compute ensemble variance using MVE formula:
        # Var[y] = E[Var[y|x]] + Var[E[y|x]]
        # = mean(variances) + mean(means^2) - mean(means)^2
        aleatoric = variances_stack.mean(dim=0)  # Average variance (aleatoric)
        epistemic = (means_stack ** 2).mean(dim=0) - mean ** 2  # Model disagreement (epistemic)
        
        # Total uncertainty
        total_variance = aleatoric + epistemic
        std = torch.sqrt(torch.clamp(total_variance, min=1e-8))
        
        return mean, std


class DrugDiscoverySVGP(nn.Module):
    """
    Sparse Variational Gaussian Process (SVGP) with Deep Kernel Learning

    Combines neural network feature extraction with Gaussian Process for
    principled uncertainty quantification.

    Architecture:
    - Feature Extractor: Deep MLP processes expert scores + embeddings → latent features
    - Gaussian Process: Sparse variational GP layer on top of features for uncertainty estimation
    - Uses inducing points for scalability (O(nm²) instead of O(n³))

    Reference: Wilson et al., "Deep Kernel Learning", AISTATS 2016
               Hensman et al., "Scalable Variational Gaussian Process Classification", AISTATS 2015
    """
    def __init__(self, hyp_params):
        super(DrugDiscoverySVGP, self).__init__()
        
        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        self.feature_dim = 64  # Latent feature dimension for GP
        self.num_inducing = hyp_params.num_inducing if hasattr(hyp_params, 'num_inducing') else 128
        
        # Feature extractor: Maps inputs to latent space
        # NOTE: No dropout here! Dropout causes train/eval distribution mismatch
        # which breaks GP uncertainty estimation. The GP sees different feature
        # distributions at train vs test time, leading to poor calibration.
        # Use weight decay for regularization instead.
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.expert_dim + self.embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim),
        )
        
        # GP parameters
        # Inducing points: learnable "representative" points in feature space
        # Initialize with spread matching typical neural network activations
        self.inducing_points = nn.Parameter(
            torch.randn(self.num_inducing, self.feature_dim) * 0.5
        )

        # GP hyperparameters - better initialization for 64-dim feature space
        # Lengthscale: sqrt(feature_dim) is a reasonable starting point
        init_lengthscale = np.sqrt(self.feature_dim)  # ~8 for 64 dims
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(init_lengthscale)))
        # Output scale: initialize to 1.0
        self.log_outputscale = nn.Parameter(torch.zeros(1))
        # Observation noise: initialize to reasonable value
        self.log_noise = nn.Parameter(torch.log(torch.tensor(0.5)))

        # Inducing point outputs (variational parameters)
        self.inducing_mean = nn.Parameter(torch.zeros(self.num_inducing, 1))
        # Initialize inducing variance to allow uncertainty
        self.inducing_log_var = nn.Parameter(torch.zeros(self.num_inducing, 1))
        
    def rbf_kernel(self, x1, x2):
        """
        RBF (Radial Basis Function) kernel
        k(x1, x2) = outputscale * exp(-||x1 - x2||^2 / (2 * lengthscale^2))
        
        Args:
            x1: [n1, d] tensor
            x2: [n2, d] tensor
            
        Returns:
            kernel: [n1, n2] kernel matrix
        """
        lengthscale = F.softplus(self.log_lengthscale) + 1e-6
        outputscale = F.softplus(self.log_outputscale) + 1e-6
        
        # Compute squared distances
        x1_norm = (x1 ** 2).sum(dim=1, keepdim=True)  # [n1, 1]
        x2_norm = (x2 ** 2).sum(dim=1, keepdim=True)  # [n2, 1]
        
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1^T*x2
        sq_dist = x1_norm + x2_norm.t() - 2 * torch.mm(x1, x2.t())  # [n1, n2]
        sq_dist = torch.clamp(sq_dist, min=0.0)  # Numerical stability
        
        # RBF kernel
        kernel = outputscale * torch.exp(-sq_dist / (2 * lengthscale ** 2))
        
        return kernel
    
    def forward(self, expert_scores, embeddings, compute_loss_terms=False):
        """
        Forward pass with GP prediction
        
        Args:
            expert_scores: [batch, num_experts]
            embeddings: [batch, embedding_dim]
            compute_loss_terms: If True, return KL divergence for training
        
        Returns:
            mean: predictive mean [batch, 1]
            std: predictive std (epistemic + aleatoric) [batch, 1]
            (optional) kl_div: KL divergence for regularization
        """
        # Extract features
        x = torch.cat([expert_scores, embeddings], dim=1)  # [batch, expert_dim + embedding_dim]
        features = self.feature_extractor(x)  # [batch, feature_dim]
        
        # GP prediction using inducing points (sparse approximation)
        # K_mm: kernel between inducing points
        K_mm = self.rbf_kernel(self.inducing_points, self.inducing_points)  # [m, m]
        K_mm = K_mm + torch.eye(self.num_inducing, device=K_mm.device) * 1e-4  # Add jitter
        
        # K_nm: kernel between data points and inducing points
        K_nm = self.rbf_kernel(features, self.inducing_points)  # [batch, m]
        
        # K_nn_diag: diagonal of kernel between data points (for uncertainty)
        K_nn_diag = F.softplus(self.log_outputscale) + 1e-6  # Constant for RBF with same input
        
        # Cholesky decomposition of K_mm
        try:
            L_mm = torch.linalg.cholesky(K_mm)  # [m, m]
        except RuntimeError:
            # If Cholesky fails, add more jitter
            K_mm = K_mm + torch.eye(self.num_inducing, device=K_mm.device) * 1e-3
            L_mm = torch.linalg.cholesky(K_mm)
        
        # Compute alpha = K_mm^-1 @ K_mn for predictions
        K_mn = K_nm.t()  # [m, batch]
        alpha = torch.cholesky_solve(K_mn, L_mm)  # K_mm^-1 @ K_mn: [m, batch]

        # Predictive mean: K_nm @ K_mm^-1 @ inducing_mean = alpha.T @ inducing_mean
        alpha_mean = torch.cholesky_solve(self.inducing_mean, L_mm)  # K_mm^-1 @ m: [m, 1]
        mean = torch.mm(K_nm, alpha_mean)  # [batch, 1]

        # Predictive variance for SVGP:
        # Var[f*] = K** - K*m @ Kmm^-1 @ Km* + K*m @ Kmm^-1 @ S @ Kmm^-1 @ Km*
        #         = K** - alpha.T @ Km* + alpha.T @ S @ alpha
        # where S = diag(inducing_var)

        inducing_var = F.softplus(self.inducing_log_var)  # [m, 1]

        # Term 1: Prior variance K** (diagonal) = outputscale for RBF
        # Term 2: Reduction from conditioning on inducing points
        # K*m @ Kmm^-1 @ Km* = sum over diagonal of (K_nm @ alpha)
        # For diagonal: (K_nm * alpha.T).sum(dim=1) = (K_nm @ Kmm^-1 @ Kmn)_diag
        var_reduction = (K_nm * alpha.t()).sum(dim=1, keepdim=True)  # [batch, 1]

        # Term 3: Variance from inducing point uncertainty
        # K*m @ Kmm^-1 @ S @ Kmm^-1 @ Km* = alpha.T @ S @ alpha (diagonal)
        # For diagonal S: sum_j( alpha[j,:]^2 * S[j,j] ) = (alpha^2).T @ inducing_var
        var_inducing = torch.mm(alpha.t() ** 2, inducing_var)  # [batch, 1]

        # Epistemic variance: K** - K*m @ Kmm^-1 @ Km* + variational term
        var_epistemic = K_nn_diag - var_reduction + var_inducing
        var_epistemic = torch.clamp(var_epistemic, min=1e-6)  # [batch, 1]

        # Observation noise (aleatoric)
        noise = F.softplus(self.log_noise) + 1e-6

        # Total variance
        total_var = var_epistemic + noise
        std = torch.sqrt(total_var)  # [batch, 1]
        
        if compute_loss_terms:
            # KL divergence: KL(q(u) || p(u)) for variational inference
            # q(u) = N(inducing_mean, diag(inducing_var))
            # p(u) = N(0, K_mm)
            #
            # KL = 0.5 * (tr(K_mm^-1 @ S) + m^T @ K_mm^-1 @ m - k + log(det(K_mm)) - log(det(S)))

            K_mm_inv_mean = torch.cholesky_solve(self.inducing_mean, L_mm)  # [m, 1]

            # Trace term: tr(K_mm^-1 @ S) where S = diag(inducing_var)
            K_mm_inv_diag = torch.diagonal(torch.cholesky_solve(
                torch.eye(self.num_inducing, device=K_mm.device), L_mm))  # [m]
            trace_term = (K_mm_inv_diag * inducing_var.squeeze()).sum()

            # Quadratic term: m^T @ K_mm^-1 @ m
            quad_term = (self.inducing_mean.squeeze() * K_mm_inv_mean.squeeze()).sum()

            # Log det of K_mm (from Cholesky: log det = 2 * sum of log diagonal of L)
            log_det_K_mm = 2 * torch.diagonal(L_mm).log().sum()

            # Log det of S = diag(inducing_var) = sum of log(inducing_var)
            # Note: inducing_var = softplus(inducing_log_var), so we need log(softplus(...))
            log_det_Sigma_q = torch.log(inducing_var.squeeze() + 1e-8).sum()

            kl_div = 0.5 * (trace_term + quad_term - self.num_inducing + log_det_K_mm - log_det_Sigma_q)

            return mean, std, kl_div
        
        return mean, std
    
    def update_inducing_points(self, features, targets, max_points=None):
        """
        Update inducing points using k-means clustering on training features
        This is called after training to improve the sparse approximation
        
        Args:
            features: [n, feature_dim] training features
            targets: [n, 1] training targets
            max_points: Maximum number of inducing points (default: self.num_inducing)
        """
        if max_points is None:
            max_points = self.num_inducing
        
        n = features.shape[0]
        if n <= max_points:
            # Use all points as inducing points
            self.inducing_points.data = features[:max_points]
            self.inducing_mean.data = targets[:max_points]
        else:
            # Use k-means to select representative points
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=max_points, random_state=42)
            kmeans.fit(features.cpu().detach().numpy())
            
            # Set inducing points to cluster centers
            centers = torch.tensor(kmeans.cluster_centers_, dtype=features.dtype, device=features.device)
            self.inducing_points.data = centers
            
            # Set inducing means to average targets in each cluster
            labels = kmeans.labels_
            inducing_means = []
            for i in range(max_points):
                cluster_targets = targets[labels == i]
                if len(cluster_targets) > 0:
                    inducing_means.append(cluster_targets.mean())
                else:
                    inducing_means.append(0.0)
            self.inducing_mean.data = torch.tensor(inducing_means, dtype=targets.dtype, device=targets.device).unsqueeze(1)


# ============================================================================
# SWAG (Stochastic Weight Averaging - Gaussian) Model
# ============================================================================

def swag_parameters(module, params, no_cov_mat=True):
    """Helper function to register SWAG parameters for a module."""
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())

        if no_cov_mat is False:
            module.register_buffer(
                "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
            )

        params.append((module, name))


class SWAG(torch.nn.Module):
    """
    SWAG (Stochastic Weight Averaging - Gaussian) wrapper.
    
    Wraps a base model and collects weight statistics during training.
    At inference time, samples from the approximate posterior distribution.
    
    Reference: Maddox et al., "A Simple Baseline for Bayesian Uncertainty 
    in Deep Learning", NeurIPS 2019
    """
    def __init__(
        self, base, no_cov_mat=True, max_num_models=20, var_clamp=1e-30, *args, **kwargs
    ):
        super(SWAG, self).__init__()

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp

        self.base = base(*args, **kwargs)
        self.base.apply(
            lambda module: swag_parameters(
                module=module, params=self.params, no_cov_mat=self.no_cov_mat
            )
        )

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def sample(self, scale=1.0, cov=False, seed=None, block=False, fullrank=True):
        """
        Sample weights from the SWAG posterior distribution.
        
        Args:
            scale: Scale factor for sampling (default: 1.0)
            cov: Whether to use covariance matrix (default: False)
            seed: Random seed for reproducibility
            block: Whether to sample blockwise (default: False)
            fullrank: Whether to use full-rank covariance (default: True)
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not block:
            self.sample_fullrank(scale, cov, fullrank)
        else:
            self.sample_blockwise(scale, cov, fullrank)

    def sample_blockwise(self, scale, cov, fullrank):
        """Sample weights blockwise (per-layer)."""
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)

            sq_mean = module.__getattr__("%s_sq_mean" % name)
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if cov is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                if fullrank:
                    w = mean + scaled_diag_sample + cov_sample
                else:
                    w = mean + scaled_diag_sample

            else:
                w = mean + scaled_diag_sample

            module.__setattr__(name, w)

    def sample_fullrank(self, scale, cov, fullrank):
        """Sample weights using full-rank covariance."""
        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        # Determine device from the mean buffers (parameters were replaced with buffers)
        device = mean_list[0].device if mean_list else torch.device('cpu')

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, sample.to(device))

    def collect_model(self, base_model):
        """Collect a model snapshot for SWAG statistics."""
        for (module, name), base_param in zip(self.params, base_model.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data ** 2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (base_param.data - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with proper initialization of covariance matrices."""
        if not self.no_cov_mat:
            n_models = state_dict["n_models"].item()
            rank = min(n_models, self.max_num_models)
            for module, name in self.params:
                mean = module.__getattr__("%s_mean" % name)
                module.__setattr__(
                    "%s_cov_mat_sqrt" % name,
                    mean.new_empty((rank, mean.numel())).zero_(),
                )
        super(SWAG, self).load_state_dict(state_dict, strict)


class DrugDiscoverySWAG(nn.Module):
    """
    SWAG model for Drug Discovery.

    Uses SWAG to approximate the posterior distribution over model weights.
    Base model is a Gaussian model (predicts mean and variance).

    Training procedure:
    1. Train base_model with standard optimizer using train_forward()
    2. After initial training, periodically call collect_model() during SWA phase
    3. At inference, forward() samples from the approximate posterior

    Note: SWAG requires a separate trainable model because the SWAG wrapper
    removes parameters and replaces them with buffers for statistics tracking.

    Reference: Maddox et al., "A Simple Baseline for Bayesian Inference in Deep Learning", NeurIPS 2019
    """
    def __init__(self, hyp_params):
        super(DrugDiscoverySWAG, self).__init__()

        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        self.max_num_models = hyp_params.max_num_models if hasattr(hyp_params, 'max_num_models') else 20
        self.no_cov_mat = hyp_params.no_cov_mat if hasattr(hyp_params, 'no_cov_mat') else True
        self.num_samples = hyp_params.num_swag_samples if hasattr(hyp_params, 'num_swag_samples') else 30

        # Store hyp_params for creating base model
        self._hyp_params = hyp_params

        # Trainable base model - this is what we actually train
        # Keep this separate because SWAG wrapper removes parameters
        self.base_model = DrugDiscoveryGaussianEmb(hyp_params)

        # SWAG wrapper for collecting statistics and sampling
        # Note: SWAG removes parameters from its internal base and replaces with buffers
        self.swag = SWAG(
            base=DrugDiscoveryGaussianEmb,
            no_cov_mat=self.no_cov_mat,
            max_num_models=self.max_num_models,
            hyp_params=hyp_params
        )

    def get_base_model(self):
        """
        Get the trainable base model.

        Use this model's parameters for the optimizer.
        Train using train_forward(), then call collect_model() periodically.
        """
        return self.base_model

    def forward(self, expert_scores, embeddings, num_samples=None):
        """
        Forward pass with SWAG sampling.

        Args:
            expert_scores: [batch, num_experts]
            embeddings: [batch, embedding_dim]
            num_samples: Number of SWAG samples (default: self.num_samples)

        Returns:
            mean: mean prediction [batch, 1]
            std: total uncertainty (aleatoric + epistemic) [batch, 1]
        """
        # If no models collected yet, use base model directly (pre-SWA phase)
        if self.swag.n_models.item() == 0:
            mu, sigma_sq = self.base_model(expert_scores, embeddings)
            return mu, torch.sqrt(sigma_sq + 1e-8)

        if num_samples is None:
            num_samples = self.num_samples

        means = []
        variances = []

        for _ in range(num_samples):
            # Sample weights from SWAG posterior
            self.swag.sample(scale=1.0, cov=not self.no_cov_mat, fullrank=True)

            # Get prediction with sampled weights (mu and aleatoric variance)
            mu, sigma_sq = self.swag(expert_scores, embeddings)
            means.append(mu)
            variances.append(sigma_sq)

        # Stack predictions
        means_stack = torch.stack(means, dim=0)  # [num_samples, batch, 1]
        variances_stack = torch.stack(variances, dim=0)  # [num_samples, batch, 1]

        # Compute ensemble mean
        mean = means_stack.mean(dim=0)  # [batch, 1]

        # Compute total uncertainty using law of total variance:
        # Var[Y] = E[Var[Y|θ]] + Var[E[Y|θ]]
        # aleatoric = E[σ²(θ)] - average predicted variance across weight samples
        # epistemic = Var[μ(θ)] - variance of predicted means across weight samples
        aleatoric = variances_stack.mean(dim=0)  # [batch, 1]
        epistemic = means_stack.var(dim=0)  # [batch, 1]

        # Total uncertainty
        total_variance = aleatoric + epistemic
        std = torch.sqrt(torch.clamp(total_variance, min=1e-8))

        return mean, std

    def collect_model(self):
        """
        Collect current base_model weights for SWAG statistics.

        Call this periodically during the SWA phase of training
        (after initial convergence, with cyclical or constant learning rate).
        """
        self.swag.collect_model(self.base_model)

    def train_forward(self, expert_scores, embeddings):
        """
        Forward pass for training (no sampling, uses trainable base_model).

        Use this during the training phase before collecting SWAG statistics.

        Returns:
            mu: predicted mean [batch, 1]
            sigma_sq: predicted variance [batch, 1]
        """
        return self.base_model(expert_scores, embeddings)
