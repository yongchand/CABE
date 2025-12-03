import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
        # Per-expert calibrators (adjust expert scores)
        self.calibrators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(self.num_experts)
        ])
        
        # Per-expert evidential heads (produce NIG params from calibrated scores)
        self.evidential_heads = nn.ModuleList([
            self._build_evidential_head() for _ in range(self.num_experts)
        ])
        
        # Reliability Net: Embeddings → reliability scores r_j ∈ (0,1) per engine
        self.reliability_net = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_experts),  # Output one reliability score per expert
            nn.Sigmoid()  # Ensure r_j ∈ (0,1)
        )
    
    def _build_evidential_head(self):
        """
        Evidential head that produces NIG parameters from calibrated expert score
        Input: calibrated expert score (1)
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
            # Calibrate expert score
            expert_i = expert_scores[:, i:i+1]  # [batch, 1]
            calibrated = self.calibrators[i](expert_i)  # [batch, 1]
            
            # Process through evidential head
            head = self.evidential_heads[i]
            fused = head['fusion'](calibrated)  # [batch, 64]
            
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
            calibrated = self.calibrators[i](expert_i)
            head = self.evidential_heads[i]
            fused = head['fusion'](calibrated)
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
            calibrated = self.calibrators[i](expert_i)
            head = self.evidential_heads[i]
            fused = head['fusion'](calibrated)
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


class DrugDiscoveryMoNIG_FixedReliability(DrugDiscoveryMoNIGEmb):
    """
    Ablation 3: Use fixed reliability values (r_j = fixed_r for all experts)
    This tests if context-dependent reliability matters.
    """
    def __init__(self, hyp_params, fixed_r=0.5):
        super().__init__(hyp_params)
        self.fixed_r = fixed_r
    
    def forward(self, expert_scores, embeddings):
        # Get per-expert NIG params (same as base)
        nigs = []
        for i in range(self.num_experts):
            expert_i = expert_scores[:, i:i+1]
            calibrated = self.calibrators[i](expert_i)
            head = self.evidential_heads[i]
            fused = head['fusion'](calibrated)
            mu = head['mu_head'](fused)
            logv = head['v_head'](fused)
            logalpha = head['alpha_head'](fused)
            logbeta = head['beta_head'](fused)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
            nigs.append((mu, v, alpha, beta))
        
        # ABLATION: Use fixed reliability (ignore embeddings)
        batch_size = expert_scores.shape[0]
        reliability_scores = torch.full(
            (batch_size, self.num_experts), 
            self.fixed_r, 
            device=expert_scores.device
        )
        
        # Apply scaling with fixed reliability
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
    Ablation 4: Use uniform weight aggregation instead of MoNIG aggregation
    This tests if MoNIG-style aggregation is necessary vs simple averaging.
    Note: This still uses reliability scaling, but aggregates with uniform weights.
    """
    def forward(self, expert_scores, embeddings):
        # Get per-expert NIG params with reliability scaling (same as base)
        nigs = []
        for i in range(self.num_experts):
            expert_i = expert_scores[:, i:i+1]
            calibrated = self.calibrators[i](expert_i)
            head = self.evidential_heads[i]
            fused = head['fusion'](calibrated)
            mu = head['mu_head'](fused)
            logv = head['v_head'](fused)
            logalpha = head['alpha_head'](fused)
            logbeta = head['beta_head'](fused)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
            nigs.append((mu, v, alpha, beta))
        
        # Apply reliability scaling (keep this)
        reliability_scores = self.reliability_net(embeddings)
        scaled_nigs = []
        for i in range(self.num_experts):
            mu, v, alpha, beta = nigs[i]
            r_j = reliability_scores[:, i:i+1]
            v_scaled = v * r_j
            alpha_scaled = 1.0 + (alpha - 1.0) * r_j + 1e-6
            beta_scaled = beta * r_j
            scaled_nigs.append((mu, v_scaled, alpha_scaled, beta_scaled))
        
        # ABLATION: Use uniform weight aggregation instead of MoNIG
        # Simple average of means and uncertainty parameters
        mus = torch.stack([nig[0] for nig in scaled_nigs], dim=0)  # [num_experts, batch, 1]
        vs = torch.stack([nig[1] for nig in scaled_nigs], dim=0)
        alphas = torch.stack([nig[2] for nig in scaled_nigs], dim=0)
        betas = torch.stack([nig[3] for nig in scaled_nigs], dim=0)
        
        mu_avg = mus.mean(dim=0)  # [batch, 1]
        v_avg = vs.mean(dim=0)
        alpha_avg = alphas.mean(dim=0)
        beta_avg = betas.mean(dim=0)
        
        # Return as list for compatibility (but only one element)
        return [(mu_avg, v_avg, alpha_avg, beta_avg)]


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
    Monte Carlo Dropout: Uses dropout at inference time for uncertainty estimation
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryMCDropout, self).__init__()
        
        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        self.num_samples = hyp_params.num_mc_samples if hasattr(hyp_params, 'num_mc_samples') else 50
        
        # Use same architecture as Baseline but with dropout enabled at inference
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
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, expert_scores, embeddings, num_samples=None):
        """
        Forward pass with Monte Carlo sampling
        
        Args:
            expert_scores: [batch, num_experts]
            embeddings: [batch, embedding_dim]
            num_samples: Number of MC samples (default: self.num_samples)
        
        Returns:
            mean: mean prediction [batch, 1]
            std: standard deviation [batch, 1]
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        # Enable dropout for MC sampling
        self.train()
        
        x = torch.cat([expert_scores, embeddings], dim=1)
        predictions = []
        
        for _ in range(num_samples):
            pred = self.model(x)  # [batch, 1]
            predictions.append(pred)
        
        # Stack and compute statistics
        pred_stack = torch.stack(predictions, dim=0)  # [num_samples, batch, 1]
        mean = pred_stack.mean(dim=0)  # [batch, 1]
        std = pred_stack.std(dim=0)  # [batch, 1]
        
        return mean, std


class DrugDiscoveryRandomForest:
    """
    Random Forest Regressor wrapper for compatibility with training/inference pipeline
    Note: This is not a PyTorch module, but a sklearn wrapper
    """
    def __init__(self, hyp_params):
        self.num_experts = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.n_estimators = hyp_params.n_estimators if hasattr(hyp_params, 'n_estimators') else 100
        self.max_depth = hyp_params.max_depth if hasattr(hyp_params, 'max_depth') else None
        self.min_samples_split = hyp_params.min_samples_split if hasattr(hyp_params, 'min_samples_split') else 2
        self.min_samples_leaf = hyp_params.min_samples_leaf if hasattr(hyp_params, 'min_samples_leaf') else 1
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=42
        )
        self.is_trained = False
    
    def to(self, device):
        """Compatibility method - RF doesn't use GPU"""
        return self
    
    def train(self):
        """Compatibility method"""
        return self
    
    def eval(self):
        """Compatibility method"""
        return self
    
    def parameters(self):
        """Compatibility method - return empty iterator"""
        return iter([])
    
    def state_dict(self):
        """Return model state for saving"""
        return {
            'model': self.model,
            'is_trained': self.is_trained
        }
    
    def load_state_dict(self, state_dict):
        """Load model state"""
        self.model = state_dict['model']
        self.is_trained = state_dict.get('is_trained', False)
    
    def fit(self, X, y):
        """Train the random forest"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X, return_std=True):
        """
        Predict with uncertainty estimation
        
        Args:
            X: Input features [n_samples, n_features]
            return_std: Whether to return standard deviation
        
        Returns:
            mean: Mean prediction [n_samples]
            std: Standard deviation [n_samples] (if return_std=True)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Compute mean and std across trees
        mean = predictions.mean(axis=0)
        
        if return_std:
            std = predictions.std(axis=0)
            return mean, std
        else:
            return mean
    
    def forward(self, expert_scores, embeddings):
        """
        Forward pass compatible with PyTorch interface
        
        Args:
            expert_scores: torch.Tensor [batch, num_experts]
            embeddings: torch.Tensor [batch, embedding_dim]
        
        Returns:
            mean: torch.Tensor [batch, 1]
            std: torch.Tensor [batch, 1]
        """
        # Convert to numpy
        expert_scores_np = expert_scores.detach().cpu().numpy()
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Concatenate features
        X = np.concatenate([expert_scores_np, embeddings_np], axis=1)
        
        # Predict
        mean, std = self.predict(X, return_std=True)
        
        # Convert back to torch tensors
        mean_tensor = torch.from_numpy(mean).float().unsqueeze(1)
        std_tensor = torch.from_numpy(std).float().unsqueeze(1)
        
        return mean_tensor, std_tensor
    
    def __call__(self, expert_scores, embeddings):
        """Make the object callable like a PyTorch module"""
        return self.forward(expert_scores, embeddings)
