import torch
from torch import nn
import torch.nn.functional as F


class DrugDiscoveryMoNIGEmb(nn.Module):
    """
    Drug Discovery model using Mixture of Normal-Inverse Gamma (MoNIG) with embeddings
    
    Architecture:
    - Input: Expert Scores + Molecular Embeddings (704-dim)
    - Per-expert: Calibrator → Context-Aware Evidential Head → NIG parameters
    - MoNIG Aggregator: Combine all expert NIGs
    """
    def __init__(self, hyp_params):
        super(DrugDiscoveryMoNIGEmb, self).__init__()
        
        self.num_experts = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = hyp_params.hidden_dim if hasattr(hyp_params, 'hidden_dim') else 256
        self.dropout = hyp_params.dropout if hasattr(hyp_params, 'dropout') else 0.2
        
        # Embedding encoder (reduce dimension from 704 to manageable size)
        self.embedding_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
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
        
        # Per-expert evidential heads
        self.evidential_heads = nn.ModuleList([
            self._build_evidential_head() for _ in range(self.num_experts)
        ])
    
    def _build_evidential_head(self):
        """
        Context-aware evidential head that produces NIG parameters
        Input: embedding context (64) + calibrated expert score (1)
        Output: μ, ν, α, β (NIG parameters)
        """
        return nn.ModuleDict({
            'fusion': nn.Sequential(
                nn.Linear(64 + 1, self.hidden_dim),  # Embedding context (64) + Score (1)
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
            embeddings: [batch, embedding_dim] - Molecular/protein embeddings
        
        Returns:
            List of (mu, v, alpha, beta) tuples, one per expert
        """
        batch_size = expert_scores.shape[0]
        
        # Encode embeddings to lower dimension
        emb_encoded = self.embedding_encoder(embeddings)  # [batch, 64]
        
        # Process each expert
        nigs = []
        for i in range(self.num_experts):
            # Step 1: Calibrate expert score
            expert_i = expert_scores[:, i:i+1]  # [batch, 1]
            calibrated = self.calibrators[i](expert_i)  # [batch, 1]
            
            # Step 2: Fuse calibrated score with embedding context
            head = self.evidential_heads[i]
            fused = head['fusion'](torch.cat([emb_encoded, calibrated], dim=1))  # [batch, 64]
            
            # Step 3: Predict NIG parameters
            mu = head['mu_head'](fused)          # [batch, 1]
            logv = head['v_head'](fused)         # [batch, 1]
            logalpha = head['alpha_head'](fused) # [batch, 1]
            logbeta = head['beta_head'](fused)   # [batch, 1]
            
            # Step 4: Apply evidence function to ensure proper parameter ranges
            v = self.evidence(logv)              # ν > 0
            alpha = self.evidence(logalpha) + 1  # α > 1
            beta = self.evidence(logbeta)        # β > 0
            
            nigs.append((mu, v, alpha, beta))
        
        return nigs


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
