import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugDiscoveryMoNIGEmb(nn.Module):
    """
    Mixture-of-NIG model with expert scores + molecular embeddings.

    Key points:
    - Embeddings are pre-normalized in the dataset.
    - Expert scores are z-scored inside the model using dataset-level stats
      (mean/std per expert passed via hyp_params.score_mean / score_std).
    """

    def __init__(self, hyp_params):
        super().__init__()

        self.num_experts = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = getattr(hyp_params, "hidden_dim", 256)
        self.dropout = getattr(hyp_params, "dropout", 0.2)

        # ---- NIG parameter lower bounds to avoid degeneracy ----
        self.v_min = 1e-4
        self.alpha_min = 1e-4
        self.beta_min = 1e-4

        # Expert dropout probability
        self.expert_dropout_p = getattr(hyp_params, "expert_dropout_p", 0.2)
        
        # Expert balance normalization: normalize expert precisions to prevent dominance
        self.use_expert_normalization = getattr(hyp_params, "use_expert_normalization", True)
        self.expert_normalization_strength = getattr(hyp_params, "expert_normalization_strength", 0.5)

        # ---- Dataset-level score normalization stats (buffers) ----
        if hasattr(hyp_params, "score_mean") and hasattr(hyp_params, "score_std"):
            score_mean = torch.as_tensor(
                hyp_params.score_mean, dtype=torch.float32
            ).view(-1)
            score_std = torch.as_tensor(
                hyp_params.score_std, dtype=torch.float32
            ).view(-1)
            if score_mean.numel() != self.num_experts or score_std.numel() != self.num_experts:
                raise ValueError(
                    f"score_mean/std must have length {self.num_experts}, "
                    f"got {score_mean.numel()} / {score_std.numel()}"
                )
        else:
            score_mean = torch.zeros(self.num_experts, dtype=torch.float32)
            score_std = torch.ones(self.num_experts, dtype=torch.float32)

        self.register_buffer("score_mean", score_mean)
        self.register_buffer("score_std", score_std)

        # ---- Shared embedding MLP ----
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # ---- Per-expert embedding projection ----
        self.embedding_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                )
                for _ in range(self.num_experts)
            ]
        )

        # ---- Per-expert score towers (z-scored scores in, richer MLP) ----
        score_hidden = max(self.hidden_dim // 2, 32)
        self.score_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, score_hidden),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(score_hidden, score_hidden),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(score_hidden, score_hidden),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                )
                for _ in range(self.num_experts)
            ]
        )

        # ---- Gating MLPs to scale precision parameters (v, alpha) ----
        gate_hidden = max(self.hidden_dim // 4, 16)
        self.gating_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim + score_hidden, gate_hidden),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(gate_hidden, 2),
                    nn.Sigmoid(),  # outputs in (0, 1)
                )
                for _ in range(self.num_experts)
            ]
        )

        # ---- NIG heads per expert ----
        head_input_dim = self.hidden_dim + score_hidden
        head_hidden = self.hidden_dim
        self.nig_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_input_dim, head_hidden),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(head_hidden, 4),  # mu, log_v, log_alpha_shift, log_beta
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, expert_scores, embeddings):
        """
        expert_scores: [B, E] raw scores
        embeddings:    [B, D] pre-normalized embeddings

        Returns:
            list of (mu, v, alpha, beta) for each expert, each of shape [B, 1]
        """
        B, E = expert_scores.shape
        if E != self.num_experts:
            raise ValueError(
                f"Expected expert_scores with {self.num_experts} experts, got {E}"
            )

        shared_emb = self.embedding_mlp(embeddings)  # [B, H]

        nigs = []

        for k in range(self.num_experts):
            # Select scalar score for expert k: [B, 1]
            score_k = expert_scores[:, k : k + 1]

            # ---- Z-score normalization using dataset-level stats ----
            mean_k = self.score_mean[k].view(1, 1)
            std_k = self.score_std[k].view(1, 1)
            score_k = (score_k - mean_k) / (std_k + 1e-6)

            # ---- Expert dropout ----
            if self.training and self.expert_dropout_p > 0.0:
                mask = (torch.rand_like(score_k) > self.expert_dropout_p).float()
                score_k = score_k * mask

            # ---- Per-expert embedding projection + score tower ----
            emb_k = self.embedding_projections[k](shared_emb)  # [B, H]
            score_feat = self.score_mlps[k](score_k)  # [B, score_hidden]

            fused = torch.cat([emb_k, score_feat], dim=-1)  # [B, H + score_hidden]

            # Gating to scale v, alpha
            gate_out = self.gating_networks[k](fused)  # [B, 2]
            gate_v = gate_out[:, 0:1]
            gate_alpha = gate_out[:, 1:2]

            # NIG params
            out = self.nig_heads[k](fused)  # [B, 4]
            mu, log_v, log_alpha_shift, log_beta = torch.split(out, 1, dim=-1)

            v_raw = F.softplus(log_v) + self.v_min
            alpha_raw = F.softplus(log_alpha_shift) + 1.0 + self.alpha_min
            beta = F.softplus(log_beta) + self.beta_min

            # Scale v and alpha by gate factors (more constrained ranges to prevent dominance)
            # Constrain v scaling to 0.2–1.5 (reduced from 0.1–2.0) to prevent single expert dominance
            v = v_raw * (0.2 + 1.3 * gate_v)
            # Constrain alpha scaling to 0.6–1.8 (reduced from 0.5–2.0)
            alpha = alpha_raw * (0.6 + 1.2 * gate_alpha)

            nigs.append((mu, v, alpha, beta))

        # Normalize expert precisions to prevent dominance
        if self.use_expert_normalization and len(nigs) > 1:
            # Extract all v values
            vs = torch.stack([v for _, v, _, _ in nigs], dim=0)  # [E, B, 1]
            vs_mean = vs.mean(dim=0, keepdim=True)  # [1, B, 1]
            vs_max = vs.max(dim=0, keepdim=True)[0]
            vs_min = vs.min(dim=0, keepdim=True)[0]
            
            # Very aggressive normalization: compress differences toward mean
            # Use stronger compression to prevent any expert from dominating
            compression = self.expert_normalization_strength
            vs_normalized = vs_mean + (vs - vs_mean) * (1.0 - compression)
            
            # Additional: cap max/min ratio to prevent extreme dominance
            # If max/min ratio > 2.0, compress further
            max_min_ratio = (vs_max / (vs_min + 1e-6))
            ratio_mask = (max_min_ratio > 2.0).float()
            # Extra compression when ratio is too high
            extra_compression = ratio_mask * 0.3  # Additional 30% compression
            vs_normalized = vs_mean + (vs_normalized - vs_mean) * (1.0 - extra_compression)
            
            vs_normalized = torch.clamp(vs_normalized, min=1e-6)
            
            # Replace v values with normalized ones
            normalized_nigs = []
            for i, (mu, _, alpha, beta) in enumerate(nigs):
                normalized_nigs.append((mu, vs_normalized[i], alpha, beta))
            nigs = normalized_nigs

        return nigs


class DrugDiscoveryNIGEmb(nn.Module):
    """
    Single NIG model (no mixture) with expert scores + embeddings.
    """

    def __init__(self, hyp_params):
        super().__init__()

        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = getattr(hyp_params, "hidden_dim", 256)
        self.dropout = getattr(hyp_params, "dropout", 0.2)

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

        self.mu_head = nn.Linear(64, 1)
        self.v_head = nn.Linear(64, 1)
        self.alpha_head = nn.Linear(64, 1)
        self.beta_head = nn.Linear(64, 1)

    @staticmethod
    def evidence(x):
        return F.softplus(x)

    def forward(self, expert_scores, embeddings):
        x = torch.cat([expert_scores, embeddings], dim=1)
        h = self.encoder(x)

        mu = self.mu_head(h)
        logv = self.v_head(h)
        logalpha = self.alpha_head(h)
        logbeta = self.beta_head(h)

        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1.0
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta


class DrugDiscoveryGaussianEmb(nn.Module):
    """
    Gaussian model: predict mean + variance (σ²) from scores + embeddings.
    """

    def __init__(self, hyp_params):
        super().__init__()

        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = getattr(hyp_params, "hidden_dim", 256)
        self.dropout = getattr(hyp_params, "dropout", 0.2)

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

        self.mu_head = nn.Linear(64, 1)
        self.sigma_head = nn.Linear(64, 1)

    @staticmethod
    def evidence(x):
        return F.softplus(x)

    def forward(self, expert_scores, embeddings):
        x = torch.cat([expert_scores, embeddings], dim=1)
        h = self.encoder(x)
        mu = self.mu_head(h)
        log_sigma = self.sigma_head(h)
        sigma = self.evidence(log_sigma)
        return mu, sigma


class DrugDiscoveryBaselineEmb(nn.Module):
    """
    Simple MLP baseline: scores + embeddings -> scalar regression.
    """

    def __init__(self, hyp_params):
        super().__init__()

        self.expert_dim = hyp_params.num_experts
        self.embedding_dim = hyp_params.embedding_dim
        self.hidden_dim = getattr(hyp_params, "hidden_dim", 256)
        self.dropout = getattr(hyp_params, "dropout", 0.2)

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
            nn.Linear(64, 1),
        )

    def forward(self, expert_scores, embeddings):
        x = torch.cat([expert_scores, embeddings], dim=1)
        return self.model(x)