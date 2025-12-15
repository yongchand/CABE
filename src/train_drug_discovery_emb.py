import os
import sys
import argparse
import pickle
import random

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import uncertainty_toolbox as uct

from src.drug_dataset_emb import DrugDiscoveryDatasetEmb
from src.drug_models_emb import (
    DrugDiscoveryMoNIGEmb, DrugDiscoveryNIGEmb, DrugDiscoveryGaussianEmb, 
    DrugDiscoveryBaselineEmb, DrugDiscoveryDeepEnsemble, DrugDiscoveryMCDropout,
    DrugDiscoveryMoNIG_NoReliabilityScaling, DrugDiscoveryMoNIG_UniformReliability,
    DrugDiscoveryMoNIG_NoContextReliability, DrugDiscoveryMoNIG_UniformWeightAggregation,
    DrugDiscoverySoftmaxMoE, 
    DrugDiscoveryDeepEnsembleMVE,
    DrugDiscoveryCFGP, DrugDiscoverySWAG
)
from src.utils import moe_nig, criterion_nig as criterion_nig_original


def aggregate_nigs(nigs):
    """
    Aggregate multiple NIGs using moe_nig from utils.py (Equation 9)
    Directly uses the original implementation without rewriting
    """
    if len(nigs) == 0:
        raise ValueError("Cannot aggregate empty list of NIGs")
    if len(nigs) == 1:
        return nigs[0]
    
    mu_final, v_final, alpha_final, beta_final = nigs[0]
    for mu, v, alpha, beta in nigs[1:]:
        # Direct call to original moe_nig from utils.py
        mu_final, v_final, alpha_final, beta_final = moe_nig(
            mu_final, v_final, alpha_final, beta_final,
            mu, v, alpha, beta
        )
    return mu_final, v_final, alpha_final, beta_final


def criterion_nig(u, la, alpha, beta, y, risk_weight=0.01):
    """
    Thin wrapper for criterion_nig from utils.py
    Only adapts the interface (risk_weight -> hyp_params.risk)
    """
    class HypParams:
        pass
    hyp = HypParams()
    hyp.risk = risk_weight
    # Direct call to original criterion_nig from utils.py
    return criterion_nig_original(u, la, alpha, beta, y, hyp)


def nig_uncertainty(v, alpha, beta):
    """
    Compute epistemic and aleatoric uncertainty from NIG parameters
    These formulas are derived from NIG distribution properties
    """
    aleatoric = beta / (alpha - 1)
    epistemic = beta / (v * (alpha - 1))
    return epistemic, aleatoric


def compute_conformal_quantile(model, loader, device, model_type, expert_indices=None, coverage=0.95):
    """
    Compute conformal prediction quantile from calibration set.
    
    Uses normalized residuals: |y - y_pred| / uncertainty for adaptive intervals,
    or absolute residuals |y - y_pred| for fixed-width intervals.
    
    Args:
        model: Trained model
        loader: DataLoader for calibration set (typically validation set)
        device: Device to run on
        model_type: Model type ('MoNIG', 'NIG', 'Gaussian', etc.)
        expert_indices: Optional expert indices for ablation
        coverage: Target coverage probability (default: 0.95)
    
    Returns:
        quantile: Quantile value for conformal intervals
    """
    model.eval()
    residuals = []
    eps = 1e-8
    
    with torch.no_grad():
        for inputs, labels, _ in loader:
            expert_scores, embeddings = inputs
            if expert_indices is not None and len(expert_indices) < expert_scores.shape[1]:
                expert_scores = expert_scores[:, expert_indices]
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            if model_type == 'MoNIG' or model_type.startswith('MoNIG_'):
                nigs = model(expert_scores, embeddings)
                if model_type == 'MoNIG_UniformWeightAggregation':
                    mu, v, alpha, beta = nigs[0]
                else:
                    mu, v, alpha, beta = aggregate_nigs(nigs)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                total_std = torch.sqrt(torch.clamp(epistemic + aleatoric, min=eps))
            elif model_type == 'NIG':
                mu, v, alpha, beta = model(expert_scores, embeddings)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                total_std = torch.sqrt(torch.clamp(epistemic + aleatoric, min=eps))
            elif model_type in ['Gaussian', 'DeepEnsemble', 'MCDropout', 'SoftmaxMoE', 'DeepEnsembleMVE', 'CFGP', 'SWAG']:
                mu, std = model(expert_scores, embeddings)
                total_std = std
            else:  # Baseline
                mu = model(expert_scores, embeddings)
                total_std = torch.ones_like(mu) * eps  # Use fixed-width intervals
            
            # Compute normalized residuals
            abs_residual = torch.abs(labels - mu)
            normalized_residual = abs_residual / (total_std + eps)
            residuals.extend(normalized_residual.cpu().numpy().flatten())
    
    if len(residuals) == 0:
        return 0.0
    
    residuals = np.array(residuals)
    # Compute quantile: (n+1) * (1 - alpha) / n for finite-sample correction
    n = len(residuals)
    quantile_idx = int(np.ceil((n + 1) * (1 - (1 - coverage) / 2))) - 1
    quantile_idx = min(quantile_idx, n - 1)
    quantile = np.sort(residuals)[quantile_idx]
    
    return float(quantile)


def get_conformal_intervals(y_pred, y_std, quantile):
    """
    Compute conformal prediction intervals.
    
    Args:
        y_pred: Predictions [n_samples] or [n_samples, 1]
        y_std: Standard deviations [n_samples] or [n_samples, 1]
        quantile: Conformal quantile from calibration set
    
    Returns:
        lower: Lower bounds [n_samples]
        upper: Upper bounds [n_samples]
    """
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.asarray(y_std).flatten()
    eps = 1e-8
    
    # Conformal interval: [y_pred - quantile * y_std, y_pred + quantile * y_std]
    width = quantile * (y_std + eps)
    lower = y_pred - width
    upper = y_pred + width
    
    return lower, upper


def criterion_gaussian(mu, sigma, y):
    """
    Gaussian negative log-likelihood loss
    
    Args:
        mu: predicted mean [batch, 1]
        sigma: predicted variance (σ²) [batch, 1]
        y: true values [batch, 1]
    
    Returns:
        NLL loss
    """
    loss = torch.mean(0.5 * (torch.log(2 * np.pi * sigma) + ((y - mu) ** 2) / sigma))
    return loss




def evaluate_predictions(mu, y_true):
    """
    Compute standard regression metrics
    Could potentially use eval_metrics.py, but this is simpler for regression
    """
    mu = mu.detach().cpu().numpy().flatten()
    y_true = y_true.detach().cpu().numpy().flatten()
    
    mae = np.mean(np.abs(mu - y_true))
    rmse = np.sqrt(np.mean((mu - y_true) ** 2))
    corr = np.corrcoef(mu, y_true)[0, 1]
    
    ss_res = np.sum((y_true - mu) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'mae': mae, 'rmse': rmse, 'corr': corr, 'r2': r2}


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, device, model_type, risk_weight, expert_indices=None):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, labels, _) in enumerate(loader):
        expert_scores, embeddings = inputs
        
        # Select specific experts if specified
        if expert_indices is not None and len(expert_indices) < 2:
            expert_scores = expert_scores[:, expert_indices]
        
        # Move to device
        expert_scores = expert_scores.to(device)
        embeddings = embeddings.to(device)
        labels = labels.to(device).unsqueeze(1)  # [batch, 1]
        
        optimizer.zero_grad()
        
        if model_type == 'MoNIG' or model_type.startswith('MoNIG_'):
            # Get per-expert NIGs
            nigs = model(expert_scores, embeddings)
            
            # For UniformWeightAggregation, nigs is already aggregated (single element)
            if model_type == 'MoNIG_UniformWeightAggregation':
                # Already aggregated, use directly
                mu_final, v_final, alpha_final, beta_final = nigs[0]
                loss = criterion_nig(mu_final, v_final, alpha_final, beta_final, labels, risk_weight)
            else:
                # Aggregate NIGs
                mu_final, v_final, alpha_final, beta_final = aggregate_nigs(nigs)
                
                # Compute loss for each expert + aggregated
                loss = 0
                for mu, v, alpha, beta in nigs:
                    loss += criterion_nig(mu, v, alpha, beta, labels, risk_weight)
                loss += criterion_nig(mu_final, v_final, alpha_final, beta_final, labels, risk_weight)
                loss = loss / (len(nigs) + 1)  # Average
            
        elif model_type == 'NIG':
            # Single NIG
            mu, v, alpha, beta = model(expert_scores, embeddings)
            loss = criterion_nig(mu, v, alpha, beta, labels, risk_weight)
            
        elif model_type == 'Gaussian':
            # Gaussian with uncertainty
            mu, sigma = model(expert_scores, embeddings)
            loss = criterion_gaussian(mu, sigma, labels)
            
        elif model_type == 'DeepEnsemble':
            # Deep Ensemble: train each model separately
            loss = 0
            for ensemble_model in model.models:
                predictions = ensemble_model(expert_scores, embeddings)
                loss += torch.nn.functional.mse_loss(predictions, labels)
            loss = loss / len(model.models)
            
        elif model_type == 'MCDropout':
            # MC Dropout: single prediction (dropout handled in forward)
            # Concatenate inputs for the Sequential model
            x = torch.cat([expert_scores, embeddings], dim=1)
            predictions = model.model(x)
            loss = torch.nn.functional.mse_loss(predictions, labels)
        
        elif model_type == 'SWAG':
            # SWAG: train base model normally, collect snapshots later
            # Use the base model inside SWAG for training
            mu, sigma = model.base_model(expert_scores, embeddings)
            loss = criterion_gaussian(mu, sigma, labels)
            
        elif model_type == 'SoftmaxMoE':
            # Softmax MoE: mean and std prediction
            mu, std = model(expert_scores, embeddings)
            # Use Gaussian NLL loss
            variance = std ** 2
            loss = criterion_gaussian(mu, variance, labels)
            
        elif model_type == 'DeepEnsembleMVE':
            # Deep Ensemble with MVE
            mu, std = model(expert_scores, embeddings)
            # Use Gaussian NLL loss
            variance = std ** 2
            loss = criterion_gaussian(mu, variance, labels)
            
        elif model_type == 'CFGP':
            # Convolutional-Fed Gaussian Process
            mu, std, kl_div = model(expert_scores, embeddings, compute_loss_terms=True)
            # Negative log-likelihood loss
            nll_loss = criterion_gaussian(mu, std ** 2, labels)
            # KL divergence for variational GP (regularization)
            kl_weight = 0.01  # Weight for KL term
            loss = nll_loss + kl_weight * kl_div / len(loader.dataset)  # Scale KL by dataset size
            
        else:  # Baseline
            predictions = model(expert_scores, embeddings)
            loss = torch.nn.functional.mse_loss(predictions, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device, model_type, expert_indices=None):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_epistemic = []
    all_aleatoric = []
    all_variance = []
    
    with torch.no_grad():
        for inputs, labels, _ in loader:
            expert_scores, embeddings = inputs
            
            # Select specific experts if specified
            if expert_indices is not None and len(expert_indices) < 2:
                expert_scores = expert_scores[:, expert_indices]
            
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            if model_type == 'MoNIG' or model_type.startswith('MoNIG_'):
                nigs = model(expert_scores, embeddings)
                if model_type == 'MoNIG_UniformWeightAggregation':
                    mu, v, alpha, beta = nigs[0]
                else:
                    mu, v, alpha, beta = aggregate_nigs(nigs)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                all_epistemic.extend(epistemic.cpu().numpy().flatten())
                all_aleatoric.extend(aleatoric.cpu().numpy().flatten())
                
            elif model_type == 'NIG':
                mu, v, alpha, beta = model(expert_scores, embeddings)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                all_epistemic.extend(epistemic.cpu().numpy().flatten())
                all_aleatoric.extend(aleatoric.cpu().numpy().flatten())
                
            elif model_type == 'Gaussian':
                mu, sigma = model(expert_scores, embeddings)
                all_variance.extend(sigma.cpu().numpy().flatten())
                
            elif model_type == 'DeepEnsemble':
                mu, std = model(expert_scores, embeddings)
                all_variance.extend((std ** 2).cpu().numpy().flatten())
                
            elif model_type == 'MCDropout':
                mu, std = model(expert_scores, embeddings)
                all_variance.extend((std ** 2).cpu().numpy().flatten())
                
            elif model_type in ['SoftmaxMoE', 'DeepEnsembleMVE', 'CFGP', 'SWAG']:
                mu, std = model(expert_scores, embeddings)
                all_variance.extend((std ** 2).cpu().numpy().flatten())
                
            else:  # Baseline
                mu = model(expert_scores, embeddings)
            
            all_preds.append(mu)
            all_labels.append(labels)
    
    # Handle empty loader case
    if len(all_preds) == 0:
        print("Warning: Empty dataset, returning default metrics")
        metrics = {
            'mae': float('nan'),
            'rmse': float('nan'),
            'corr': float('nan'),
            'r2': float('nan')
        }
        if model_type in ['MoNIG', 'NIG']:
            metrics['mean_epistemic'] = float('nan')
            metrics['mean_aleatoric'] = float('nan')
        elif model_type == 'Gaussian':
            metrics['mean_variance'] = float('nan')
            metrics['mean_std'] = float('nan')
        return metrics
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = evaluate_predictions(all_preds, all_labels)
    
    if model_type in ['MoNIG', 'NIG'] or model_type.startswith('MoNIG_'):
        metrics['mean_epistemic'] = np.mean(all_epistemic)
        metrics['mean_aleatoric'] = np.mean(all_aleatoric)
    elif model_type in ['Gaussian', 'DeepEnsemble', 'MCDropout', 'SoftmaxMoE', 'DeepEnsembleMVE', 'CFGP', 'SWAG']:
        metrics['mean_variance'] = np.mean(all_variance)
        metrics['mean_std'] = np.sqrt(metrics['mean_variance'])
    
    return metrics
    

def collect_predictions_with_uncertainty(model, loader, device, model_type, expert_indices=None):
    """Return numpy arrays of predictions, stds, and true values for calibration."""
    if model_type not in ['MoNIG', 'NIG', 'Gaussian', 'DeepEnsemble', 'MCDropout', 'SoftmaxMoE', 'DeepEnsembleMVE', 'SWAG'] and not model_type.startswith('MoNIG_'):
        return None, None, None
    
    model.eval()
    preds = []
    trues = []
    stds = []
    eps = 1e-8
    
    with torch.no_grad():
        for inputs, labels, _ in loader:
            expert_scores, embeddings = inputs
            if expert_indices is not None and len(expert_indices) < expert_scores.shape[1]:
                expert_scores = expert_scores[:, expert_indices]
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            if model_type == 'MoNIG' or model_type.startswith('MoNIG_'):
                nigs = model(expert_scores, embeddings)
                if model_type == 'MoNIG_UniformWeightAggregation':
                    mu, v, alpha, beta = nigs[0]
                else:
                    mu, v, alpha, beta = aggregate_nigs(nigs)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                total_std = torch.sqrt(torch.clamp(epistemic + aleatoric, min=eps))
            elif model_type == 'NIG':
                mu, v, alpha, beta = model(expert_scores, embeddings)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                total_std = torch.sqrt(torch.clamp(epistemic + aleatoric, min=eps))
            elif model_type in ['Gaussian', 'DeepEnsemble', 'MCDropout', 'SoftmaxMoE', 'DeepEnsembleMVE', 'CFGP', 'SWAG']:
                mu, std = model(expert_scores, embeddings)
                total_std = std
            else:
                # Baseline - no uncertainty
                mu = model(expert_scores, embeddings)
                total_std = torch.zeros_like(mu)
            
            preds.append(mu.cpu().numpy())
            stds.append(total_std.cpu().numpy())
            trues.append(labels.cpu().numpy())
    
    if len(preds) == 0:
        return None, None, None
    
    y_pred = np.concatenate(preds, axis=0).flatten()
    y_std = np.concatenate(stds, axis=0).flatten()
    y_true = np.concatenate(trues, axis=0).flatten()
    return y_pred, y_std, y_true


def fit_isotonic_interval_calibrator(y_pred, y_std, y_true):
    """Fit isotonic regression calibrator (interval-based) using uncertainty_toolbox."""
    if y_pred is None or y_std is None or y_true is None:
        return None
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
        y_pred, y_std, y_true, prop_type='interval'
    )
    iso_model = uct.recalibration.iso_recal(exp_props, obs_props)
    return iso_model


def main():
    parser = argparse.ArgumentParser(description='Drug Discovery with MoNIG (Embeddings)')
    
    # Data
    parser.add_argument('--csv_path', type=str, 
                       default='pdbbind_descriptors_with_experts_and_binding.csv',
                       help='Path to CSV file')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    
    # Model
    parser.add_argument('--model_type', type=str, default='MoNIG',
                       choices=['MoNIG', 'NIG', 'Gaussian', 'Baseline', 'DeepEnsemble', 'MCDropout',
                                'MoNIG_NoReliabilityScaling', 'MoNIG_UniformReliability', 
                                'MoNIG_NoContextReliability', 'MoNIG_UniformWeightAggregation',
                                'SoftmaxMoE', 'DeepEnsembleMVE', 'CFGP', 'SWAG'],
                       help='Model type (including ablation variants and UQ baselines)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--num_models', type=int, default=5,
                       help='Number of models in ensemble (for DeepEnsemble)')
    parser.add_argument('--num_mc_samples', type=int, default=50,
                       help='Number of MC samples for MCDropout')
    parser.add_argument('--num_inducing', type=int, default=128,
                       help='Number of inducing points for CFGP (default: 128)')
    parser.add_argument('--max_num_models', type=int, default=20,
                       help='Maximum number of models for SWAG (default: 20)')
    parser.add_argument('--swag_start', type=int, default=75,
                       help='Epoch to start collecting SWAG models (default: 75)')
    parser.add_argument('--swag_lr', type=float, default=1e-3,
                       help='Learning rate for SWAG collection (default: 1e-3)')
    parser.add_argument('--swag_freq', type=int, default=1,
                       help='Frequency of collecting SWAG models (default: 1, every epoch)')
    parser.add_argument('--num_swag_samples', type=int, default=30,
                       help='Number of SWAG samples for inference (default: 30)')
    
    # Expert selection (for ablation studies)
    parser.add_argument('--expert1_only', action='store_true',
                       help='Use only Expert 1 (GNINA) (default: False)')
    parser.add_argument('--expert2_only', action='store_true',
                       help='Use only Expert 2 (BIND) (default: False)')
    parser.add_argument('--expert3_only', action='store_true',
                       help='Use only Expert 3 (flowdock) (default: False)')
    parser.add_argument('--expert4_only', action='store_true',
                       help='Use only Expert 4 (DynamicBind) (default: False)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--risk_weight', type=float, default=0.005,
                       help='Risk regularization weight')
    parser.add_argument('--conformal_coverage', type=float, default=0.95,
                       help='Target coverage for conformal prediction intervals (default: 0.95)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Validate and adjust device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    # Validate expert selection
    expert_flags = [args.expert1_only, args.expert2_only, getattr(args, 'expert3_only', False), getattr(args, 'expert4_only', False)]
    expert_mode = sum(expert_flags)
    expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
    
    if expert_mode == 0:
        # Use all experts (default)
        args.expert1_only = args.expert2_only = True
        args.expert3_only = True
        args.expert4_only = True
        expert_config = "All Experts (GNINA, BIND, flowdock, DynamicBind)"
    elif expert_mode == 1:
        # Use single expert
        selected_idx = [i for i, flag in enumerate(expert_flags) if flag][0]
        expert_config = f"Expert {selected_idx + 1} ({expert_names[selected_idx]}) only"
    else:
        # Multiple experts selected
        selected_indices = [i for i, flag in enumerate(expert_flags) if flag]
        selected_names = [expert_names[i] for i in selected_indices]
        expert_config = f"Experts: {', '.join(selected_names)}"
    
    # Set seed
    set_seed(args.seed)
    
    # Load test set PDB IDs from test.csv file
    test_pdb_ids = None
    test_file = 'data/test.csv'
    if os.path.isfile(test_file):
        # Read from CSV file (expecting 'name' column with PDB IDs)
        test_df = pd.read_csv(test_file)
        if 'name' in test_df.columns:
            test_pdb_ids = test_df['name'].astype(str).tolist()
        else:
            # Fallback: use first column
            test_pdb_ids = test_df.iloc[:, 0].astype(str).tolist()
        print(f"Loaded {len(test_pdb_ids)} test PDB IDs from {test_file}")
    else:
        raise FileNotFoundError(f"Test file not found at {test_file}. Please provide data/test.csv")
    
    print("="*50)
    print("Drug Discovery with MoNIG (Embeddings)")
    print("="*50)
    print(f"Model: {args.model_type}")
    print(f"Expert Config: {expert_config}")
    print(f"Device: {args.device}")
    print(f"CSV: {args.csv_path}")
    print(f"Test set: {len(test_pdb_ids)} complexes")
    print("="*50)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = DrugDiscoveryDatasetEmb(
        args.csv_path, split='train', seed=args.seed, test_pdb_ids=test_pdb_ids)
    norm_stats = {
        'mean': train_dataset.emb_mean,
        'std': train_dataset.emb_std
    }
    valid_dataset = DrugDiscoveryDatasetEmb(
        args.csv_path, split='valid', seed=args.seed, normalization_stats=norm_stats,
        test_pdb_ids=test_pdb_ids)
    test_dataset = DrugDiscoveryDatasetEmb(
        args.csv_path, split='test', seed=args.seed, normalization_stats=norm_stats,
        test_pdb_ids=test_pdb_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    embedding_dim, num_experts = train_dataset.get_dim()
    
    # Adjust number of experts based on selection
    expert_flags = [args.expert1_only, args.expert2_only, getattr(args, 'expert3_only', False), getattr(args, 'expert4_only', False)]
    selected_experts = [i for i, flag in enumerate(expert_flags) if flag]
    
    if len(selected_experts) == 0:
        # Default: use all experts
        actual_num_experts = num_experts
        expert_indices = list(range(num_experts))
        print(f"\nUsing all {num_experts} experts")
    elif len(selected_experts) == 1:
        actual_num_experts = 1
        expert_indices = selected_experts
        expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
        print(f"\nUsing Expert {selected_experts[0] + 1} ({expert_names[selected_experts[0]]}) only")
    else:
        actual_num_experts = len(selected_experts)
        expert_indices = selected_experts
        expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
        selected_names = [expert_names[i] for i in selected_experts]
        print(f"\nUsing experts: {', '.join(selected_names)}")
    
    print(f"\nData dimensions:")
    print(f"  Embeddings: {embedding_dim}")
    print(f"  Experts (available): {num_experts}")
    print(f"  Experts (using): {actual_num_experts}")
    
    # Create model
    class HyperParams:
        pass
    
    hyp_params = HyperParams()
    hyp_params.num_experts = actual_num_experts
    hyp_params.embedding_dim = embedding_dim
    hyp_params.hidden_dim = args.hidden_dim
    hyp_params.dropout = args.dropout
    hyp_params.expert_indices = expert_indices
    
    if args.model_type == 'MoNIG':
        model = DrugDiscoveryMoNIGEmb(hyp_params)
    elif args.model_type == 'MoNIG_NoReliabilityScaling':
        model = DrugDiscoveryMoNIG_NoReliabilityScaling(hyp_params)
    elif args.model_type == 'MoNIG_UniformReliability':
        model = DrugDiscoveryMoNIG_UniformReliability(hyp_params)
    elif args.model_type == 'MoNIG_NoContextReliability':
        model = DrugDiscoveryMoNIG_NoContextReliability(hyp_params)
    elif args.model_type == 'MoNIG_UniformWeightAggregation':
        model = DrugDiscoveryMoNIG_UniformWeightAggregation(hyp_params)
    elif args.model_type == 'NIG':
        model = DrugDiscoveryNIGEmb(hyp_params)
    elif args.model_type == 'Gaussian':
        model = DrugDiscoveryGaussianEmb(hyp_params)
    elif args.model_type == 'DeepEnsemble':
        hyp_params.num_models = args.num_models
        model = DrugDiscoveryDeepEnsemble(hyp_params)
    elif args.model_type == 'MCDropout':
        hyp_params.num_mc_samples = args.num_mc_samples
        model = DrugDiscoveryMCDropout(hyp_params)
    elif args.model_type == 'SoftmaxMoE':
        model = DrugDiscoverySoftmaxMoE(hyp_params)
    elif args.model_type == 'DeepEnsembleMVE':
        hyp_params.num_models = args.num_models
        model = DrugDiscoveryDeepEnsembleMVE(hyp_params)
    elif args.model_type == 'CFGP':
        hyp_params.num_inducing = args.num_inducing if hasattr(args, 'num_inducing') else 128
        model = DrugDiscoveryCFGP(hyp_params)
    elif args.model_type == 'SWAG':
        hyp_params.max_num_models = args.max_num_models if hasattr(args, 'max_num_models') else 20
        hyp_params.no_cov_mat = True  # Use diagonal covariance for efficiency
        hyp_params.num_swag_samples = args.num_swag_samples if hasattr(args, 'num_swag_samples') else 30
        model = DrugDiscoverySWAG(hyp_params)
        # Base model is stored inside SWAG wrapper
        base_model = model.base_model
    else:
        model = DrugDiscoveryBaselineEmb(hyp_params)
    
    model = model.to(args.device)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer and scheduler (for neural network models)
    if args.model_type == 'SWAG':
        # For SWAG, optimize the base_model (separate from SWAG wrapper's internal base)
        base_model = model.base_model
        optimizer = optim.Adam(base_model.parameters(), lr=args.lr, weight_decay=1e-5)
        # SWAG collection uses a separate optimizer with different LR
        swag_optimizer = optim.Adam(base_model.parameters(), lr=args.swag_lr, weight_decay=1e-5) if hasattr(args, 'swag_lr') else None
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        swag_optimizer = None
        base_model = None
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
    
    # Training loop
    print("\nStarting training...")
    best_val_mae = float('inf')
    patience_counter = 0
    max_patience = 30
    
    # Precompute where to save model/stats
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/best_{args.model_type}_emb.pt'
    norm_stats_path = f'saved_models/best_{args.model_type}_emb_norm_stats.npz'

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device, 
                                args.model_type, args.risk_weight, expert_indices)
        
        # Evaluate
        val_metrics = evaluate(model, valid_loader, args.device, args.model_type, expert_indices)
        
        # Scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['mae'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # SWAG collection (after swag_start epoch)
        if args.model_type == 'SWAG' and epoch >= args.swag_start and epoch % args.swag_freq == 0:
            # Switch to SWAG learning rate for collection
            if swag_optimizer is not None:
                for param_group in swag_optimizer.param_groups:
                    param_group['lr'] = args.swag_lr
                # Do one step with SWAG LR
                for batch_idx, (inputs, labels, _) in enumerate(train_loader):
                    expert_scores, embeddings = inputs
                    if expert_indices is not None and len(expert_indices) < 2:
                        expert_scores = expert_scores[:, expert_indices]
                    expert_scores = expert_scores.to(args.device)
                    embeddings = embeddings.to(args.device)
                    labels = labels.to(args.device).unsqueeze(1)
                    
                    swag_optimizer.zero_grad()
                    mu, sigma = base_model(expert_scores, embeddings)
                    loss = criterion_gaussian(mu, sigma, labels)
                    loss.backward()
                    swag_optimizer.step()
                    break  # Just one batch for SWAG collection
            
            # Collect model snapshot
            model.collect_model(base_model)
            print(f"  → Collected SWAG model snapshot (n={model.swag.n_models.item()})")
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                  f"Corr: {val_metrics['corr']:.4f}, R²: {val_metrics['r2']:.4f}")
            if args.model_type in ['MoNIG', 'NIG'] or args.model_type.startswith('MoNIG_'):
                print(f"  Epistemic: {val_metrics['mean_epistemic']:.4f}, "
                      f"Aleatoric: {val_metrics['mean_aleatoric']:.4f}")
            if args.model_type == 'SWAG' and epoch >= args.swag_start:
                print(f"  SWAG models collected: {model.swag.n_models.item()}")
        
        # Save best model and early stopping
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            # Create models directory if it doesn't exist
            if args.model_type == 'SWAG':
                # For SWAG, save the SWAG model (which includes statistics)
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            print(f"  → New best model saved to {model_path} (MAE: {best_val_mae:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    # Load best model and fit isotonic recalibrator on validation set
    print("\n" + "="*50)
    print("Final evaluation on test set...")
    model.load_state_dict(torch.load(model_path))
    
    # Compute conformal prediction quantile on validation set
    # TEMPORARILY DISABLED - CP removed
    # print("\nComputing conformal prediction quantile on validation set...")
    # conformal_quantile = compute_conformal_quantile(
    #     model, valid_loader, args.device, args.model_type, 
    #     expert_indices, coverage=args.conformal_coverage
    # )
    # conformal_path = f'saved_models/best_{args.model_type}_emb_conformal.npz'
    # np.savez(conformal_path, quantile=conformal_quantile, coverage=args.conformal_coverage)
    # print(f"Conformal quantile: {conformal_quantile:.4f} (coverage: {args.conformal_coverage})")
    # print(f"Saved conformal quantile to {conformal_path}")
    
    iso_model = None
    calibrator_path = f'saved_models/best_{args.model_type}_emb_calibrator.pkl'
    if args.model_type in ['MoNIG', 'NIG'] or args.model_type.startswith('MoNIG_'):
        y_pred_val, y_std_val, y_true_val = collect_predictions_with_uncertainty(
            model, valid_loader, args.device, args.model_type, expert_indices)
        try:
            iso_model = fit_isotonic_interval_calibrator(y_pred_val, y_std_val, y_true_val)
        except RuntimeError as err:
            print(f"Warning: isotonic recalibration failed ({err}). Proceeding without it.")
            iso_model = None
        if iso_model is not None:
            with open(calibrator_path, 'wb') as f:
                pickle.dump({'iso_model': iso_model}, f)
            print(f"Saved isotonic recalibrator to {calibrator_path}")
        else:
            calibrator_path = None
    else:
        calibrator_path = None
    
    np.savez(norm_stats_path,
             mean=norm_stats['mean'],
             std=norm_stats['std'])
    print(f"Saved normalization stats to {norm_stats_path}")
    if calibrator_path:
        print(f"Calibration artifact: {calibrator_path}")

    # Evaluate on test set if it has data
    if len(test_dataset) > 0:
        test_metrics = evaluate(model, test_loader, args.device, args.model_type, expert_indices)
        
        print(f"\nTest Results:")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Correlation: {test_metrics['corr']:.4f}")
        print(f"  R²: {test_metrics['r2']:.4f}")
        if args.model_type in ['MoNIG', 'NIG'] or args.model_type.startswith('MoNIG_'):
            print(f"  Mean Epistemic Uncertainty: {test_metrics['mean_epistemic']:.4f}")
            print(f"  Mean Aleatoric Uncertainty: {test_metrics['mean_aleatoric']:.4f}")
        elif args.model_type == 'Gaussian':
            print(f"  Mean Predicted Variance: {test_metrics['mean_variance']:.4f}")
            print(f"  Mean Predicted Std Dev: {test_metrics['mean_std']:.4f}")
    else:
        print("\nTest Results: [Skipped - test set is empty]")
        print("Note: Run inference separately on test split for evaluation")
    print("="*50)


if __name__ == '__main__':
    main()

