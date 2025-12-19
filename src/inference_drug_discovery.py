"""
Inference script for Drug Discovery MoNIG
Analyzes expert trustworthiness and predictions for each sample
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
from src.utils import moe_nig


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


def aggregate_nigs(nigs):
    """Aggregate multiple NIGs using moe_nig from utils.py"""
    if len(nigs) == 0:
        raise ValueError("Cannot aggregate empty list of NIGs")
    if len(nigs) == 1:
        return nigs[0]
    
    mu_final, v_final, alpha_final, beta_final = nigs[0]
    for mu, v, alpha, beta in nigs[1:]:
        mu_final, v_final, alpha_final, beta_final = moe_nig(
            mu_final, v_final, alpha_final, beta_final,
            mu, v, alpha, beta
        )
    return mu_final, v_final, alpha_final, beta_final


def inference_monig(model, loader, device, expert_indices=None):
    """
    Run inference and extract detailed per-expert information
    
    Args:
        model: Trained MoNIG model
        loader: DataLoader
        device: Device to run on
        expert_indices: Optional list of expert indices to use (for filtering)
    
    Returns:
        DataFrame with columns:
        - ComplexID
        - True_Affinity
        - Expert1_Prediction, Expert2_Prediction, Expert3_Prediction, Expert4_Prediction, ...
        - Expert1_Confidence (ν), Expert2_Confidence (ν), Expert3_Confidence (ν), Expert4_Confidence (ν), ...
        - Expert1_Weight, Expert2_Weight, Expert3_Weight, Expert4_Weight, ... (normalized)
        - MoNIG_Prediction (aggregated)
        - Expert1_Epistemic, Expert2_Epistemic, Expert3_Epistemic, Expert4_Epistemic, ...
        - Expert1_Aleatoric, Expert2_Aleatoric, Expert3_Aleatoric, Expert4_Aleatoric, ...
        - MoNIG_Epistemic
        - MoNIG_Aleatoric
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for inputs, labels, complex_ids in loader:
            expert_scores, embeddings = inputs
            
            # Filter expert scores if expert_indices is provided
            # Only filter if we have more experts available than we need
            if expert_indices is not None and len(expert_indices) > 0 and len(expert_indices) < expert_scores.shape[1]:
                expert_scores = expert_scores[:, expert_indices]
            
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.cpu().numpy()
            
            # Get per-expert NIGs
            nigs = model(expert_scores, embeddings)
            num_experts = len(nigs)
            
            # For UniformWeightAggregation, nigs is already aggregated (single element)
            # We need to handle this case specially since we can't extract per-expert info
            is_aggregated = (num_experts == 1)
            
            if is_aggregated:
                # Already aggregated - use directly
                mu_final, v_final, alpha_final, beta_final = nigs[0]
                # Create dummy expert data for compatibility
                expert_data = []
                # We don't have per-expert data, so we'll skip per-expert columns
            else:
                # Aggregate
                mu_final, v_final, alpha_final, beta_final = aggregate_nigs(nigs)
                
                # Extract all experts to numpy
                expert_data = []
                for j in range(num_experts):
                    mu_j, v_j, alpha_j, beta_j = [x.cpu().numpy() for x in nigs[j]]
                    expert_data.append({
                        'mu': mu_j,
                        'v': v_j,
                        'alpha': alpha_j,
                        'beta': beta_j
                    })
            
            mu_agg = mu_final.cpu().numpy()
            v_agg = v_final.cpu().numpy()
            alpha_agg = alpha_final.cpu().numpy()
            beta_agg = beta_final.cpu().numpy()
            
            epistemic_agg = beta_agg / (v_agg * (alpha_agg - 1))
            aleatoric_agg = beta_agg / (alpha_agg - 1)
            
            # Process each sample in batch
            for i in range(len(labels)):
                epistemic_val = epistemic_agg[i, 0]
                aleatoric_val = aleatoric_agg[i, 0]
                total_std = np.sqrt(epistemic_val + aleatoric_val)
                
                # Build result dictionary
                result = {
                    'ComplexID': complex_ids[i],
                    'True_Affinity': labels[i],
                }
                
                # Add per-expert data (skip if already aggregated)
                if not is_aggregated:
                    # Compute weights (how much each expert contributes)
                    total_v = sum(expert_data[j]['v'][i, 0] for j in range(num_experts))
                    weights = [expert_data[j]['v'][i, 0] / total_v for j in range(num_experts)]
                    
                    # Compute uncertainties for each expert
                    expert_uncertainties = []
                    for j in range(num_experts):
                        v_j = expert_data[j]['v'][i, 0]
                        alpha_j = expert_data[j]['alpha'][i, 0]
                        beta_j = expert_data[j]['beta'][i, 0]
                        epistemic_j = beta_j / (v_j * (alpha_j - 1))
                        aleatoric_j = beta_j / (alpha_j - 1)
                        expert_uncertainties.append({
                            'epistemic': epistemic_j,
                            'aleatoric': aleatoric_j
                        })
                    
                    for j in range(num_experts):
                        expert_num = j + 1
                        result[f'Expert{expert_num}_Prediction'] = expert_data[j]['mu'][i, 0]
                        result[f'Expert{expert_num}_Confidence_nu'] = expert_data[j]['v'][i, 0]
                        result[f'Expert{expert_num}_Weight'] = weights[j]
                        result[f'Expert{expert_num}_Epistemic'] = expert_uncertainties[j]['epistemic']
                        result[f'Expert{expert_num}_Aleatoric'] = expert_uncertainties[j]['aleatoric']
                    
                    # Analysis: find most confident expert
                    confidences = [expert_data[j]['v'][i, 0] for j in range(num_experts)]
                    most_confident_idx = np.argmax(confidences)
                    result['More_Confident_Expert'] = most_confident_idx + 1
                    result['Confidence_Ratio'] = max(confidences) / min(confidences)
                    
                    # Expert disagreement: max pairwise difference
                    predictions = [expert_data[j]['mu'][i, 0] for j in range(num_experts)]
                    if num_experts >= 2:
                        max_disagreement = max(abs(predictions[idx1] - predictions[idx2]) 
                                              for idx1 in range(num_experts) 
                                              for idx2 in range(idx1+1, num_experts))
                        result['Expert_Disagreement'] = max_disagreement
                    else:
                        result['Expert_Disagreement'] = 0.0
                else:
                    # For aggregated models, set these to NaN
                    result['More_Confident_Expert'] = np.nan
                    result['Confidence_Ratio'] = np.nan
                    result['Expert_Disagreement'] = np.nan
                
                # MoNIG aggregated
                result['MoNIG_Prediction'] = mu_agg[i, 0]
                result['MoNIG_Epistemic'] = epistemic_val
                result['MoNIG_Aleatoric'] = aleatoric_val
                result['MoNIG_Std'] = total_std
                
                results.append(result)
    
    return pd.DataFrame(results)


def inference_general(model, loader, device, model_type, expert_indices=None):
    """
    General inference function for non-MoNIG models
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Device to run on
        model_type: Type of model
        expert_indices: Optional list of expert indices to use (for filtering)
    
    Returns:
        DataFrame with columns:
        - ComplexID
        - True_Affinity
        - Prediction
        - Uncertainty (std)
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for inputs, labels, complex_ids in loader:
            expert_scores, embeddings = inputs
            
            # Filter expert scores if expert_indices is provided
            # Only filter if we have more experts available than we need
            if expert_indices is not None and len(expert_indices) > 0 and len(expert_indices) < expert_scores.shape[1]:
                expert_scores = expert_scores[:, expert_indices]
            
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.cpu().numpy()
            
            if model_type in ['DeepEnsemble', 'MCDropout', 'SoftmaxMoE', 'DeepEnsembleMVE', 'CFGP', 'SWAG']:
                mu, std = model(expert_scores, embeddings)
                predictions = mu.cpu().numpy()
                uncertainties = std.cpu().numpy()
            elif model_type == 'Gaussian':
                mu, sigma = model(expert_scores, embeddings)
                predictions = mu.cpu().numpy()
                uncertainties = torch.sqrt(sigma).cpu().numpy()
            elif model_type == 'NIG':
                mu, v, alpha, beta = model(expert_scores, embeddings)
                epistemic = beta / (v * (alpha - 1))
                aleatoric = beta / (alpha - 1)
                total_var = epistemic + aleatoric
                predictions = mu.cpu().numpy()
                uncertainties = torch.sqrt(total_var).cpu().numpy()
            else:  # Baseline
                predictions = model(expert_scores, embeddings).cpu().numpy()
                uncertainties = np.zeros_like(predictions)
            
            # Process each sample in batch
            for i in range(len(labels)):
                result = {
                    'ComplexID': complex_ids[i],
                    'True_Affinity': labels[i],
                    'Prediction': predictions[i, 0],
                    'Uncertainty': uncertainties[i, 0],
                }
                
                results.append(result)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='MoNIG Inference - Analyze Expert Predictions')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--csv_path', type=str, 
                       default='pdbbind_descriptors_with_experts_and_binding.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output_path', type=str, default='test_inference_results.csv',
                       help='Path to save output CSV (default: test_inference_results.csv)')
    parser.add_argument('--norm_stats_path', type=str, default=None,
                        help='Path to normalization stats (.npz). Defaults to model_path-derived file')
    
    # Data split
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Which split to run inference on')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    
    # Model config (must match training)
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['MoNIG', 'NIG', 'Gaussian', 'Baseline', 'DeepEnsemble', 'MCDropout',
                                'SoftmaxMoE', 'DeepEnsembleMVE', 'CFGP', 'SWAG'],
                       help='Model type (auto-detected from model_path if not provided)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension (must match training)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (must match training)')
    parser.add_argument('--num_models', type=int, default=5,
                       help='Number of models in ensemble (for DeepEnsemble, must match training)')
    parser.add_argument('--num_mc_samples', type=int, default=50,
                       help='Number of MC samples for MCDropout')
    parser.add_argument('--num_inducing', type=int, default=128,
                       help='Number of inducing points for CFGP (must match training)')
    parser.add_argument('--max_num_models', type=int, default=20,
                       help='Maximum number of models for SWAG (must match training)')
    parser.add_argument('--num_swag_samples', type=int, default=30,
                       help='Number of SWAG samples for inference (default: 30)')
    
    # Expert selection (must match training)
    parser.add_argument('--expert1_only', action='store_true',
                       help='Use only Expert 1 (GNINA) - must match training')
    parser.add_argument('--expert2_only', action='store_true',
                       help='Use only Expert 2 (BIND) - must match training')
    parser.add_argument('--expert3_only', action='store_true',
                       help='Use only Expert 3 (flowdock) - must match training')
    parser.add_argument('--expert4_only', action='store_true',
                       help='Use only Expert 4 (DynamicBind) - must match training')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Validate and adjust device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    # Set seed for reproducibility
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
    
    print("="*70)
    print("MoNIG Inference - Expert Analysis")
    print("="*70)
    print(f"Random seed: 42")
    print(f"Model: {args.model_path}")
    print(f"Input CSV: {args.csv_path}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_path}")
    print(f"Device: {args.device}")
    print(f"Test set: {len(test_pdb_ids)} complexes")
    print("="*70)
    # Load normalization stats
    if args.norm_stats_path is not None:
        norm_stats_path = args.norm_stats_path
    else:
        base, _ = os.path.splitext(args.model_path)
        norm_stats_path = base + '_norm_stats.npz'
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f"Normalization stats not found at {norm_stats_path}. Provide via --norm_stats_path")
    norm_stats_npz = np.load(norm_stats_path)
    norm_stats = {'mean': norm_stats_npz['mean'], 'std': norm_stats_npz['std']}
    print(f"Loaded normalization stats from {norm_stats_path}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = DrugDiscoveryDatasetEmb(
        args.csv_path, split=args.split, normalization_stats=norm_stats,
        test_pdb_ids=test_pdb_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    embedding_dim, num_experts_available = dataset.get_dim()
    print(f"Embeddings: {embedding_dim}, Experts (available): {num_experts_available}")
    
    # Determine which experts to use (must match training)
    expert_flags = [args.expert1_only, args.expert2_only, args.expert3_only, args.expert4_only]
    expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
    
    if sum(expert_flags) == 0:
        # Use all experts (default)
        expert_indices = list(range(num_experts_available))
        actual_num_experts = num_experts_available
        print(f"Using all {num_experts_available} experts")
    else:
        # Use selected experts
        expert_indices = [i for i, flag in enumerate(expert_flags) if flag]
        actual_num_experts = len(expert_indices)
        selected_names = [expert_names[i] for i in expert_indices]
        print(f"Using {actual_num_experts} experts: {', '.join(selected_names)}")
    
    # Create model
    print("\nCreating model...")
    class HyperParams:
        pass
    
    # Auto-detect model type from model_path if not provided
    if args.model_type is None:
        model_name = os.path.basename(args.model_path)
        # Check for MoNIG variants first (before generic MoNIG, since 'MoNIG' matches all variants)
        if 'MoNIG_NoReliabilityScaling' in model_name:
            args.model_type = 'MoNIG_NoReliabilityScaling'
        elif 'MoNIG_UniformReliability' in model_name:
            args.model_type = 'MoNIG_UniformReliability'
        elif 'MoNIG_NoContextReliability' in model_name:
            args.model_type = 'MoNIG_NoContextReliability'
        elif 'MoNIG_UniformWeightAggregation' in model_name:
            args.model_type = 'MoNIG_UniformWeightAggregation'
        elif 'MoNIG_FixedReliability' in model_name:
            args.model_type = 'MoNIG_FixedReliability'
        elif 'SoftmaxMoE' in model_name:
            args.model_type = 'SoftmaxMoE'
        elif 'DeepEnsembleMVE' in model_name:
            args.model_type = 'DeepEnsembleMVE'
        elif 'CFGP' in model_name:
            args.model_type = 'CFGP'
        elif 'SWAG' in model_name:
            args.model_type = 'SWAG'
        elif 'MoNIG' in model_name:
            args.model_type = 'MoNIG'
        elif 'NIG' in model_name:
            args.model_type = 'NIG'
        elif 'Gaussian' in model_name:
            args.model_type = 'Gaussian'
        elif 'DeepEnsemble' in model_name:
            args.model_type = 'DeepEnsemble'
        elif 'MCDropout' in model_name:
            args.model_type = 'MCDropout'
        else:
            args.model_type = 'Baseline'
        print(f"Auto-detected model type: {args.model_type}")
    
    hyp_params = HyperParams()
    hyp_params.num_experts = actual_num_experts  # Use actual number of experts, not available
    hyp_params.embedding_dim = embedding_dim
    hyp_params.hidden_dim = args.hidden_dim
    hyp_params.dropout = args.dropout
    hyp_params.expert_indices = expert_indices  # Store expert indices for filtering
    
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
    elif args.model_type == 'Baseline':
        model = DrugDiscoveryBaselineEmb(hyp_params)
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
        hyp_params.num_inducing = getattr(args, 'num_inducing', 128)
        model = DrugDiscoveryCFGP(hyp_params)
    elif args.model_type == 'SWAG':
        hyp_params.max_num_models = getattr(args, 'max_num_models', 20)
        hyp_params.no_cov_mat = True
        hyp_params.num_swag_samples = getattr(args, 'num_swag_samples', 30)
        model = DrugDiscoverySWAG(hyp_params)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(args.device)
    
    # Load weights
    print(f"Loading model weights from {args.model_path}...")
    try:
        if args.model_type == 'SWAG':
            # SWAG models may have special state dict structure
            state_dict = torch.load(args.model_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load(args.model_path, map_location=args.device)
            # Try strict loading first, fall back to non-strict if it fails
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: Strict loading failed: {e}")
                print("Attempting non-strict loading...")
                model.load_state_dict(state_dict, strict=False)
        model = model.to(args.device)
        print(f"Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load model weights: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Run inference
    print("\nRunning inference...")
    if args.model_type == 'MoNIG' or args.model_type.startswith('MoNIG_'):
        df_results = inference_monig(model, loader, args.device, expert_indices=expert_indices)
    else:
        # General inference for other models
        df_results = inference_general(model, loader, args.device, args.model_type, expert_indices=expert_indices)
    
    # Save results
    print(f"\nSaving results to {args.output_path}...")
    df_results.to_csv(args.output_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("INFERENCE SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df_results)}")
    print(f"Model type: {args.model_type}")
    
    if args.model_type == 'MoNIG':
        print(f"\nPrediction Accuracy:")
        print(f"  MoNIG MAE:  {np.mean(np.abs(df_results['MoNIG_Prediction'] - df_results['True_Affinity'])):.4f}")
        
        # Print per-expert MAE dynamically
        for j in range(actual_num_experts):
            expert_num = j + 1
            expert_mae = np.mean(np.abs(df_results[f'Expert{expert_num}_Prediction'] - df_results['True_Affinity']))
            print(f"  Expert{expert_num} MAE: {expert_mae:.4f}")
        
        print(f"\nExpert Confidence:")
        for j in range(actual_num_experts):
            expert_num = j + 1
            avg_nu = df_results[f'Expert{expert_num}_Confidence_nu'].mean()
            print(f"  Expert{expert_num} avg ν: {avg_nu:.2f}")
        
        print(f"\nExpert Weights:")
        for j in range(actual_num_experts):
            expert_num = j + 1
            avg_weight = df_results[f'Expert{expert_num}_Weight'].mean()
            print(f"  Expert{expert_num} avg weight: {avg_weight:.3f}")
        
        print(f"\nExpert Trust Distribution:")
        for j in range(actual_num_experts):
            expert_num = j + 1
            count = (df_results['More_Confident_Expert'] == expert_num).sum()
            percentage = count / len(df_results) * 100
            print(f"  Expert{expert_num} more confident: {count} samples ({percentage:.1f}%)")
        
        print(f"\nUncertainty:")
        print(f"  Avg epistemic (MoNIG): {df_results['MoNIG_Epistemic'].mean():.4f}")
        print(f"  Avg aleatoric (MoNIG): {df_results['MoNIG_Aleatoric'].mean():.4f}")
        
        # Highlight interesting cases
        print(f"\nInteresting Cases:")
        high_disagreement = df_results.nlargest(5, 'Expert_Disagreement')
        # Build column list dynamically based on number of experts
        expert_pred_cols = [f'Expert{j+1}_Prediction' for j in range(actual_num_experts)]
        print(f"\nTop 5 samples with highest expert disagreement:")
        cols_to_show = ['ComplexID'] + expert_pred_cols + ['Expert_Disagreement', 'MoNIG_Prediction', 'True_Affinity']
        cols_to_show = [c for c in cols_to_show if c in high_disagreement.columns]
        print(high_disagreement[cols_to_show].to_string(index=False))
        
        high_conf_diff = df_results.nlargest(5, 'Confidence_Ratio')
        print(f"\nTop 5 samples with highest confidence difference:")
        expert_weight_cols = [f'Expert{j+1}_Weight' for j in range(actual_num_experts)]
        cols_to_show = ['ComplexID', 'More_Confident_Expert', 'Confidence_Ratio'] + expert_weight_cols + ['MoNIG_Prediction']
        cols_to_show = [c for c in cols_to_show if c in high_conf_diff.columns]
        print(high_conf_diff[cols_to_show].to_string(index=False))
    else:
        # General model statistics
        pred_col = 'Prediction' if 'Prediction' in df_results.columns else 'MoNIG_Prediction'
        true_col = 'True_Affinity'
        unc_col = 'Uncertainty' if 'Uncertainty' in df_results.columns else None
        
        mae = np.mean(np.abs(df_results[pred_col] - df_results[true_col]))
        rmse = np.sqrt(np.mean((df_results[pred_col] - df_results[true_col]) ** 2))
        corr = np.corrcoef(df_results[pred_col], df_results[true_col])[0, 1]
        
        print(f"\nPrediction Accuracy:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Correlation: {corr:.4f}")
        
        if unc_col and unc_col in df_results.columns:
            print(f"\nUncertainty:")
            print(f"  Avg uncertainty: {df_results[unc_col].mean():.4f}")
            print(f"  Std uncertainty: {df_results[unc_col].std():.4f}")
    
    print("="*70)
    print(f"✓ Inference complete! Results saved to: {args.output_path}")
    print("="*70)


if __name__ == '__main__':
    main()

