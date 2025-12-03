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
    DrugDiscoveryRandomForest,
    DrugDiscoveryMoNIG_NoReliabilityScaling, DrugDiscoveryMoNIG_UniformReliability,
    DrugDiscoveryMoNIG_FixedReliability, DrugDiscoveryMoNIG_UniformWeightAggregation
)
from src.utils import moe_nig


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


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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


def inference_monig(model, loader, device, conformal_quantile=None):
    """
    Run inference and extract detailed per-expert information
    
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
                
                # Conformal prediction intervals
                if conformal_quantile is not None:
                    lower, upper = get_conformal_intervals(
                        mu_agg[i, 0], total_std, conformal_quantile
                    )
                    result['Conformal_Lower'] = lower
                    result['Conformal_Upper'] = upper
                    result['Conformal_Width'] = upper - lower
                else:
                    result['Conformal_Lower'] = np.nan
                    result['Conformal_Upper'] = np.nan
                    result['Conformal_Width'] = np.nan
                
                results.append(result)
    
    return pd.DataFrame(results)


def inference_general(model, loader, device, model_type, conformal_quantile=None):
    """
    General inference function for non-MoNIG models
    
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
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.cpu().numpy()
            
            if model_type in ['DeepEnsemble', 'MCDropout', 'RandomForest']:
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
                
                # Conformal prediction intervals
                if conformal_quantile is not None:
                    lower, upper = get_conformal_intervals(
                        predictions[i, 0], uncertainties[i, 0], conformal_quantile
                    )
                    result['Conformal_Lower'] = lower
                    result['Conformal_Upper'] = upper
                    result['Conformal_Width'] = upper - lower
                else:
                    result['Conformal_Lower'] = np.nan
                    result['Conformal_Upper'] = np.nan
                    result['Conformal_Width'] = np.nan
                
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
    parser.add_argument('--conformal_path', type=str, default=None,
                        help='Path to conformal quantile (.npz). Defaults to model_path-derived file')
    
    # Data split
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Which split to run inference on')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    
    # Model config (must match training)
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['MoNIG', 'NIG', 'Gaussian', 'Baseline', 'DeepEnsemble', 'MCDropout', 'RandomForest'],
                       help='Model type (auto-detected from model_path if not provided)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension (must match training)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (must match training)')
    parser.add_argument('--num_models', type=int, default=5,
                       help='Number of models in ensemble (for DeepEnsemble, must match training)')
    parser.add_argument('--num_mc_samples', type=int, default=50,
                       help='Number of MC samples for MCDropout')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees for RandomForest (must match training)')
    parser.add_argument('--max_depth', type=int, default=20,
                       help='Max depth for RandomForest (must match training)')
    
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
    
    # Load conformal quantile
    if args.conformal_path is not None:
        conformal_path = args.conformal_path
    else:
        base, _ = os.path.splitext(args.model_path)
        conformal_path = base + '_conformal.npz'
    
    conformal_quantile = None
    if os.path.exists(conformal_path):
        conformal_npz = np.load(conformal_path)
        conformal_quantile = float(conformal_npz['quantile'])
        conformal_coverage = float(conformal_npz.get('coverage', 0.95))
        print(f"Loaded conformal quantile: {conformal_quantile:.4f} (coverage: {conformal_coverage})")
    else:
        print(f"Warning: Conformal quantile not found at {conformal_path}. Inference will proceed without conformal intervals.")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = DrugDiscoveryDatasetEmb(
        args.csv_path, split=args.split, normalization_stats=norm_stats,
        test_pdb_ids=test_pdb_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    embedding_dim, num_experts = dataset.get_dim()
    print(f"Embeddings: {embedding_dim}, Experts: {num_experts}")
    
    # Create model
    print("\nCreating model...")
    class HyperParams:
        pass
    
    # Auto-detect model type from model_path if not provided
    if args.model_type is None:
        model_name = os.path.basename(args.model_path)
        if 'MoNIG' in model_name:
            args.model_type = 'MoNIG'
        elif 'NIG' in model_name:
            args.model_type = 'NIG'
        elif 'Gaussian' in model_name:
            args.model_type = 'Gaussian'
        elif 'DeepEnsemble' in model_name:
            args.model_type = 'DeepEnsemble'
        elif 'MCDropout' in model_name:
            args.model_type = 'MCDropout'
        elif 'RandomForest' in model_name:
            args.model_type = 'RandomForest'
        else:
            args.model_type = 'Baseline'
        print(f"Auto-detected model type: {args.model_type}")
    
    hyp_params = HyperParams()
    hyp_params.num_experts = num_experts
    hyp_params.embedding_dim = embedding_dim
    hyp_params.hidden_dim = args.hidden_dim
    hyp_params.dropout = args.dropout
    
    if args.model_type == 'MoNIG':
        model = DrugDiscoveryMoNIGEmb(hyp_params)
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
    elif args.model_type == 'RandomForest':
        hyp_params.n_estimators = args.n_estimators
        hyp_params.max_depth = getattr(args, 'max_depth', 20)  # Default is 20
        hyp_params.min_samples_split = 2
        hyp_params.min_samples_leaf = 1
        model = DrugDiscoveryRandomForest(hyp_params)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    if args.model_type != 'RandomForest':
        model = model.to(args.device)
    
    # Load weights
    print(f"Loading model weights from {args.model_path}...")
    # RandomForest models contain sklearn objects, so we need weights_only=False
    if args.model_type == 'RandomForest':
        model.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=False))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    if args.model_type != 'RandomForest':
        model = model.to(args.device)
    print(f"Model loaded successfully!")
    
    # Run inference
    print("\nRunning inference...")
    if args.model_type == 'MoNIG' or args.model_type.startswith('MoNIG_'):
        df_results = inference_monig(model, loader, args.device, conformal_quantile=conformal_quantile)
    else:
        # General inference for other models
        df_results = inference_general(model, loader, args.device, args.model_type, conformal_quantile=conformal_quantile)
    
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
        for j in range(num_experts):
            expert_num = j + 1
            expert_mae = np.mean(np.abs(df_results[f'Expert{expert_num}_Prediction'] - df_results['True_Affinity']))
            print(f"  Expert{expert_num} MAE: {expert_mae:.4f}")
        
        print(f"\nExpert Confidence:")
        for j in range(num_experts):
            expert_num = j + 1
            avg_nu = df_results[f'Expert{expert_num}_Confidence_nu'].mean()
            print(f"  Expert{expert_num} avg ν: {avg_nu:.2f}")
        
        print(f"\nExpert Weights:")
        for j in range(num_experts):
            expert_num = j + 1
            avg_weight = df_results[f'Expert{expert_num}_Weight'].mean()
            print(f"  Expert{expert_num} avg weight: {avg_weight:.3f}")
        
        print(f"\nExpert Trust Distribution:")
        for j in range(num_experts):
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
        print(f"\nTop 5 samples with highest expert disagreement:")
        print(high_disagreement[['ComplexID', 'Expert1_Prediction', 'Expert2_Prediction', 
                                 'Expert_Disagreement', 'MoNIG_Prediction', 'True_Affinity']].to_string(index=False))
        
        high_conf_diff = df_results.nlargest(5, 'Confidence_Ratio')
        print(f"\nTop 5 samples with highest confidence difference:")
        print(high_conf_diff[['ComplexID', 'More_Confident_Expert', 'Confidence_Ratio',
                              'Expert1_Weight', 'Expert2_Weight', 'MoNIG_Prediction']].to_string(index=False))
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

