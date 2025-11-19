"""
Inference script for Drug Discovery MoNIG
Analyzes expert trustworthiness and predictions for each sample
"""
import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

import uncertainty_toolbox as uct

from src.drug_dataset_emb import DrugDiscoveryDatasetEmb
from src.drug_models_emb import DrugDiscoveryMoNIGEmb
from src.utils import moe_nig


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


def apply_interval_calibration(y_pred, y_std, iso_model, coverage=0.682689):
    """Recalibrate total std using isotonic interval recalibration."""
    if iso_model is None:
        return y_std
    if y_pred.size == 0:
        return y_std
    z = stats.norm.ppf(0.5 + coverage / 2.0)
    bounds = uct.metrics_calibration.get_prediction_interval(
        y_pred.flatten(),
        y_std.flatten(),
        np.array([coverage]),
        recal_model=iso_model
    )
    width = bounds.upper - bounds.lower
    recalibrated_std = (width / (2.0 * z)).reshape(y_std.shape)
    return recalibrated_std


def inference_monig(model, loader, device, iso_model=None):
    """
    Run inference and extract detailed per-expert information
    
    Returns:
        DataFrame with columns:
        - ComplexID
        - True_Affinity
        - Expert1_Prediction, Expert2_Prediction, Expert3_Prediction
        - Expert1_Confidence (ν), Expert2_Confidence (ν), Expert3_Confidence (ν)
        - Expert1_Weight, Expert2_Weight, Expert3_Weight (normalized)
        - MoNIG_Prediction (aggregated)
        - Expert1_Epistemic, Expert2_Epistemic, Expert3_Epistemic
        - Expert1_Aleatoric, Expert2_Aleatoric, Expert3_Aleatoric
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
            total_std = np.sqrt(epistemic_agg + aleatoric_agg)
            calibrated_std = apply_interval_calibration(
                mu_agg, total_std, iso_model
            )
            total_var = total_std ** 2
            calibrated_var = calibrated_std ** 2
            scale = np.ones_like(calibrated_var)
            valid_mask = total_var > 0
            scale[valid_mask] = calibrated_var[valid_mask] / total_var[valid_mask]
            scale = np.clip(scale, 1e-4, None)
            epistemic_agg = epistemic_agg * scale
            aleatoric_agg = aleatoric_agg * scale
            
            # Process each sample in batch
            for i in range(len(labels)):
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
                
                epistemic_val = epistemic_agg[i, 0]
                aleatoric_val = aleatoric_agg[i, 0]
                
                # Build result dictionary
                result = {
                    'ComplexID': complex_ids[i],
                    'True_Affinity': labels[i],
                }
                
                # Add per-expert data
                for j in range(num_experts):
                    expert_num = j + 1
                    result[f'Expert{expert_num}_Prediction'] = expert_data[j]['mu'][i, 0]
                    result[f'Expert{expert_num}_Confidence_nu'] = expert_data[j]['v'][i, 0]
                    result[f'Expert{expert_num}_Weight'] = weights[j]
                    result[f'Expert{expert_num}_Epistemic'] = expert_uncertainties[j]['epistemic']
                    result[f'Expert{expert_num}_Aleatoric'] = expert_uncertainties[j]['aleatoric']
                
                # MoNIG aggregated
                result['MoNIG_Prediction'] = mu_agg[i, 0]
                result['MoNIG_Epistemic'] = epistemic_val
                result['MoNIG_Aleatoric'] = aleatoric_val
                
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
    parser.add_argument('--calibrator_path', type=str, default=None,
                        help='Optional isotonic calibrator (.pkl). Defaults to model_path-derived file')
    
    # Data split
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test', 'casf2016'],
                       help='Which split to run inference on (test=internal test set, casf2016=CASF 2016 benchmark)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    
    # Model config (must match training)
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension (must match training)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (must match training)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Validate and adjust device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    # Set seed for reproducibility
    set_seed(0)
    
    # Load CASF 2016 PDB IDs from ground truth file
    casf2016_pdb_ids = None
    casf2016_file = 'data/coreset_dirs.csv'
    if os.path.isfile(casf2016_file):
        # Read from file (one PDB ID per line, skip header if present)
        with open(casf2016_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            # Skip header if it looks like a header (contains "directory" or similar)
            if len(lines) > 0 and ('directory' in lines[0].lower() or 'pdb' in lines[0].lower()):
                lines = lines[1:]
            casf2016_pdb_ids = lines
        print(f"Loaded {len(casf2016_pdb_ids)} CASF 2016 PDB IDs from {casf2016_file}")
    else:
        print(f"Warning: CASF 2016 file not found at {casf2016_file}, proceeding without exclusion")
    
    print("="*70)
    print("MoNIG Inference - Expert Analysis")
    print("="*70)
    print(f"Random seed: 42")
    print(f"Model: {args.model_path}")
    print(f"Input CSV: {args.csv_path}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_path}")
    print(f"Device: {args.device}")
    if casf2016_pdb_ids:
        print(f"CASF 2016 excluded: {len(casf2016_pdb_ids)} complexes")
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
    
    if args.calibrator_path is not None:
        calibrator_path = args.calibrator_path
    else:
        base, _ = os.path.splitext(args.model_path)
        calibrator_path = base + '_calibrator.pkl'
    iso_model = None
    if os.path.exists(calibrator_path):
        with open(calibrator_path, 'rb') as f:
            data = pickle.load(f)
            iso_model = data.get('iso_model')
        print(f"Loaded isotonic calibrator from {calibrator_path}")
    else:
        print("No calibrator file found; proceeding without recalibration.")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = DrugDiscoveryDatasetEmb(
        args.csv_path, split=args.split, normalization_stats=norm_stats,
        casf2016_pdb_ids=casf2016_pdb_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    embedding_dim, num_experts = dataset.get_dim()
    print(f"Embeddings: {embedding_dim}, Experts: {num_experts}")
    
    # Create model
    print("\nCreating model...")
    class HyperParams:
        pass
    
    hyp_params = HyperParams()
    hyp_params.num_experts = num_experts
    hyp_params.embedding_dim = embedding_dim
    hyp_params.hidden_dim = args.hidden_dim
    hyp_params.dropout = args.dropout
    
    model = DrugDiscoveryMoNIGEmb(hyp_params)
    
    # Load weights
    print(f"Loading model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    print(f"Model loaded successfully!")
    
    # Run inference
    print("\nRunning inference...")
    df_results = inference_monig(model, loader, args.device, iso_model=iso_model)
    
    # Save results
    print(f"\nSaving results to {args.output_path}...")
    df_results.to_csv(args.output_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("INFERENCE SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df_results)}")
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
    
    print("="*70)
    print(f"✓ Inference complete! Results saved to: {args.output_path}")
    print("="*70)


if __name__ == '__main__':
    main()

