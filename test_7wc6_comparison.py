#!/usr/bin/env python
"""
Test CABE model on 7wc6 dataset and compare with individual engines.
Skips rows with empty engine values.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.drug_models_emb import DrugDiscoveryMoNIGEmb
from src.utils import moe_nig


class TestDataset7WC6(Dataset):
    """Dataset for 7wc6 test data with engine scores and embeddings."""
    
    def __init__(self, csv_path, norm_stats=None):
        """
        Args:
            csv_path: Path to 7wc6_embeddings_704D_final.csv
            norm_stats: Dict with 'mean' and 'std' for embedding normalization
        """
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        
        # Engine columns in the CSV
        # CSV has: DynamicBind, Flowdock, Gnina, BIND_pKd
        # CABE expects: GNINA_Affinity, BIND_pIC50, flowdock_score, DynamicBind_score
        engine_cols = ['Gnina', 'BIND_pKd', 'Flowdock', 'DynamicBind']
        
        # Check for empty/NaN engine values and filter them out
        valid_mask = ~df[engine_cols].isna().any(axis=1)
        # Also check for empty strings
        for col in engine_cols:
            valid_mask &= df[col].astype(str).str.strip() != ''
        
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"Filtering out {n_invalid} rows with empty engine values")
        
        df = df[valid_mask].reset_index(drop=True)
        print(f"Using {len(df)} valid rows")
        
        self.df = df
        
        # Extract embeddings
        emb_cols = [col for col in df.columns if col.startswith('Emb_')]
        self.embeddings = df[emb_cols].values.astype(np.float32)
        print(f"Embeddings shape: {self.embeddings.shape}")
        
        # Extract engine scores in CABE order: [GNINA, BIND, FlowDock, DynamicBind]
        self.engine_scores = df[engine_cols].values.astype(np.float32)
        print(f"Engine scores shape: {self.engine_scores.shape}")
        
        # Normalize embeddings
        if norm_stats is not None:
            self.emb_mean = norm_stats['mean']
            self.emb_std = norm_stats['std']
            self.embeddings = (self.embeddings - self.emb_mean) / (self.emb_std + 1e-8)
            print("Applied normalization from training stats")
        else:
            print("WARNING: No normalization stats provided")
        
        # Ground truth
        self.labels = df['true_affinity_p'].values.astype(np.float32)
        self.smiles = df['SMILES'].values
        
        print(f"Label range: [{self.labels.min():.2f}, {self.labels.max():.2f}]")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.engine_scores[idx]),
            torch.FloatTensor(self.embeddings[idx])
        ), self.labels[idx], self.smiles[idx]
    
    def get_engine_scores_raw(self):
        """Return raw engine scores for individual engine evaluation."""
        return self.engine_scores
    
    def get_labels(self):
        return self.labels


def aggregate_nigs(nigs):
    """Aggregate multiple NIGs using moe_nig."""
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


def run_cabe_inference(model, loader, device):
    """Run CABE inference and return predictions with uncertainties."""
    model.eval()
    all_preds = []
    all_labels = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for (expert_scores, embeddings), labels, _ in loader:
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            
            # Get NIG outputs from each expert
            nigs = model(expert_scores, embeddings)
            
            # Aggregate NIGs
            mu_final, v_final, alpha_final, beta_final = aggregate_nigs(nigs)
            
            # Extract predictions and uncertainties
            predictions = mu_final.cpu().numpy()
            epistemic = (beta_final / (v_final * (alpha_final - 1))).cpu().numpy()
            aleatoric = (beta_final / (alpha_final - 1)).cpu().numpy()
            
            all_preds.extend(predictions.flatten())
            all_labels.extend(labels.numpy().flatten())
            all_epistemic.extend(epistemic.flatten())
            all_aleatoric.extend(aleatoric.flatten())
    
    return (np.array(all_preds), np.array(all_labels), 
            np.array(all_epistemic), np.array(all_aleatoric))


def compute_metrics(y_true, y_pred, name=""):
    """Compute regression metrics (same as run_multi_seed_experiments.py)."""
    # MAE
    mae = np.mean(np.abs(y_pred - y_true))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Correlation (Pearson)
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        'name': name,
        'MAE': mae,
        'RMSE': rmse,
        'Corr': corr,
        'R2': r2
    }


def main():
    parser = argparse.ArgumentParser(description='Test CABE on 7wc6 dataset')
    parser.add_argument('--csv_path', type=str, 
                        default='7wc6_embeddings_704D_final.csv',
                        help='Path to 7wc6 embeddings CSV')
    parser.add_argument('--experiment_dir', type=str,
                        default='experiments',
                        help='Directory containing seed experiments')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                        help='Seeds to test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_csv', type=str, default='7wc6_comparison_results.csv',
                        help='Output CSV for detailed results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CABE vs Individual Engines - 7wc6 Dataset (Multi-Seed)")
    print("="*80)
    print(f"Seeds: {args.seeds}")
    print(f"Experiment dir: {args.experiment_dir}")
    
    # Collect results across all seeds
    all_seed_results = []
    all_cabe_preds = []
    all_epistemic = []
    all_aleatoric = []
    labels = None
    engine_scores = None
    dataset = None
    
    for seed in args.seeds:
        seed_dir = os.path.join(args.experiment_dir, f"MoNIG_seed{seed}")
        model_path = os.path.join(seed_dir, "best_MoNIG_emb.pt")
        norm_stats_path = os.path.join(seed_dir, "best_MoNIG_emb_norm_stats.npz")
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Skipping seed {seed}: model not found at {model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing seed {seed}")
        print(f"{'='*60}")
        
        # Load normalization stats
        if os.path.exists(norm_stats_path):
            norm_stats_npz = np.load(norm_stats_path)
            norm_stats = {'mean': norm_stats_npz['mean'], 'std': norm_stats_npz['std']}
        else:
            print(f"  WARNING: Norm stats not found, using first available")
            norm_stats = None
        
        # Load dataset (only once, but re-normalize for each seed)
        if dataset is None or norm_stats is not None:
            dataset = TestDataset7WC6(args.csv_path, norm_stats=norm_stats)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            labels = dataset.get_labels()
            engine_scores = dataset.get_engine_scores_raw()
        
        # Load CABE model
        class HyperParams:
            num_experts = 4  # GNINA, BIND, FlowDock, DynamicBind
            embedding_dim = 704
            hidden_dim = 256
            dropout = 0.2
        
        model = DrugDiscoveryMoNIGEmb(HyperParams())
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model = model.to(args.device)
        model.eval()
        print(f"  ✅ Loaded model from: {model_path}")
        
        # Run inference
        cabe_preds, _, epistemic, aleatoric = run_cabe_inference(model, loader, args.device)
        
        # Store predictions
        all_cabe_preds.append(cabe_preds)
        all_epistemic.append(epistemic)
        all_aleatoric.append(aleatoric)
        
        # Compute metrics for this seed
        cabe_metrics = compute_metrics(labels, cabe_preds, f"CABE_seed{seed}")
        cabe_metrics['seed'] = seed
        cabe_metrics['Mean_Epistemic'] = np.mean(epistemic)
        cabe_metrics['Mean_Aleatoric'] = np.mean(aleatoric)
        all_seed_results.append(cabe_metrics)
        
        print(f"  MAE: {cabe_metrics['MAE']:.4f}, RMSE: {cabe_metrics['RMSE']:.4f}, "
              f"Corr: {cabe_metrics['Corr']:.4f}, R²: {cabe_metrics['R2']:.4f}")
    
    if len(all_seed_results) == 0:
        print("\n❌ No seeds found!")
        return
    
    # Compute aggregated CABE metrics (mean across seeds)
    all_cabe_preds = np.array(all_cabe_preds)
    all_epistemic = np.array(all_epistemic)
    all_aleatoric = np.array(all_aleatoric)
    
    mean_cabe_preds = np.mean(all_cabe_preds, axis=0)
    mean_epistemic = np.mean(all_epistemic, axis=0)
    mean_aleatoric = np.mean(all_aleatoric, axis=0)
    
    # Print aggregated results
    print("\n" + "="*80)
    print(f"AGGREGATED RESULTS ({len(all_seed_results)} seeds)")
    print("="*80)
    
    # Compute final metrics
    results = []
    
    # CABE aggregated metrics
    cabe_agg_metrics = compute_metrics(labels, mean_cabe_preds, "CABE (mean)")
    cabe_agg_metrics['Mean_Epistemic'] = np.mean(mean_epistemic)
    cabe_agg_metrics['Mean_Aleatoric'] = np.mean(mean_aleatoric)
    results.append(cabe_agg_metrics)
    
    # Individual engine metrics
    engine_names = ['GNINA', 'BIND', 'FlowDock', 'DynamicBind']
    for i, name in enumerate(engine_names):
        engine_preds = engine_scores[:, i]
        metrics = compute_metrics(labels, engine_preds, name)
        results.append(metrics)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print formatted table
    print(f"\n{'Method':<15} {'MAE':>8} {'RMSE':>8} {'Corr':>8} {'R²':>10}")
    print("-"*55)
    
    for _, row in results_df.iterrows():
        print(f"{row['name']:<15} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
              f"{row['Corr']:>8.4f} {row['R2']:>10.4f}")
    
    print("-"*55)
    print(f"N samples: {len(labels)}")
    
    # Print per-seed CABE statistics
    seed_df = pd.DataFrame(all_seed_results)
    print(f"\nCABE Per-Seed Statistics:")
    print(f"  MAE:  {seed_df['MAE'].mean():.4f} ± {seed_df['MAE'].std():.4f}")
    print(f"  RMSE: {seed_df['RMSE'].mean():.4f} ± {seed_df['RMSE'].std():.4f}")
    print(f"  Corr: {seed_df['Corr'].mean():.4f} ± {seed_df['Corr'].std():.4f}")
    print(f"  R²:   {seed_df['R2'].mean():.4f} ± {seed_df['R2'].std():.4f}")
    
    # Print CABE uncertainty info
    print(f"\nCABE Uncertainty (averaged across seeds):")
    print(f"  Mean Epistemic: {np.mean(mean_epistemic):.4f}")
    print(f"  Mean Aleatoric: {np.mean(mean_aleatoric):.4f}")
    print(f"  Mean Total Std: {np.mean(np.sqrt(mean_epistemic + mean_aleatoric)):.4f}")
    
    # Highlight best method
    print("\n" + "="*70)
    print("BEST METHODS BY METRIC")
    print("="*70)
    
    for metric in ['MAE', 'RMSE', 'Corr', 'R2']:
        if metric in ['MAE', 'RMSE']:
            best_idx = results_df[metric].idxmin()
        else:
            best_idx = results_df[metric].idxmax()
        best_name = results_df.loc[best_idx, 'name']
        best_val = results_df.loc[best_idx, metric]
        print(f"  {metric:<8}: {best_name:<15} ({best_val:.4f})")
    
    # Save summary
    summary_csv = '7wc6_4engines_comparison_summary.csv'
    results_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")
    
    # Save per-seed results
    seed_csv = '7wc6_4engines_per_seed_results.csv'
    seed_df.to_csv(seed_csv, index=False)
    print(f"Per-seed results saved to: {seed_csv}")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()

