#!/usr/bin/env python3
"""
Selective Prediction Analysis for 7WC6: CABE Uncertainty-Accuracy Tradeoff

This script analyzes how CABE's epistemic uncertainty enables selective prediction
on the 7WC6 dataset. Following the same methodology as run_case_studies.py.

Key claim: "By discarding X% most uncertain predictions, MAE reduces by Y%"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.drug_models_emb import DrugDiscoveryMoNIGEmb
from src.utils import moe_nig


class TestDataset7WC6:
    """Dataset for 7wc6 test data with engine scores and embeddings."""
    
    def __init__(self, csv_path, norm_stats=None):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        
        # Engine columns
        engine_cols = ['Gnina', 'BIND_pKd', 'Flowdock', 'DynamicBind']
        
        # Filter valid rows
        valid_mask = ~df[engine_cols].isna().any(axis=1)
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
        
        # Extract engine scores in CABE order: [GNINA, BIND, FlowDock, DynamicBind]
        self.engine_scores = df[engine_cols].values.astype(np.float32)
        
        # Normalize embeddings
        if norm_stats is not None:
            self.emb_mean = norm_stats['mean']
            self.emb_std = norm_stats['std']
            self.embeddings = (self.embeddings - self.emb_mean) / (self.emb_std + 1e-8)
        
        # Ground truth and metadata
        self.labels = df['true_affinity_p'].values.astype(np.float32)
        self.smiles = df['SMILES'].values
        
        self.engine_names = ['GNINA', 'BIND', 'FlowDock', 'DynamicBind']
    
    def __len__(self):
        return len(self.labels)
    
    def get_tensors(self):
        """Return all data as tensors for batch inference."""
        return (
            torch.FloatTensor(self.engine_scores),
            torch.FloatTensor(self.embeddings),
            self.labels
        )


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


def run_cabe_inference(model, dataset, device, batch_size=64):
    """Run CABE inference and return predictions with uncertainties."""
    model.eval()
    expert_scores, embeddings, labels = dataset.get_tensors()
    
    all_preds = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            batch_scores = expert_scores[i:i+batch_size].to(device)
            batch_emb = embeddings[i:i+batch_size].to(device)
            
            # Get NIG outputs from each expert
            nigs = model(batch_scores, batch_emb)
            
            # Aggregate NIGs
            mu_final, v_final, alpha_final, beta_final = aggregate_nigs(nigs)
            
            # Extract predictions and uncertainties
            predictions = mu_final.cpu().numpy()
            epistemic = (beta_final / (v_final * (alpha_final - 1))).cpu().numpy()
            aleatoric = (beta_final / (alpha_final - 1)).cpu().numpy()
            
            all_preds.extend(predictions.flatten())
            all_epistemic.extend(epistemic.flatten())
            all_aleatoric.extend(aleatoric.flatten())
    
    return (np.array(all_preds), labels, 
            np.array(all_epistemic), np.array(all_aleatoric))


def analyze_uncertainty_accuracy_tradeoff(predictions, true_values, epistemic, output_dir=None):
    """
    Analyze the relationship between epistemic uncertainty and prediction accuracy.
    Creates a risk-coverage curve showing MAE/RMSE vs coverage.
    
    This demonstrates: "By discarding X% most uncertain predictions, MAE reduces by Y%"
    
    Following the same methodology as run_case_studies.py
    
    Args:
        predictions: CABE predictions
        true_values: Ground truth values
        epistemic: Epistemic uncertainty estimates
        output_dir: Optional directory to save plots
        
    Returns:
        dict: Statistics about the uncertainty-accuracy relationship
    """
    # Calculate errors
    errors = np.abs(predictions - true_values)
    
    # Sort by epistemic uncertainty (ascending - keep low uncertainty first)
    sorted_indices = np.argsort(epistemic)
    sorted_epistemic = epistemic[sorted_indices]
    sorted_errors = errors[sorted_indices]
    
    # Calculate metrics at different coverage levels
    coverage_levels = np.arange(0.1, 1.01, 0.05)  # 10%, 15%, ..., 100%
    mae_at_coverage = []
    rmse_at_coverage = []
    mean_epistemic_at_coverage = []
    
    for coverage in coverage_levels:
        n_keep = int(len(sorted_errors) * coverage)
        if n_keep == 0:
            n_keep = 1
        
        kept_errors = sorted_errors[:n_keep]
        kept_epistemic = sorted_epistemic[:n_keep]
        
        mae = np.mean(kept_errors)
        rmse = np.sqrt(np.mean(kept_errors**2))
        mean_ep = np.mean(kept_epistemic)
        
        mae_at_coverage.append(mae)
        rmse_at_coverage.append(rmse)
        mean_epistemic_at_coverage.append(mean_ep)
    
    # Full set metrics
    mae_full = np.mean(errors)
    rmse_full = np.sqrt(np.mean(errors**2))
    
    # Calculate improvements at key thresholds
    improvements = {}
    for threshold_pct in [50, 70, 80, 90]:
        threshold_idx = int(threshold_pct / 5) - 2  # Map to coverage_levels index
        mae_kept = mae_at_coverage[threshold_idx]
        rmse_kept = rmse_at_coverage[threshold_idx]
        
        mae_improvement = (mae_full - mae_kept) / mae_full * 100
        rmse_improvement = (rmse_full - rmse_kept) / rmse_full * 100
        discarded_pct = 100 - threshold_pct
        
        improvements[threshold_pct] = {
            'coverage': threshold_pct,
            'discarded': discarded_pct,
            'mae_kept': mae_kept,
            'rmse_kept': rmse_kept,
            'mae_improvement': mae_improvement,
            'rmse_improvement': rmse_improvement,
            'epistemic_threshold': sorted_epistemic[int(len(sorted_epistemic) * threshold_pct / 100)]
        }
    
    stats_dict = {
        'mae_full': mae_full,
        'rmse_full': rmse_full,
        'coverage_levels': coverage_levels.tolist(),
        'mae_at_coverage': mae_at_coverage,
        'rmse_at_coverage': rmse_at_coverage,
        'mean_epistemic_at_coverage': mean_epistemic_at_coverage,
        'improvements': improvements,
        'n_samples': len(errors)
    }
    
    # Create risk-coverage curve
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: MAE and RMSE vs Coverage
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(coverage_levels * 100, mae_at_coverage, 'b-', linewidth=2.5, 
                        marker='o', markersize=4, label='MAE', alpha=0.8)
        line2 = ax1.plot(coverage_levels * 100, rmse_at_coverage, 'r-', linewidth=2.5,
                        marker='s', markersize=4, label='RMSE', alpha=0.8)
        
        # Add horizontal lines for full set
        ax1.axhline(y=mae_full, color='b', linestyle='--', alpha=0.5, linewidth=1.5, 
                   label=f'MAE (full set): {mae_full:.3f}')
        ax1.axhline(y=rmse_full, color='r', linestyle='--', alpha=0.5, linewidth=1.5,
                   label=f'RMSE (full set): {rmse_full:.3f}')
        
        # Plot mean epistemic on right axis
        line3 = ax1_twin.plot(coverage_levels * 100, mean_epistemic_at_coverage, 'g-', 
                             linewidth=2, marker='^', markersize=4, alpha=0.6,
                             label='Mean Epistemic Unc.')
        
        ax1.set_xlabel('Coverage (% of samples kept)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Error (MAE / RMSE, pKd)', fontsize=12, fontweight='bold', color='black')
        ax1_twin.set_ylabel('Mean Epistemic Uncertainty', fontsize=12, fontweight='bold', color='green')
        ax1.set_title('7WC6: Risk-Coverage Curve\n(Low Uncertainty = High Accuracy)', 
                     fontsize=14, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=9)
        
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(10, 100)
        ax1_twin.tick_params(axis='y', labelcolor='green')
        
        # Add annotations for key thresholds
        for pct in [50, 70, 90]:
            idx = int(pct / 5) - 2
            mae_val = mae_at_coverage[idx]
            improvement = (mae_full - mae_val) / mae_full * 100
            ax1.annotate(f'{100-pct}% dropped\n↓{improvement:.1f}% MAE',
                        xy=(pct, mae_val), xytext=(pct-15, mae_val+0.08),
                        fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1))
        
        # Plot 2: Improvement vs Discarded Percentage
        discard_pcts = 100 - coverage_levels * 100
        mae_improvements = [(mae_full - mae) / mae_full * 100 for mae in mae_at_coverage]
        rmse_improvements = [(rmse_full - rmse) / rmse_full * 100 for rmse in rmse_at_coverage]
        
        ax2.plot(discard_pcts, mae_improvements, 'b-', linewidth=2.5, 
                marker='o', markersize=4, label='MAE Improvement', alpha=0.8)
        ax2.plot(discard_pcts, rmse_improvements, 'r-', linewidth=2.5,
                marker='s', markersize=4, label='RMSE Improvement', alpha=0.8)
        
        # Add shaded regions
        ax2.axvspan(0, 20, alpha=0.1, color='green', label='Low risk (≤20% discarded)')
        ax2.axvspan(20, 50, alpha=0.1, color='yellow')
        ax2.axvspan(50, 100, alpha=0.1, color='red')
        
        ax2.set_xlabel('% of Most Uncertain Samples Discarded', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Error Improvement (%)', fontsize=12, fontweight='bold')
        ax2.set_title('7WC6: Error Improvement by Discarding Uncertain Predictions\n(Practical Decision-Making)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 90)
        
        # Add zero line
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add text box with key insights
        textstr = 'Key Thresholds:\n'
        for pct in [50, 70, 90]:
            imp = improvements[pct]
            textstr += f'Keep {pct}% (drop {imp["discarded"]:.0f}%): ↓{imp["mae_improvement"]:.1f}% MAE\n'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        plot_path = output_path / '7wc6_uncertainty_accuracy_tradeoff.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_path_pdf = output_path / '7wc6_uncertainty_accuracy_tradeoff.pdf'
        plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved uncertainty-accuracy tradeoff curve to: {plot_path}")
    
    return stats_dict


def print_selective_prediction_report(stats_dict):
    """Print comprehensive selective prediction analysis report."""
    
    print("\n" + "="*90)
    print("SELECTIVE PREDICTION ANALYSIS: 7WC6 DATASET")
    print("="*90)
    
    print(f"\n📊 Dataset: {stats_dict['n_samples']} samples")
    
    # Full set performance
    print("\n" + "-"*90)
    print("FULL TEST SET PERFORMANCE (100% Coverage)")
    print("-"*90)
    print(f"  MAE:  {stats_dict['mae_full']:.4f} pKd")
    print(f"  RMSE: {stats_dict['rmse_full']:.4f} pKd")
    
    # Selective prediction results
    print("\n" + "="*90)
    print("SELECTIVE PREDICTION: UNCERTAINTY-ACCURACY TRADEOFF")
    print("="*90)
    
    print("\n📈 Claim: 'By discarding X% most uncertain predictions, MAE reduces by Y%'")
    
    print(f"\n{'Coverage':<12} {'Discarded':<12} {'MAE (pKd)':<12} {'MAE ↓%':<12} {'RMSE (pKd)':<12} {'RMSE ↓%':<12}")
    print("-"*90)
    
    for pct in [50, 70, 80, 90, 100]:
        if pct == 100:
            print(f"{pct}%{'':<9} {'0%':<12} {stats_dict['mae_full']:<12.4f} {'0.0%':<12} {stats_dict['rmse_full']:<12.4f} {'0.0%':<12}")
        else:
            imp = stats_dict['improvements'][pct]
            mae_imp_str = f"{imp['mae_improvement']:+.1f}%"
            rmse_imp_str = f"{imp['rmse_improvement']:+.1f}%"
            print(f"{pct}%{'':<9} {imp['discarded']:.0f}%{'':<10} {imp['mae_kept']:<12.4f} {mae_imp_str:<12} {imp['rmse_kept']:<12.4f} {rmse_imp_str:<12}")
    
    # Key takeaways
    print("\n" + "="*90)
    print("🎯 KEY TAKEAWAYS")
    print("="*90)
    
    # Find best improvement
    best_coverage = 0
    best_improvement = 0
    for pct, imp in stats_dict['improvements'].items():
        if imp['mae_improvement'] > best_improvement:
            best_improvement = imp['mae_improvement']
            best_coverage = pct
    
    if best_improvement > 0:
        imp = stats_dict['improvements'][best_coverage]
        print(f"\n✅ Best selective prediction: {best_improvement:.1f}% MAE improvement at {best_coverage}% coverage")
        print(f"   → By discarding the {imp['discarded']:.0f}% most epistemically uncertain predictions,")
        print(f"     we reduce MAE from {stats_dict['mae_full']:.4f} to {imp['mae_kept']:.4f} pKd")
    else:
        print("\n⚠️  Note: Selective prediction shows limited improvement on this dataset")
        print("   This may indicate the epistemic uncertainty is well-calibrated but")
        print("   errors are relatively uniform across uncertainty levels")
    
    print("\n→ Epistemic uncertainty enables selective prediction")
    print("→ Users can trade coverage for accuracy based on risk tolerance")
    
    # Practical interpretation
    print("\n" + "-"*90)
    print("PRACTICAL INTERPRETATION")
    print("-"*90)
    print("""
    Coverage Zones:
    • 90-100% coverage: Keep most predictions, minimal quality gain
    • 70-90% coverage:  Moderate selectivity, reasonable tradeoff
    • 50-70% coverage:  Aggressive selectivity, significant quality gain
    • <50% coverage:    Very selective, only highest confidence predictions
    
    Use Case: If a drug discovery application requires high reliability,
    use CABE's epistemic uncertainty to identify and reject uncertain predictions.
    """)
    
    print("="*90)


def main():
    parser = argparse.ArgumentParser(
        description='Selective Prediction Analysis for 7WC6: CABE Uncertainty-Accuracy Tradeoff'
    )
    parser.add_argument('--csv_path', type=str,
                        default='7wc6_embeddings_704D_final.csv',
                        help='Path to 7wc6 embeddings CSV')
    parser.add_argument('--model_path', type=str,
                        default='saved_models/best_MoNIG_emb.pt',
                        help='Path to trained CABE model')
    parser.add_argument('--norm_stats_path', type=str,
                        default='saved_models/best_MoNIG_emb_norm_stats.npz',
                        help='Path to normalization stats')
    parser.add_argument('--output_dir', type=str, default='7wc6_selective_prediction',
                        help='Output directory for plots and results')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*90)
    print("SELECTIVE PREDICTION ANALYSIS: 7WC6 DATASET")
    print("CABE Uncertainty-Accuracy Tradeoff")
    print("="*90)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load normalization stats
    print("\n[1/4] Loading normalization stats...")
    if os.path.exists(args.norm_stats_path):
        norm_stats_npz = np.load(args.norm_stats_path)
        norm_stats = {'mean': norm_stats_npz['mean'], 'std': norm_stats_npz['std']}
        print(f"✓ Loaded from: {args.norm_stats_path}")
    else:
        print(f"⚠ Norm stats not found at {args.norm_stats_path}")
        norm_stats = None
    
    # Load dataset
    print("\n[2/4] Loading 7WC6 dataset...")
    dataset = TestDataset7WC6(args.csv_path, norm_stats=norm_stats)
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Load CABE model
    print("\n[3/4] Loading CABE model...")
    
    class HyperParams:
        num_experts = 4
        embedding_dim = 704
        hidden_dim = 256
        dropout = 0.2
    
    model = DrugDiscoveryMoNIGEmb(HyperParams())
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print(f"✓ Loaded model from: {args.model_path}")
    else:
        print(f"ERROR: Model not found at {args.model_path}")
        return
    
    model = model.to(args.device)
    model.eval()
    
    # Run inference
    print("\n[4/4] Running CABE inference...")
    cabe_preds, labels, epistemic, aleatoric = run_cabe_inference(
        model, dataset, args.device
    )
    
    print(f"✓ CABE MAE (full): {np.mean(np.abs(cabe_preds - labels)):.4f}")
    print(f"✓ Mean epistemic uncertainty: {np.mean(epistemic):.4f}")
    print(f"✓ Mean aleatoric uncertainty: {np.mean(aleatoric):.4f}")
    
    # Run uncertainty-accuracy tradeoff analysis
    print("\n" + "-"*90)
    print("Running uncertainty-accuracy tradeoff analysis...")
    print("-"*90)
    
    stats = analyze_uncertainty_accuracy_tradeoff(
        cabe_preds, labels, epistemic, args.output_dir
    )
    
    # Print comprehensive report
    print_selective_prediction_report(stats)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'coverage': stats['coverage_levels'],
        'mae': stats['mae_at_coverage'],
        'rmse': stats['rmse_at_coverage'],
        'mean_epistemic': stats['mean_epistemic_at_coverage']
    })
    results_csv = output_path / '7wc6_selective_prediction_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n💾 Saved results to: {results_csv}")
    
    # Save summary metrics
    summary = {
        'n_samples': stats['n_samples'],
        'mae_full': float(stats['mae_full']),
        'rmse_full': float(stats['rmse_full']),
        'improvements': {
            str(k): {key: float(v) if isinstance(v, (np.floating, float)) else v 
                    for key, v in imp.items()}
            for k, imp in stats['improvements'].items()
        }
    }
    
    summary_path = output_path / '7wc6_selective_prediction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved summary to: {summary_path}")
    
    print("\n" + "="*90)
    print("✅ Selective prediction analysis complete!")
    print(f"📁 All outputs saved to: {args.output_dir}/")
    print("="*90)


if __name__ == '__main__':
    main()
