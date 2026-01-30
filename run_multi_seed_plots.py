#!/usr/bin/env python3
"""
Generate publication-quality plots aggregated across multiple seeds.
Specifically for epistemic_vs_disagreement and disagreement_vs_error plots.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

EXPERT_NAMES = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']


def load_and_merge_seed_data(experiment_dir, model_name, seeds, original_csv_path):
    """Load inference results from multiple seeds and merge with original experts."""
    all_data = []
    
    for seed in seeds:
        csv_path = Path(experiment_dir) / f"{model_name}_seed{seed}" / "test_inference_results.csv"
        if not csv_path.exists():
            print(f"  ⚠️  Skipping seed {seed}: file not found")
            continue
        
        df = pd.read_csv(csv_path)
        
        # Merge with original expert predictions
        original_df = pd.read_csv(original_csv_path)
        expert_cols = ['ComplexID', 'GNINA_Affinity', 'BIND_pIC50', 'flowdock_score', 'DynamicBind_score']
        original_subset = original_df[expert_cols].copy()
        df = df.merge(original_subset, on='ComplexID', how='left', suffixes=('', '_orig'))
        
        # Calculate expert predictions
        expert_pred_cols = ['GNINA_Affinity', 'BIND_pIC50', 'flowdock_score', 'DynamicBind_score']
        for j, col in enumerate(expert_pred_cols):
            if col in df.columns:
                df[f'Expert{j+1}_Prediction'] = df[col]
        
        df['seed'] = seed
        all_data.append(df)
        print(f"  ✅ Loaded seed {seed}: {len(df)} samples")
    
    if len(all_data) == 0:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total samples across {len(all_data)} seeds: {len(combined_df)}")
    return combined_df


def plot_combined_disagreement_analysis(df, output_dir):
    """Plot epistemic vs disagreement AND disagreement vs error in one figure (2 panels)."""
    
    # Calculate expert disagreement
    expert_pred_cols = [col for col in df.columns if col.startswith('Expert') and col.endswith('_Prediction')]
    if len(expert_pred_cols) == 0:
        print("⚠️  No expert predictions found")
        return None
    
    df['Expert_Disagreement'] = df[expert_pred_cols].std(axis=1)
    
    # Calculate MoNIG error
    if 'MoNIG_Error' not in df.columns:
        df['MoNIG_Error'] = abs(df['MoNIG_Prediction'] - df['True_Affinity'])
    
    epistemic = df['MoNIG_Epistemic'].values
    disagreement = df['Expert_Disagreement'].values
    monig_error = df['MoNIG_Error'].values
    
    # Calculate correlations (Spearman)
    spearman_corr_ep, _ = stats.spearmanr(epistemic, disagreement)
    spearman_corr_err, _ = stats.spearmanr(disagreement, monig_error)
    
    # Publication-quality settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Epistemic vs Disagreement
    ax1.scatter(disagreement, epistemic, 
                c='#2E86AB', alpha=0.4, s=25, edgecolors='white', linewidth=0.2)
    
    z1 = np.polyfit(disagreement, epistemic, 1)
    p1 = np.poly1d(z1)
    x_trend1 = np.linspace(disagreement.min(), disagreement.max(), 100)
    y_trend1 = p1(x_trend1)
    ax1.plot(x_trend1, y_trend1, color='#E94F37', linestyle='--', linewidth=2.5,
             label=f'ρ = {spearman_corr_ep:.3f}')
    
    ax1.set_xlabel('Expert Disagreement (pKd)', fontsize=16)
    ax1.set_ylabel('RELIABLE-BA Epistemic Uncertainty', fontsize=16)
    ax1.legend(loc='upper left', fontsize=14, frameon=True, fancybox=False, 
               edgecolor='gray', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    
    # Panel 2: Disagreement vs Error
    ax2.scatter(disagreement, monig_error, 
                c='#2E86AB', alpha=0.4, s=25, edgecolors='white', linewidth=0.2)
    
    z2 = np.polyfit(disagreement, monig_error, 1)
    p2 = np.poly1d(z2)
    x_trend2 = np.linspace(disagreement.min(), disagreement.max(), 100)
    y_trend2 = p2(x_trend2)
    ax2.plot(x_trend2, y_trend2, color='#E94F37', linestyle='--', linewidth=2.5,
             label=f'ρ = {spearman_corr_err:.3f}')
    
    ax2.set_xlabel('Expert Disagreement (pKd)', fontsize=16)
    ax2.set_ylabel('RELIABLE-BA Prediction Error (pKd)', fontsize=16)
    ax2.legend(loc='upper left', fontsize=14, frameon=True, fancybox=False,
               edgecolor='gray', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save combined plot as PNG and PDF
    plot_path = output_path / 'disagreement_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plot_path_pdf = output_path / 'disagreement_analysis.pdf'
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"📊 Saved combined disagreement analysis to: {plot_path}")
    
    return {
        'epistemic_spearman': spearman_corr_ep,
        'error_spearman': spearman_corr_err,
        'num_samples': len(epistemic)
    }


def main():
    parser = argparse.ArgumentParser(description='Generate multi-seed plots')
    parser.add_argument('--experiment_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='MoNIG')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    parser.add_argument('--output_dir', type=str, default='risk_coverage_comparison')
    parser.add_argument('--original_csv', type=str, default='pdbbind_descriptors_with_experts_and_binding.csv')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTI-SEED PLOT GENERATION")
    print("="*80)
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Load data from all seeds
    print("\nLoading data from all seeds...")
    combined_df = load_and_merge_seed_data(
        args.experiment_dir, args.model, args.seeds, args.original_csv
    )
    
    if combined_df is None:
        print("❌ No data loaded!")
        return
    
    # Generate combined plot (both panels in one figure)
    print("\nGenerating combined disagreement analysis plot...")
    combined_stats = plot_combined_disagreement_analysis(combined_df, args.output_dir)
    if combined_stats:
        print(f"  Epistemic vs Disagreement: ρ = {combined_stats['epistemic_spearman']:.3f}")
        print(f"  Disagreement vs Error: ρ = {combined_stats['error_spearman']:.3f}")
        print(f"  N = {combined_stats['num_samples']}")
    
    print("\n✅ Multi-seed plots complete!")
    print("="*80)


if __name__ == '__main__':
    main()

