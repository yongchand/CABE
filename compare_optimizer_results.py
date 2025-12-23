#!/usr/bin/env python3
"""
Compare results from different optimizers (Adam vs SLSQP vs SGD).

Usage:
    python compare_optimizer_results.py experiments_ablation/ablation_results_*.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_merge_results(*csv_files):
    """Load and merge multiple result CSV files"""
    dfs = []
    for csv_file in csv_files:
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid CSV files found")
    
    return pd.concat(dfs, ignore_index=True)


def compute_summary_stats(df, group_by=['ablation_type', 'optimizer']):
    """Compute summary statistics grouped by ablation type and optimizer"""
    metrics = ['test_mae', 'test_rmse', 'test_corr', 'test_r2', 
               'test_crps', 'test_nll', 'test_picp_95', 'test_ece']
    
    summary = {}
    for metric in metrics:
        if metric in df.columns:
            grouped = df.groupby(group_by)[metric].agg(['mean', 'std', 'count'])
            summary[metric] = grouped
    
    return summary


def print_comparison_table(df):
    """Print comparison table of optimizers"""
    print("\n" + "="*80)
    print("OPTIMIZER COMPARISON")
    print("="*80)
    
    summary = compute_summary_stats(df)
    
    for metric_name, metric_df in summary.items():
        if metric_df.empty:
            continue
        
        print(f"\n{metric_name.upper()}")
        print("-" * 80)
        print(metric_df.to_string())
        print()


def plot_optimizer_comparison(df, output_dir='optimizer_comparison_plots'):
    """Create comparison plots for different optimizers"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics to plot
    metrics = {
        'test_mae': 'Test MAE (↓)',
        'test_rmse': 'Test RMSE (↓)',
        'test_corr': 'Test Correlation (↑)',
        'test_r2': 'Test R² (↑)',
        'test_crps': 'Test CRPS (↓)',
        'test_nll': 'Test NLL (↓)',
        'test_picp_95': 'Test PICP@95% (target: 0.95)',
        'test_ece': 'Test ECE (↓)'
    }
    
    # Filter successful runs
    df_success = df[df['success'] == True].copy()
    
    if df_success.empty:
        print("Warning: No successful runs found for plotting")
        return
    
    # 1. Bar plot comparison by optimizer
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(metrics.items()):
        if metric not in df_success.columns or idx >= len(axes):
            continue
        
        ax = axes[idx]
        
        # Group by optimizer and compute mean/std
        grouped = df_success.groupby('optimizer')[metric].agg(['mean', 'std']).reset_index()
        
        # Plot
        x = np.arange(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['optimizer'])
        ax.set_ylabel(label)
        ax.set_title(f'{label}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_comparison_bars.png', dpi=150, bbox_inches='tight')
    print(f"Saved bar plot: {output_dir / 'optimizer_comparison_bars.png'}")
    plt.close()
    
    # 2. Box plot comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(metrics.items()):
        if metric not in df_success.columns or idx >= len(axes):
            continue
        
        ax = axes[idx]
        
        # Create box plot
        df_success.boxplot(column=metric, by='optimizer', ax=ax)
        ax.set_title(label)
        ax.set_xlabel('Optimizer')
        ax.set_ylabel(label)
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    plt.suptitle('Optimizer Comparison - Box Plots')
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_comparison_boxes.png', dpi=150, bbox_inches='tight')
    print(f"Saved box plot: {output_dir / 'optimizer_comparison_boxes.png'}")
    plt.close()
    
    # 3. Scatter plot: MAE vs CRPS colored by optimizer
    if 'test_mae' in df_success.columns and 'test_crps' in df_success.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for optimizer in df_success['optimizer'].unique():
            df_opt = df_success[df_success['optimizer'] == optimizer]
            ax.scatter(df_opt['test_mae'], df_opt['test_crps'], 
                      label=optimizer, alpha=0.6, s=100)
        
        ax.set_xlabel('Test MAE (↓)')
        ax.set_ylabel('Test CRPS (↓)')
        ax.set_title('MAE vs CRPS by Optimizer')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mae_vs_crps_by_optimizer.png', dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot: {output_dir / 'mae_vs_crps_by_optimizer.png'}")
        plt.close()
    
    print(f"\nAll plots saved to: {output_dir}/")


def perform_statistical_test(df):
    """Perform statistical tests to compare optimizers"""
    from scipy import stats
    
    print("\n" + "="*80)
    print("STATISTICAL TESTS (Paired t-test)")
    print("="*80)
    
    metrics = ['test_mae', 'test_rmse', 'test_corr', 'test_r2']
    optimizers = df['optimizer'].unique()
    
    if len(optimizers) < 2:
        print("Need at least 2 optimizers to compare")
        return
    
    # Perform pairwise comparisons
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        print(f"\n{metric.upper()}:")
        print("-" * 40)
        
        for i, opt1 in enumerate(optimizers):
            for opt2 in optimizers[i+1:]:
                # Get values for each optimizer (matched by seed and ablation_type)
                df1 = df[df['optimizer'] == opt1]
                df2 = df[df['optimizer'] == opt2]
                
                # Merge on seed and ablation_type
                merged = df1.merge(df2, on=['seed', 'ablation_type'], 
                                  suffixes=('_1', '_2'))
                
                if len(merged) < 2:
                    continue
                
                vals1 = merged[f'{metric}_1'].dropna()
                vals2 = merged[f'{metric}_2'].dropna()
                
                if len(vals1) > 0 and len(vals2) > 0:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(vals1, vals2)
                    mean_diff = vals1.mean() - vals2.mean()
                    
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"  {opt1} vs {opt2}:")
                    print(f"    Mean diff: {mean_diff:+.4f}, t={t_stat:.3f}, p={p_value:.4f} {sig}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare optimizer results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('csv_files', nargs='+', help='CSV result files to compare')
    parser.add_argument('--output_dir', type=str, default='optimizer_comparison_plots',
                       help='Output directory for plots')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no_stats', action='store_true',
                       help='Skip statistical tests')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    df = load_and_merge_results(*args.csv_files)
    print(f"Loaded {len(df)} experiments")
    
    # Print comparison table
    print_comparison_table(df)
    
    # Statistical tests
    if not args.no_stats:
        perform_statistical_test(df)
    
    # Generate plots
    if not args.no_plots:
        plot_optimizer_comparison(df, args.output_dir)
    
    # Save combined results
    output_file = Path(args.output_dir) / 'combined_optimizer_results.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nCombined results saved to: {output_file}")


if __name__ == '__main__':
    main()

