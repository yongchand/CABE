#!/usr/bin/env python3
"""
Hyperparameter study for MoNIG risk_weight (lambda) parameter.

This script trains MoNIG models with different lambda values across multiple seeds
to find the optimal risk regularization weight.

Usage:
    python run_lambda_hyperparam_study.py --seeds 42 43 44 45 46 --epochs 150
    
    # Test with fewer seeds first
    python run_lambda_hyperparam_study.py --seeds 42 43 --epochs 100
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
import seaborn as sns


# Lambda values to test
LAMBDA_VALUES = [0.001, 0.001, 0.005, 0.01]

# Default seeds for experiments
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def compute_picp(y_pred, y_std, y_true, coverage=0.95):
    """Prediction interval coverage probability for symmetric intervals."""
    z = norm.ppf(0.5 + coverage / 2.0)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def compute_interval_ece(y_pred, y_std, y_true):
    """Expected calibration error from interval coverage."""
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
        y_pred, y_std, y_true, prop_type='interval'
    )
    return float(np.mean(np.abs(exp_props - obs_props)))


def compute_metrics_from_inference_csv(inference_csv_path):
    """
    Compute test metrics from an existing inference CSV file for MoNIG.
    
    Args:
        inference_csv_path: Path to inference results CSV
    
    Returns:
        dict: Test metrics dictionary
    """
    metrics = {}
    
    if not Path(inference_csv_path).exists():
        return metrics
    
    try:
        df = pd.read_csv(inference_csv_path)
        
        y_pred = None
        y_true = None
        y_std = None
        
        # MoNIG has specific columns
        if 'MoNIG_Prediction' in df.columns and 'True_Affinity' in df.columns:
            y_pred = df['MoNIG_Prediction'].values
            y_true = df['True_Affinity'].values
            
            # Compute total std from epistemic + aleatoric
            if 'MoNIG_Epistemic' in df.columns and 'MoNIG_Aleatoric' in df.columns:
                y_std = np.sqrt(df['MoNIG_Epistemic'].values + df['MoNIG_Aleatoric'].values)
            
            metrics['test_mae'] = np.mean(np.abs(y_pred - y_true))
            metrics['test_rmse'] = np.sqrt(np.mean((y_pred - y_true) ** 2))
            metrics['test_corr'] = np.corrcoef(y_pred, y_true)[0, 1]
            
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['test_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            # Uncertainty metrics
            if 'MoNIG_Epistemic' in df.columns:
                metrics['test_mean_epistemic'] = df['MoNIG_Epistemic'].mean()
            if 'MoNIG_Aleatoric' in df.columns:
                metrics['test_mean_aleatoric'] = df['MoNIG_Aleatoric'].mean()
            if 'MoNIG_Epistemic' in df.columns and 'MoNIG_Aleatoric' in df.columns:
                metrics['test_mean_total_uncertainty'] = (df['MoNIG_Epistemic'] + df['MoNIG_Aleatoric']).mean()
        
        # Compute PICP, ECE, and interval width if uncertainty is available
        if y_pred is not None and y_true is not None and y_std is not None and len(y_std) > 0 and np.all(y_std > 0):
            # Standard 95% intervals
            picp_95 = compute_picp(y_pred, y_std, y_true, coverage=0.95)
            picp_90 = compute_picp(y_pred, y_std, y_true, coverage=0.90)
            ece = compute_interval_ece(y_pred, y_std, y_true)
            
            # Interval width for 95% intervals
            z_95 = norm.ppf(0.975)
            interval_width_95 = (2 * z_95 * y_std).mean()
            
            metrics['test_picp_95'] = picp_95
            metrics['test_picp_90'] = picp_90
            metrics['test_ece'] = ece
            metrics['test_avg_interval_width_95'] = interval_width_95
            
            # Compute CRPS and NLL using uncertainty_toolbox
            try:
                uq_metrics = uct.metrics.get_all_metrics(y_pred, y_std, y_true, verbose=False)
                scoring_metrics = uq_metrics.get('scoring_rule', {})
                metrics['test_crps'] = scoring_metrics.get('crps', np.nan)
                metrics['test_nll'] = scoring_metrics.get('nll', np.nan)
            except Exception as e:
                print(f"    Warning: Could not compute CRPS/NLL: {e}")
                metrics['test_crps'] = np.nan
                metrics['test_nll'] = np.nan
        
        metrics['test_samples'] = len(df)
        
    except Exception as e:
        print(f"    Error computing metrics from CSV: {e}")
    
    return metrics


def run_test_evaluation(model_path, csv_path, seed, device, output_dir):
    """
    Run inference on test set and extract metrics.
    
    Returns:
        dict: Test metrics dictionary
    """
    print("  Running test evaluation...")
    
    # Create inference output path
    inference_output = Path(output_dir) / 'test_inference_results.csv'
    
    # Build inference command
    cmd = [
        sys.executable,
        'main.py',
        'infer',
        '--model_path', str(model_path),
        '--csv_path', csv_path,
        '--split', 'test',
        '--output_path', str(inference_output),
        '--seed', str(seed),
        '--device', device,
    ]
    
    try:
        # Run inference
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Parse results from CSV
        if inference_output.exists():
            metrics = compute_metrics_from_inference_csv(str(inference_output))
            metrics['inference_output'] = str(inference_output)
            return metrics
        else:
            print(f"    Warning: Inference output not found: {inference_output}")
            return {}
            
    except subprocess.CalledProcessError as e:
        print(f"    Warning: Test evaluation failed: {e}")
        return {}
    except Exception as e:
        print(f"    Warning: Error parsing test results: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_training(lambda_val, seed, csv_path, epochs, batch_size, hidden_dim,
                dropout, lr, device, output_dir):
    """
    Run training for MoNIG with a specific lambda value and seed.

    Returns:
        dict: Results dictionary with success status, output paths, and runtime
    """
    print(f"\n{'='*80}")
    print(f"Training MoNIG with lambda={lambda_val}, seed={seed}")
    print(f"{'='*80}")

    # Start timing
    start_time = time.time()

    # Create output directory for this experiment
    exp_dir = Path(output_dir) / f"MoNIG_lambda{lambda_val}_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,
        'main.py',
        'train',
        '--model_type', 'MoNIG',
        '--csv_path', csv_path,
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--hidden_dim', str(hidden_dim),
        '--dropout', str(dropout),
        '--lr', str(lr),
        '--risk_weight', str(lambda_val),
        '--device', device,
    ]

    # Run training
    log_file = exp_dir / 'training.log'
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        # Calculate training time
        training_time = time.time() - start_time

        # Check if model files were created
        model_path = Path('saved_models') / 'best_MoNIG_emb.pt'
        norm_stats_path = Path('saved_models') / 'best_MoNIG_emb_norm_stats.npz'

        success = model_path.exists() and norm_stats_path.exists()

        if success:
            # Copy model files to experiment directory
            import shutil
            if model_path.exists():
                shutil.copy(model_path, exp_dir / model_path.name)
            if norm_stats_path.exists():
                shutil.copy(norm_stats_path, exp_dir / norm_stats_path.name)

        result = {
            'success': success,
            'lambda': lambda_val,
            'seed': seed,
            'log_file': str(log_file),
            'model_path': str(model_path) if success else None,
            'exp_dir': str(exp_dir),
            'training_time_sec': training_time,
            'training_time_min': training_time / 60.0,
        }

        print(f"  Training completed in {training_time:.1f}s ({training_time/60:.2f} min)")

        # Run test evaluation if training succeeded
        inference_start = time.time()
        if success:
            test_results = run_test_evaluation(
                model_path=model_path,
                csv_path=csv_path,
                seed=seed,
                device=device,
                output_dir=exp_dir
            )
            result.update(test_results)
            inference_time = time.time() - inference_start
            result['inference_time_sec'] = inference_time
            print(f"  Inference completed in {inference_time:.1f}s")

        # Total time including inference
        total_time = time.time() - start_time
        result['total_time_sec'] = total_time
        result['total_time_min'] = total_time / 60.0

        return result
    except subprocess.CalledProcessError as e:
        total_time = time.time() - start_time
        print(f"ERROR: Training failed with return code {e.returncode} after {total_time:.1f}s")
        return {
            'success': False,
            'lambda': lambda_val,
            'seed': seed,
            'log_file': str(log_file),
            'error': str(e),
            'total_time_sec': total_time,
            'total_time_min': total_time / 60.0,
        }


def create_comparison_plots(results_df, output_dir):
    """Create comparison plots for the hyperparameter study."""
    
    output_dir = Path(output_dir)
    
    # Filter successful results
    df = results_df[results_df['success'] == True].copy()
    
    if len(df) == 0:
        print("No successful results to plot")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MoNIG Lambda (risk_weight) Hyperparameter Study', fontsize=14, fontweight='bold')
    
    # Metrics to plot
    metrics_to_plot = [
        ('test_mae', 'MAE ↓', axes[0, 0]),
        ('test_rmse', 'RMSE ↓', axes[0, 1]),
        ('test_corr', 'Correlation ↑', axes[0, 2]),
        ('test_picp_95', 'PICP@95% (target=0.95)', axes[1, 0]),
        ('test_ece', 'ECE ↓', axes[1, 1]),
        ('test_nll', 'NLL ↓', axes[1, 2]),
    ]
    
    lambda_values = sorted(df['lambda'].unique())
    
    for metric, title, ax in metrics_to_plot:
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        
        means = []
        stds = []
        for lam in lambda_values:
            lam_data = df[df['lambda'] == lam][metric].dropna()
            means.append(lam_data.mean())
            stds.append(lam_data.std())
        
        # Plot with error bars
        x_pos = range(len(lambda_values))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                      color='steelblue', edgecolor='navy')
        
        ax.set_xlabel('Lambda (risk_weight)')
        ax.set_ylabel(metric.replace('test_', '').replace('_', ' ').upper())
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(lam) for lam in lambda_values])
        
        # Add PICP target line
        if metric == 'test_picp_95':
            ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Target (0.95)')
            ax.legend()
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            if not np.isnan(mean):
                ax.annotate(f'{mean:.4f}', xy=(i, mean), ha='center', va='bottom',
                           fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lambda_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'lambda_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison plots to {output_dir / 'lambda_comparison.pdf'}")
    
    # Create detailed box plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MoNIG Lambda Study - Distribution Analysis', fontsize=14, fontweight='bold')
    
    box_metrics = [
        ('test_mae', 'MAE Distribution', axes[0, 0]),
        ('test_picp_95', 'PICP@95% Distribution', axes[0, 1]),
        ('test_ece', 'ECE Distribution', axes[1, 0]),
        ('test_nll', 'NLL Distribution', axes[1, 1]),
    ]
    
    for metric, title, ax in box_metrics:
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        
        data_to_plot = []
        labels = []
        for lam in lambda_values:
            lam_data = df[df['lambda'] == lam][metric].dropna().values
            if len(lam_data) > 0:
                data_to_plot.append(lam_data)
                labels.append(str(lam))
        
        if len(data_to_plot) > 0:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightsteelblue')
                patch.set_edgecolor('navy')
            
            ax.set_xlabel('Lambda (risk_weight)')
            ax.set_ylabel(metric.replace('test_', '').replace('_', ' ').upper())
            ax.set_title(title)
            
            if metric == 'test_picp_95':
                ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lambda_boxplots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'lambda_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved box plots to {output_dir / 'lambda_boxplots.pdf'}")


def save_results_to_csv(results, csv_file):
    """Save experiment results to CSV file."""
    rows = []
    for r in results:
        row = {
            'lambda': r['lambda'],
            'seed': r['seed'],
            'success': r['success'],
        }

        # Add runtime metrics
        runtime_metrics = ['training_time_sec', 'training_time_min',
                          'inference_time_sec', 'total_time_sec', 'total_time_min']
        for metric in runtime_metrics:
            row[metric] = r.get(metric, np.nan)

        # Add test metrics
        test_metrics = ['test_mae', 'test_rmse', 'test_corr', 'test_r2',
                       'test_mean_epistemic', 'test_mean_aleatoric',
                       'test_mean_total_uncertainty', 'test_samples',
                       'test_picp_95', 'test_picp_90', 'test_ece',
                       'test_avg_interval_width_95',
                       'test_crps', 'test_nll']

        for metric in test_metrics:
            row[metric] = r.get(metric, np.nan)

        # Add paths
        row['model_path'] = r.get('model_path', '')
        row['log_file'] = r.get('log_file', '')
        row['exp_dir'] = r.get('exp_dir', '')

        if not r['success'] and 'error' in r:
            row['error'] = r['error']

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    return df


def print_summary_statistics(df, lambda_values):
    """Print summary statistics for each lambda value."""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY LAMBDA VALUE")
    print("="*80)
    
    successful_df = df[df['success'] == True]
    
    if len(successful_df) == 0:
        print("No successful experiments to summarize.")
        return
    
    # Create summary table
    summary_rows = []
    
    for lam in lambda_values:
        lam_df = successful_df[successful_df['lambda'] == lam]
        if len(lam_df) == 0:
            continue
        
        row = {'lambda': lam, 'n_seeds': len(lam_df)}
        
        for metric in ['test_mae', 'test_rmse', 'test_corr', 'test_picp_95', 'test_ece', 'test_nll', 'test_crps']:
            if metric in lam_df.columns:
                values = lam_df[metric].dropna()
                if len(values) > 0:
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_std'] = values.std()
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Print formatted summary
    print(f"\n{'Lambda':<10} {'N':<4} {'MAE':^20} {'RMSE':^20} {'PICP@95%':^20} {'ECE':^20} {'NLL':^20}")
    print("-" * 114)
    
    for _, row in summary_df.iterrows():
        lam = row['lambda']
        n = row['n_seeds']
        
        mae_str = f"{row.get('test_mae_mean', np.nan):.4f}±{row.get('test_mae_std', np.nan):.4f}"
        rmse_str = f"{row.get('test_rmse_mean', np.nan):.4f}±{row.get('test_rmse_std', np.nan):.4f}"
        picp_str = f"{row.get('test_picp_95_mean', np.nan):.4f}±{row.get('test_picp_95_std', np.nan):.4f}"
        ece_str = f"{row.get('test_ece_mean', np.nan):.4f}±{row.get('test_ece_std', np.nan):.4f}"
        nll_str = f"{row.get('test_nll_mean', np.nan):.4f}±{row.get('test_nll_std', np.nan):.4f}"
        
        print(f"{lam:<10} {n:<4} {mae_str:^20} {rmse_str:^20} {picp_str:^20} {ece_str:^20} {nll_str:^20}")
    
    # Find best lambda for each metric
    print("\n" + "-" * 80)
    print("BEST LAMBDA VALUES:")
    print("-" * 80)
    
    for metric, direction, name in [
        ('test_mae_mean', 'min', 'MAE'),
        ('test_rmse_mean', 'min', 'RMSE'),
        ('test_corr_mean', 'max', 'Correlation'),
        ('test_ece_mean', 'min', 'ECE'),
        ('test_nll_mean', 'min', 'NLL'),
    ]:
        if metric in summary_df.columns:
            values = summary_df[metric].dropna()
            if len(values) > 0:
                if direction == 'min':
                    best_idx = values.idxmin()
                else:
                    best_idx = values.idxmax()
                best_lam = summary_df.loc[best_idx, 'lambda']
                best_val = summary_df.loc[best_idx, metric]
                print(f"  Best {name}: lambda={best_lam} (value={best_val:.4f})")
    
    # PICP closest to 0.95
    if 'test_picp_95_mean' in summary_df.columns:
        picp_diff = (summary_df['test_picp_95_mean'] - 0.95).abs()
        best_idx = picp_diff.idxmin()
        best_lam = summary_df.loc[best_idx, 'lambda']
        best_val = summary_df.loc[best_idx, 'test_picp_95_mean']
        print(f"  Best PICP@95% (closest to 0.95): lambda={best_lam} (value={best_val:.4f})")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter study for MoNIG risk_weight (lambda)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_lambda_hyperparam_study.py
  
  # Run with custom seeds
  python run_lambda_hyperparam_study.py --seeds 42 43 44 45 46
  
  # Quick test run
  python run_lambda_hyperparam_study.py --seeds 42 43 --epochs 50
        """
    )
    
    # Lambda values
    parser.add_argument('--lambda_values', nargs='+', type=float, default=LAMBDA_VALUES,
                       help=f'Lambda values to test (default: {LAMBDA_VALUES})')
    
    # Seeds
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                       help=f'Random seeds to use (default: {DEFAULT_SEEDS})')
    
    # Data
    parser.add_argument('--csv_path', type=str,
                       default='pdbbind_descriptors_with_experts_and_binding.csv',
                       help='Path to CSV file')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='lambda_hyperparam_study',
                       help='Output directory for experiment results')
    
    # Options
    parser.add_argument('--recompute-stats', action='store_true',
                       help='Recompute statistics from existing inference results without retraining')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if results already exist')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Sort lambda values
    lambda_values = sorted(args.lambda_values)
    
    # Validate CSV path
    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f'lambda_study_summary_{timestamp}.json'
    
    print("="*80)
    print("MoNIG LAMBDA (risk_weight) HYPERPARAMETER STUDY")
    print("="*80)
    
    # Show mode
    if args.recompute_stats:
        print("MODE: Recompute statistics only (no training)")
    elif args.force_retrain:
        print("MODE: Force retrain all experiments")
    else:
        print("MODE: Train new experiments (skip existing)")
    
    print(f"Lambda values: {lambda_values}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"CSV path: {args.csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Run experiments
    all_results = []
    total_experiments = len(lambda_values) * len(args.seeds)
    current_experiment = 0
    
    for lambda_val in lambda_values:
        for seed in args.seeds:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] ", end='')
            
            # Check if experiment already exists
            exp_dir = output_dir / f"MoNIG_lambda{lambda_val}_seed{seed}"
            inference_output = exp_dir / 'test_inference_results.csv'
            model_path = exp_dir / 'best_MoNIG_emb.pt'
            
            # Decide whether to train or reuse existing results
            should_train = True
            if exp_dir.exists() and inference_output.exists():
                if args.recompute_stats:
                    print(f"Recomputing statistics for lambda={lambda_val}, seed={seed}")
                    should_train = False
                elif not args.force_retrain:
                    print(f"Using existing results for lambda={lambda_val}, seed={seed}")
                    should_train = False
            
            # Train or load existing results
            if should_train:
                result = run_training(
                    lambda_val=lambda_val,
                    seed=seed,
                    csv_path=args.csv_path,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    lr=args.lr,
                    device=args.device,
                    output_dir=output_dir
                )
            else:
                # Load existing results and recompute test metrics
                print(f"  Loading existing inference results from {inference_output}")
                result = {
                    'lambda': lambda_val,
                    'seed': seed,
                    'success': model_path.exists() and inference_output.exists(),
                    'model_path': str(model_path),
                    'log_file': str(exp_dir / 'training.log'),
                    'exp_dir': str(exp_dir)
                }
                
                # Recompute test metrics from existing inference results
                if inference_output.exists():
                    try:
                        test_metrics = compute_metrics_from_inference_csv(str(inference_output))
                        result.update(test_metrics)
                        mae_str = f"{test_metrics.get('test_mae', np.nan):.4f}" if 'test_mae' in test_metrics else 'N/A'
                        print(f"  ✓ Recomputed test metrics: MAE={mae_str}")
                    except Exception as e:
                        print(f"  Warning: Could not recompute test metrics: {e}")
                        result['success'] = False
                        result['error'] = str(e)
            
            all_results.append(result)
            
            # Save intermediate results
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': vars(args),
                    'results': all_results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
    
    # Save results to CSV
    csv_file = output_dir / f'lambda_study_results_{timestamp}.csv'
    results_df = save_results_to_csv(all_results, csv_file)
    print(f"\n📊 Results saved to: {csv_file}")
    
    # Print summary statistics
    summary_df = print_summary_statistics(results_df, lambda_values)
    
    # Save summary statistics
    if summary_df is not None:
        summary_csv = output_dir / f'lambda_study_summary_{timestamp}.csv'
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n📋 Summary statistics saved to: {summary_csv}")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(results_df, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  lambda={r['lambda']}, seed={r['seed']}")
            if 'error' in r:
                print(f"    Error: {r['error']}")
    
    print(f"\n📄 Full results: {summary_file}")
    print(f"📊 CSV results: {csv_file}")
    print(f"📈 Plots: {output_dir / 'lambda_comparison.pdf'}")
    print("="*80)
    
    # Exit with error code if any experiments failed
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()

