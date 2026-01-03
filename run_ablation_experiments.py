#!/usr/bin/env python3
"""
Run ablation experiments for CABE (MoNIG) to test different components.

Ablations:
1. MoNIG_NoReliabilityScaling - Remove reliability scaling (r_j = 1.0) → show MoNIG collapses
2. MoNIG_UniformReliability - Use uniform reliability (r_j = 1/num_experts) → show reliability matters
3. MoNIG_NoContextReliability - Use per-expert learned reliability (no context dependence) → test context-dependent reliability
4. MoNIG_UniformWeightAggregation - Use uniform weight aggregation → test MoNIG aggregation necessity
5. MoNIG (baseline) - Full MoNIG with learned reliability

Usage:
    python run_ablation_experiments.py [--seeds 42 43 44 45 46] [--epochs 150]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import uncertainty_toolbox as uct

# Import helper functions from run_multi_seed_experiments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default seeds for experiments
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# All available ablation types
ALL_ABLATION_TYPES = [
    'MoNIG',  # Baseline
    'MoNIG_NoReliabilityScaling',
    'MoNIG_UniformReliability',
    'MoNIG_NoContextReliability',
    'MoNIG_UniformWeightAggregation',
    'MoNIG_Improved',  # Improved version with shallow reliability net + soft scaling (0.5+0.5*r)
    'MoNIG_Improved_v2',  # v2: More conservative scaling (0.7+0.3*r) for better RMSE
    'MoNIG_Hybrid',  # Hybrid: Blend learned + uniform reliability for balanced performance
    'MoNIG_Improved_Calibrated',  # Calibrated: MoNIG_Improved + uncertainty scaling for perfect PICP
    'MoNIG_Hybrid_Calibrated',  # Calibrated: MoNIG_Hybrid + uncertainty scaling for perfect PICP
]


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


def run_test_evaluation(model_type, model_path, csv_path, seed, device, output_dir):
    """
    Run inference on test set and extract metrics including PICP, ECE, and interval width.
    
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
        '--model_type', model_type,
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
            df = pd.read_csv(inference_output)
            
            # Extract metrics based on model type
            metrics = {
                'test_samples': len(df)
            }
            
            # Initialize variables
            y_pred = None
            y_true = None
            y_std = None
            
            # Detect model type and extract predictions/uncertainties
            if 'MoNIG' in model_type:
                # MoNIG has specific columns
                if 'MoNIG_Prediction' in df.columns and 'True_Affinity' in df.columns:
                    y_pred = df['MoNIG_Prediction'].values
                    y_true = df['True_Affinity'].values
                    
                    # Get uncertainty (std)
                    if 'MoNIG_Std' in df.columns:
                        y_std = df['MoNIG_Std'].values
                    elif 'MoNIG_Epistemic' in df.columns and 'MoNIG_Aleatoric' in df.columns:
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
                
                # Conformal prediction metrics if available
                has_conformal = 'Conformal_Lower' in df.columns and 'Conformal_Upper' in df.columns
                if has_conformal:
                    try:
                        conformal_lower = df['Conformal_Lower'].astype(str).str.strip('[]').astype(float).values
                        conformal_upper = df['Conformal_Upper'].astype(str).str.strip('[]').astype(float).values
                        
                        conformal_inside = (y_true >= conformal_lower) & (y_true <= conformal_upper)
                        conformal_picp = conformal_inside.mean()
                        
                        if 'Conformal_Width' in df.columns:
                            conformal_width = df['Conformal_Width'].astype(str).str.strip('[]').astype(float).mean()
                        else:
                            conformal_width = (conformal_upper - conformal_lower).mean()
                        
                        metrics['test_conformal_picp'] = conformal_picp
                        metrics['test_conformal_avg_width'] = conformal_width
                        
                        # Coverage error
                        target_coverage = 0.95
                        metrics['test_conformal_coverage_error'] = abs(conformal_picp - target_coverage)
                    except Exception as e:
                        print(f"    Warning: Could not parse conformal intervals: {e}")
            
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


def run_training(ablation_type, seed, csv_path, epochs, batch_size, hidden_dim, 
                dropout, lr, risk_weight, device, output_dir, optimizer='adam', 
                lbfgs_maxiter=20, lbfgs_history_size=10, lbfgs_ftol=1e-8,
                lbfgs_gtol=1e-6, lbfgs_maxls=20, lbfgs_maxfun=0):
    """
    Run training for a specific ablation type and seed.
    
    Returns:
        dict: Results dictionary with success status and output paths
    """
    print(f"\n{'='*80}")
    print(f"Training {ablation_type} with seed {seed}, optimizer={optimizer}")
    print(f"{'='*80}")
    
    # Create output directory for this experiment
    exp_dir = Path(output_dir) / f"{ablation_type}_seed{seed}_opt{optimizer}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command - use ablation type as model_type
    cmd = [
        sys.executable,
        'main.py',
        'train',
        '--model_type', ablation_type,
        '--csv_path', csv_path,
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--hidden_dim', str(hidden_dim),
        '--dropout', str(dropout),
        '--lr', str(lr),
        '--optimizer', optimizer,
        '--lbfgs_maxiter', str(lbfgs_maxiter),
        '--lbfgs_history_size', str(lbfgs_history_size),
        '--lbfgs_ftol', str(lbfgs_ftol),
        '--lbfgs_gtol', str(lbfgs_gtol),
        '--lbfgs_maxls', str(lbfgs_maxls),
        '--lbfgs_maxfun', str(lbfgs_maxfun),
        '--risk_weight', str(risk_weight),
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
        
        # Check if model files were created
        model_path = Path('saved_models') / f'best_{ablation_type}_emb.pt'
        norm_stats_path = Path('saved_models') / f'best_{ablation_type}_emb_norm_stats.npz'
        
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
            'ablation_type': ablation_type,
            'seed': seed,
            'optimizer': optimizer,
            'log_file': str(log_file),
            'model_path': str(model_path) if success else None,
            'exp_dir': str(exp_dir)
        }
        
        # Run test evaluation if training succeeded
        if success:
            test_results = run_test_evaluation(
                model_type=ablation_type,
                model_path=model_path,
                csv_path=csv_path,
                seed=seed,
                device=device,
                output_dir=exp_dir
            )
            result.update(test_results)
        
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with return code {e.returncode}")
        return {
            'success': False,
            'ablation_type': ablation_type,
            'seed': seed,
            'optimizer': optimizer,
            'log_file': str(log_file),
            'error': str(e)
        }


def save_results_to_csv(results, csv_file):
    """
    Save ablation experiment results to CSV file.
    
    Args:
        results: List of result dictionaries
        csv_file: Path to output CSV file
    """
    # Prepare data for CSV
    rows = []
    for r in results:
        row = {
            'ablation_type': r['ablation_type'],
            'seed': r['seed'],
            'optimizer': r.get('optimizer', 'adam'),
            'success': r['success'],
        }
        
        # Add test metrics if available
        test_metrics = ['test_mae', 'test_rmse', 'test_corr', 'test_r2',
                       'test_mean_epistemic', 'test_mean_aleatoric', 
                       'test_mean_total_uncertainty',
                       'test_picp_95', 'test_picp_90', 'test_ece',
                       'test_avg_interval_width_95',
                       'test_crps', 'test_nll',
                       'test_conformal_picp', 'test_conformal_avg_width',
                       'test_conformal_coverage_error', 'test_samples']
        
        for metric in test_metrics:
            row[metric] = r.get(metric, np.nan)
        
        # Add paths
        row['model_path'] = r.get('model_path', '')
        row['log_file'] = r.get('log_file', '')
        row['exp_dir'] = r.get('exp_dir', '')
        
        # Add error if failed
        if not r['success'] and 'error' in r:
            row['error'] = r['error']
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Reorder columns for readability
    column_order = ['ablation_type', 'seed', 'optimizer', 'success', 'test_samples',
                   'test_mae', 'test_rmse', 'test_corr', 'test_r2',
                   'test_crps', 'test_nll',
                   'test_picp_95', 'test_picp_90', 'test_ece',
                   'test_avg_interval_width_95',
                   'test_conformal_picp', 'test_conformal_avg_width',
                   'test_conformal_coverage_error',
                   'test_mean_epistemic', 'test_mean_aleatoric', 
                   'test_mean_total_uncertainty',
                   'model_path', 'log_file', 'exp_dir', 'error']
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Print summary statistics
    print("\nSummary Statistics by Ablation Type:")
    successful_results = [r for r in results if r['success']]
    if successful_results:
        successful_df = pd.DataFrame([r for r in successful_results])
        
        for ablation_type in successful_df['ablation_type'].unique():
            ablation_df = successful_df[successful_df['ablation_type'] == ablation_type]
            print(f"\n  {ablation_type}:")
            
            if 'test_mae' in ablation_df.columns:
                mae_values = ablation_df['test_mae'].dropna()
                if len(mae_values) > 0:
                    print(f"    MAE:  {mae_values.mean():.4f} ± {mae_values.std():.4f} (n={len(mae_values)})")
            
            if 'test_rmse' in ablation_df.columns:
                rmse_values = ablation_df['test_rmse'].dropna()
                if len(rmse_values) > 0:
                    print(f"    RMSE: {rmse_values.mean():.4f} ± {rmse_values.std():.4f} (n={len(rmse_values)})")
            
            if 'test_corr' in ablation_df.columns:
                corr_values = ablation_df['test_corr'].dropna()
                if len(corr_values) > 0:
                    print(f"    Corr: {corr_values.mean():.4f} ± {corr_values.std():.4f} (n={len(corr_values)})")
            
            if 'test_r2' in ablation_df.columns:
                r2_values = ablation_df['test_r2'].dropna()
                if len(r2_values) > 0:
                    print(f"    R²:   {r2_values.mean():.4f} ± {r2_values.std():.4f} (n={len(r2_values)})")
            
            # Proper scoring rules
            if 'test_crps' in ablation_df.columns:
                crps_values = ablation_df['test_crps'].dropna()
                if len(crps_values) > 0:
                    print(f"    CRPS: {crps_values.mean():.4f} ± {crps_values.std():.4f} (n={len(crps_values)})")
            
            if 'test_nll' in ablation_df.columns:
                nll_values = ablation_df['test_nll'].dropna()
                if len(nll_values) > 0:
                    print(f"    NLL:  {nll_values.mean():.4f} ± {nll_values.std():.4f} (n={len(nll_values)})")
            
            # Calibration metrics
            if 'test_picp_95' in ablation_df.columns:
                picp_values = ablation_df['test_picp_95'].dropna()
                if len(picp_values) > 0:
                    print(f"    PICP@95%: {picp_values.mean():.4f} ± {picp_values.std():.4f} (n={len(picp_values)})")
            
            if 'test_picp_90' in ablation_df.columns:
                picp_90_values = ablation_df['test_picp_90'].dropna()
                if len(picp_90_values) > 0:
                    print(f"    PICP@90%: {picp_90_values.mean():.4f} ± {picp_90_values.std():.4f} (n={len(picp_90_values)})")
            
            if 'test_ece' in ablation_df.columns:
                ece_values = ablation_df['test_ece'].dropna()
                if len(ece_values) > 0:
                    print(f"    ECE:  {ece_values.mean():.4f} ± {ece_values.std():.4f} (n={len(ece_values)})")
            
            if 'test_avg_interval_width_95' in ablation_df.columns:
                width_values = ablation_df['test_avg_interval_width_95'].dropna()
                if len(width_values) > 0:
                    print(f"    Avg Width@95%: {width_values.mean():.4f} ± {width_values.std():.4f} (n={len(width_values)})")
            
            # Conformal prediction metrics
            if 'test_conformal_picp' in ablation_df.columns:
                conformal_picp_values = ablation_df['test_conformal_picp'].dropna()
                if len(conformal_picp_values) > 0:
                    print(f"    Conformal PICP: {conformal_picp_values.mean():.4f} ± {conformal_picp_values.std():.4f} (n={len(conformal_picp_values)})")
            
            if 'test_conformal_avg_width' in ablation_df.columns:
                conformal_width_values = ablation_df['test_conformal_avg_width'].dropna()
                if len(conformal_width_values) > 0:
                    print(f"    Conformal Width: {conformal_width_values.mean():.4f} ± {conformal_width_values.std():.4f} (n={len(conformal_width_values)})")


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation experiments for CABE (MoNIG)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Ablation Types:
  1. MoNIG - Baseline (full MoNIG with learned reliability)
  2. MoNIG_NoReliabilityScaling - Remove reliability scaling (r_j = 1.0)
  3. MoNIG_UniformReliability - Use uniform reliability (r_j = 1/num_experts)
  4. MoNIG_NoContextReliability - Use per-expert learned reliability (no context dependence)
  5. MoNIG_UniformWeightAggregation - Use uniform weight aggregation instead of MoNIG

Examples:
  # Run all ablations with default seeds
  python run_ablation_experiments.py
  
  # Run specific ablations
  python run_ablation_experiments.py --ablation_types MoNIG_NoReliabilityScaling MoNIG_UniformReliability
        """
    )
    
    # Ablation selection
    parser.add_argument('--ablation_types', nargs='+', 
                       choices=ALL_ABLATION_TYPES + ['all'],
                       default=['all'],
                       help='Ablation types to test (default: all)')
    
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
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'lbfgs', 'lbfgs_torch', 'sgd'],
                       help='Optimizer to use (default: adam)')
    parser.add_argument('--lbfgs_maxiter', type=int, default=20,
                       help='Maximum iterations per step for L-BFGS optimizer (default: 20)')
    parser.add_argument('--lbfgs_history_size', type=int, default=10,
                       help='History size for L-BFGS (default: 10)')
    parser.add_argument('--lbfgs_ftol', type=float, default=1e-8,
                       help='Function tolerance for L-BFGS-B (default: 1e-8)')
    parser.add_argument('--lbfgs_gtol', type=float, default=1e-6,
                       help='Gradient tolerance for L-BFGS-B (default: 1e-6)')
    parser.add_argument('--lbfgs_maxls', type=int, default=20,
                       help='Maximum line search steps for L-BFGS-B (default: 20)')
    parser.add_argument('--lbfgs_maxfun', type=int, default=0,
                       help='Maximum function evaluations for L-BFGS-B (0=auto maxiter*10)')
    parser.add_argument('--risk_weight', type=float, default=0.001,
                       help='Risk regularization weight (for NIG/MoNIG)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='experiments_ablation',
                       help='Output directory for ablation experiment results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Determine which ablations to run
    if 'all' in args.ablation_types:
        ablation_types = ALL_ABLATION_TYPES
    else:
        ablation_types = args.ablation_types
    
    # Validate CSV path
    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary file
    summary_file = output_dir / f'ablation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    print("="*80)
    print("CABE ABLATION EXPERIMENT RUNNER")
    print("="*80)
    print(f"Ablation types: {', '.join(ablation_types)}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"CSV path: {args.csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Run experiments
    all_results = []
    total_experiments = len(ablation_types) * len(args.seeds)
    current_experiment = 0
    
    for ablation_type in ablation_types:
        for seed in args.seeds:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] ", end='')
            
            result = run_training(
                ablation_type=ablation_type,
                seed=seed,
                csv_path=args.csv_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                risk_weight=args.risk_weight,
                device=args.device,
                output_dir=output_dir,
                optimizer=args.optimizer,
                lbfgs_maxiter=getattr(args, 'lbfgs_maxiter', 20),
                lbfgs_history_size=getattr(args, 'lbfgs_history_size', 10),
                lbfgs_ftol=getattr(args, 'lbfgs_ftol', 1e-8),
                lbfgs_gtol=getattr(args, 'lbfgs_gtol', 1e-6),
                lbfgs_maxls=getattr(args, 'lbfgs_maxls', 20),
                lbfgs_maxfun=getattr(args, 'lbfgs_maxfun', 0),
            )
            
            all_results.append(result)
            
            # Save intermediate results
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': vars(args),
                    'results': all_results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION EXPERIMENT SUMMARY")
    print("="*80)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessful experiments:")
        for r in successful:
            print(f"  {r['ablation_type']} (seed {r['seed']}) -> {r['exp_dir']}")
    
    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  {r['ablation_type']} (seed {r['seed']})")
            if 'error' in r:
                print(f"    Error: {r['error']}")
            print(f"    Log: {r['log_file']}")
    
    # Summary by ablation type
    print("\nResults by ablation type:")
    for ablation_type in ablation_types:
        ablation_results = [r for r in all_results if r['ablation_type'] == ablation_type]
        ablation_successful = [r for r in ablation_results if r['success']]
        print(f"  {ablation_type}: {len(ablation_successful)}/{len(ablation_results)} successful")
    
    print(f"\n📄 Full results saved to: {summary_file}")
    
    # Save results to CSV
    csv_file = output_dir / f'ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    save_results_to_csv(all_results, csv_file)
    print(f"CSV results saved to: {csv_file}")
    print("="*80)
    
    # Exit with error code if any experiments failed
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
