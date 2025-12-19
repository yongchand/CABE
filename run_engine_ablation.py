#!/usr/bin/env python3
"""
Run ablation experiments for CABE (MoNIG) with different engine combinations.

Engine Combinations:
1. MoNIG_2engines_dynamicbind_surfdock - 2 engines: DynamicBind, flowdock (surfdock)
2. MoNIG_3engines_dynamicbind_surfdock_gnina - 3 engines: DynamicBind, flowdock (surfdock), GNINA

Usage:
    python run_engine_ablation.py [--seeds 42 43 44 45 46] [--epochs 150]
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

# Import helper functions from run_ablation_experiments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default seeds for experiments
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# Engine combinations
ENGINE_COMBINATIONS = {
    'MoNIG_2engines_dynamicbind_surfdock': {
        'name': 'MoNIG_2engines_dynamicbind_surfdock',
        'engines': ['DynamicBind', 'flowdock'],
        'expert_flags': {'expert3_only': True, 'expert4_only': True}
    },
    'MoNIG_3engines_dynamicbind_surfdock_gnina': {
        'name': 'MoNIG_3engines_dynamicbind_surfdock_gnina',
        'engines': ['DynamicBind', 'flowdock', 'GNINA'],
        'expert_flags': {'expert1_only': True, 'expert3_only': True, 'expert4_only': True}
    }
}


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


def run_test_evaluation(model_type, model_path, norm_stats_path, csv_path, seed, device, output_dir, expert_flags):
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
        '--norm_stats_path', str(norm_stats_path),
        '--csv_path', csv_path,
        '--split', 'test',
        '--output_path', str(inference_output),
        '--seed', str(seed),
        '--device', device,
    ]
    
    # Add expert flags to inference command
    if expert_flags.get('expert1_only'):
        cmd.append('--expert1_only')
    if expert_flags.get('expert2_only'):
        cmd.append('--expert2_only')
    if expert_flags.get('expert3_only'):
        cmd.append('--expert3_only')
    if expert_flags.get('expert4_only'):
        cmd.append('--expert4_only')
    
    try:
        # Run inference
        result = subprocess.run(
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
        print(f"    Command: {' '.join(cmd)}")
        if e.stderr:
            print(f"    Error output: {e.stderr[:500]}")  # Print first 500 chars of error
        if e.stdout:
            print(f"    Stdout: {e.stdout[-500:]}")  # Print last 500 chars of stdout
        return {}
    except Exception as e:
        print(f"    Warning: Error parsing test results: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_training(combination_name, combination_config, seed, csv_path, epochs, batch_size, hidden_dim, 
                dropout, lr, risk_weight, device, output_dir):
    """
    Run training for a specific engine combination and seed.
    
    Returns:
        dict: Results dictionary with success status and output paths
    """
    print(f"\n{'='*80}")
    print(f"Training {combination_name} with seed {seed}")
    print(f"Engines: {', '.join(combination_config['engines'])}")
    print(f"{'='*80}")
    
    # Create output directory for this experiment
    exp_dir = Path(output_dir) / f"{combination_name}_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command - use MoNIG as model_type, with expert flags
    cmd = [
        sys.executable,
        'main.py',
        'train',
        '--model_type', 'MoNIG',  # Always use MoNIG
        '--csv_path', csv_path,
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--hidden_dim', str(hidden_dim),
        '--dropout', str(dropout),
        '--lr', str(lr),
        '--risk_weight', str(risk_weight),
        '--device', device,
    ]
    
    # Add expert flags
    expert_flags = combination_config['expert_flags']
    if expert_flags.get('expert1_only'):
        cmd.append('--expert1_only')
    if expert_flags.get('expert2_only'):
        cmd.append('--expert2_only')
    if expert_flags.get('expert3_only'):
        cmd.append('--expert3_only')
    if expert_flags.get('expert4_only'):
        cmd.append('--expert4_only')
    
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
        # Note: Models are saved as best_MoNIG_emb.pt regardless of expert flags
        # So we need to copy them immediately before the next training overwrites them
        model_path = Path('saved_models') / 'best_MoNIG_emb.pt'
        norm_stats_path = Path('saved_models') / 'best_MoNIG_emb_norm_stats.npz'
        
        success = model_path.exists() and norm_stats_path.exists()
        
        if success:
            # Copy model files to experiment directory immediately
            # Use unique names to avoid conflicts
            import shutil
            exp_model_path = exp_dir / f'best_{combination_name}_emb.pt'
            exp_norm_stats_path = exp_dir / f'best_{combination_name}_emb_norm_stats.npz'
            
            if model_path.exists():
                shutil.copy(model_path, exp_model_path)
            if norm_stats_path.exists():
                shutil.copy(norm_stats_path, exp_norm_stats_path)
            
            # Update paths to point to copied files for inference
            model_path = exp_model_path
            norm_stats_path = exp_norm_stats_path
        
        result = {
            'success': success,
            'combination_name': combination_name,
            'engines': ', '.join(combination_config['engines']),
            'num_engines': len(combination_config['engines']),
            'seed': seed,
            'log_file': str(log_file),
            'model_path': str(model_path) if success else None,
            'exp_dir': str(exp_dir)
        }
        
        # Run test evaluation if training succeeded
        if success:
            test_results = run_test_evaluation(
                model_type='MoNIG',
                model_path=model_path,
                norm_stats_path=norm_stats_path,
                csv_path=csv_path,
                seed=seed,
                device=device,
                output_dir=exp_dir,
                expert_flags=expert_flags
            )
            result.update(test_results)
        
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with return code {e.returncode}")
        return {
            'success': False,
            'combination_name': combination_name,
            'engines': ', '.join(combination_config['engines']),
            'num_engines': len(combination_config['engines']),
            'seed': seed,
            'log_file': str(log_file),
            'error': str(e)
        }


def save_results_to_csv(results, csv_file):
    """
    Save engine ablation experiment results to CSV file.
    
    Args:
        results: List of result dictionaries
        csv_file: Path to output CSV file
    """
    # Prepare data for CSV
    rows = []
    for r in results:
        row = {
            'combination_name': r['combination_name'],
            'engines': r.get('engines', ''),
            'num_engines': r.get('num_engines', 0),
            'seed': r['seed'],
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
    column_order = ['combination_name', 'engines', 'num_engines', 'seed', 'success', 'test_samples',
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
    print("\nSummary Statistics by Engine Combination:")
    successful_results = [r for r in results if r['success']]
    if successful_results:
        successful_df = pd.DataFrame([r for r in successful_results])
        
        for combination_name in successful_df['combination_name'].unique():
            combination_df = successful_df[successful_df['combination_name'] == combination_name]
            print(f"\n  {combination_name}:")
            
            if 'test_mae' in combination_df.columns:
                mae_values = combination_df['test_mae'].dropna()
                if len(mae_values) > 0:
                    print(f"    MAE:  {mae_values.mean():.4f} ± {mae_values.std():.4f} (n={len(mae_values)})")
            
            if 'test_rmse' in combination_df.columns:
                rmse_values = combination_df['test_rmse'].dropna()
                if len(rmse_values) > 0:
                    print(f"    RMSE: {rmse_values.mean():.4f} ± {rmse_values.std():.4f} (n={len(rmse_values)})")
            
            if 'test_corr' in combination_df.columns:
                corr_values = combination_df['test_corr'].dropna()
                if len(corr_values) > 0:
                    print(f"    Corr: {corr_values.mean():.4f} ± {corr_values.std():.4f} (n={len(corr_values)})")
            
            if 'test_r2' in combination_df.columns:
                r2_values = combination_df['test_r2'].dropna()
                if len(r2_values) > 0:
                    print(f"    R²:   {r2_values.mean():.4f} ± {r2_values.std():.4f} (n={len(r2_values)})")
            
            # Proper scoring rules
            if 'test_crps' in combination_df.columns:
                crps_values = combination_df['test_crps'].dropna()
                if len(crps_values) > 0:
                    print(f"    CRPS: {crps_values.mean():.4f} ± {crps_values.std():.4f} (n={len(crps_values)})")
            
            if 'test_nll' in combination_df.columns:
                nll_values = combination_df['test_nll'].dropna()
                if len(nll_values) > 0:
                    print(f"    NLL:  {nll_values.mean():.4f} ± {nll_values.std():.4f} (n={len(nll_values)})")
            
            # Calibration metrics
            if 'test_picp_95' in combination_df.columns:
                picp_values = combination_df['test_picp_95'].dropna()
                if len(picp_values) > 0:
                    print(f"    PICP@95%: {picp_values.mean():.4f} ± {picp_values.std():.4f} (n={len(picp_values)})")
            
            if 'test_picp_90' in combination_df.columns:
                picp_90_values = combination_df['test_picp_90'].dropna()
                if len(picp_90_values) > 0:
                    print(f"    PICP@90%: {picp_90_values.mean():.4f} ± {picp_90_values.std():.4f} (n={len(picp_90_values)})")
            
            if 'test_ece' in combination_df.columns:
                ece_values = combination_df['test_ece'].dropna()
                if len(ece_values) > 0:
                    print(f"    ECE:  {ece_values.mean():.4f} ± {ece_values.std():.4f} (n={len(ece_values)})")
            
            if 'test_avg_interval_width_95' in combination_df.columns:
                width_values = combination_df['test_avg_interval_width_95'].dropna()
                if len(width_values) > 0:
                    print(f"    Avg Width@95%: {width_values.mean():.4f} ± {width_values.std():.4f} (n={len(width_values)})")
            
            # Conformal prediction metrics
            if 'test_conformal_picp' in combination_df.columns:
                conformal_picp_values = combination_df['test_conformal_picp'].dropna()
                if len(conformal_picp_values) > 0:
                    print(f"    Conformal PICP: {conformal_picp_values.mean():.4f} ± {conformal_picp_values.std():.4f} (n={len(conformal_picp_values)})")
            
            if 'test_conformal_avg_width' in combination_df.columns:
                conformal_width_values = combination_df['test_conformal_avg_width'].dropna()
                if len(conformal_width_values) > 0:
                    print(f"    Conformal Width: {conformal_width_values.mean():.4f} ± {conformal_width_values.std():.4f} (n={len(conformal_width_values)})")


def main():
    parser = argparse.ArgumentParser(
        description='Run engine ablation experiments for CABE (MoNIG)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Engine Combinations:
  1. MoNIG_2engines_dynamicbind_surfdock - 2 engines: DynamicBind, flowdock (surfdock)
  2. MoNIG_3engines_dynamicbind_surfdock_gnina - 3 engines: DynamicBind, flowdock (surfdock), GNINA

Examples:
  # Run all engine combinations with default seeds
  python run_engine_ablation.py
  
  # Run specific combinations
  python run_engine_ablation.py --combinations MoNIG_2engines_dynamicbind_surfdock
        """
    )
    
    # Combination selection
    parser.add_argument('--combinations', nargs='+', 
                       choices=list(ENGINE_COMBINATIONS.keys()) + ['all'],
                       default=['all'],
                       help='Engine combinations to test (default: all)')
    
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
    parser.add_argument('--risk_weight', type=float, default=0.01,
                       help='Risk regularization weight (for MoNIG)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='experiments_engine_ablation',
                       help='Output directory for engine ablation experiment results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Determine which combinations to run
    if 'all' in args.combinations:
        combinations_to_run = list(ENGINE_COMBINATIONS.keys())
    else:
        combinations_to_run = args.combinations
    
    # Validate CSV path
    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary file
    summary_file = output_dir / f'engine_ablation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    print("="*80)
    print("CABE ENGINE ABLATION EXPERIMENT RUNNER")
    print("="*80)
    print(f"Engine combinations: {', '.join(combinations_to_run)}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"CSV path: {args.csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Run experiments
    all_results = []
    total_experiments = len(combinations_to_run) * len(args.seeds)
    current_experiment = 0
    
    for combination_name in combinations_to_run:
        combination_config = ENGINE_COMBINATIONS[combination_name]
        for seed in args.seeds:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] ", end='')
            
            result = run_training(
                combination_name=combination_name,
                combination_config=combination_config,
                seed=seed,
                csv_path=args.csv_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                risk_weight=args.risk_weight,
                device=args.device,
                output_dir=output_dir
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
    print("ENGINE ABLATION EXPERIMENT SUMMARY")
    print("="*80)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessful experiments:")
        for r in successful:
            print(f"  {r['combination_name']} (seed {r['seed']}) -> {r['exp_dir']}")
    
    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  {r['combination_name']} (seed {r['seed']})")
            if 'error' in r:
                print(f"    Error: {r['error']}")
            print(f"    Log: {r['log_file']}")
    
    # Summary by combination
    print("\nResults by engine combination:")
    for combination_name in combinations_to_run:
        combination_results = [r for r in all_results if r['combination_name'] == combination_name]
        combination_successful = [r for r in combination_results if r['success']]
        print(f"  {combination_name}: {len(combination_successful)}/{len(combination_results)} successful")
    
    print(f"\n📄 Full results saved to: {summary_file}")
    
    # Save results to CSV
    csv_file = output_dir / f'engine_ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    save_results_to_csv(all_results, csv_file)
    print(f"CSV results saved to: {csv_file}")
    print("="*80)
    
    # Exit with error code if any experiments failed
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
