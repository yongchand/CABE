#!/usr/bin/env python3
"""
Run training experiments for multiple model types with multiple seeds.

This script trains all model types with multiple random seeds to ensure
reproducibility and statistical significance for ICML submission.

Usage:
    python run_multi_seed_experiments.py [--seeds 42 43 44 45 46] [--epochs 150] [--csv_path ...]
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


# All available model types
ALL_MODEL_TYPES = [
    'MoNIG',
    'NIG',
    'Gaussian',
    'Baseline',
    'DeepEnsemble',
    'MCDropout',
    'RandomForest'
]

# Default seeds for experiments
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def run_test_evaluation(model_type, model_path, csv_path, seed, device, output_dir):
    """
    Run inference on test set and extract metrics.
    
    Returns:
        dict: Test metrics dictionary
    """
    print(f"  Running test evaluation...")
    
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
    
    # Add model-specific arguments
    if model_type == 'DeepEnsemble':
        cmd.extend(['--num_models', '5'])
    elif model_type == 'MCDropout':
        cmd.extend(['--num_mc_samples', '50'])
    elif model_type == 'RandomForest':
        cmd.extend(['--n_estimators', '100', '--max_depth', '20'])
    
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
            
            if model_type == 'MoNIG':
                # MoNIG has specific columns
                if 'MoNIG_Prediction' in df.columns and 'True_Affinity' in df.columns:
                    y_pred = df['MoNIG_Prediction'].values
                    y_true = df['True_Affinity'].values
                    
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
            else:
                # Other models use 'Prediction' column
                if 'Prediction' in df.columns and 'True_Affinity' in df.columns:
                    y_pred = df['Prediction'].values
                    y_true = df['True_Affinity'].values
                    
                    metrics['test_mae'] = np.mean(np.abs(y_pred - y_true))
                    metrics['test_rmse'] = np.sqrt(np.mean((y_pred - y_true) ** 2))
                    metrics['test_corr'] = np.corrcoef(y_pred, y_true)[0, 1]
                    
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    metrics['test_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    
                    # Uncertainty metrics
                    if 'Uncertainty' in df.columns:
                        metrics['test_mean_uncertainty'] = df['Uncertainty'].mean()
                        metrics['test_mean_std'] = df['Uncertainty'].mean()  # Assuming Uncertainty is std
            
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
        return {}


def run_training(model_type, seed, csv_path, epochs, batch_size, hidden_dim, 
                dropout, lr, risk_weight, device, output_dir):
    """
    Run training for a specific model type and seed.
    
    Returns:
        dict: Results dictionary with success status and output paths
    """
    print(f"\n{'='*80}")
    print(f"Training {model_type} with seed {seed}")
    print(f"{'='*80}")
    
    # Create output directory for this experiment
    exp_dir = Path(output_dir) / f"{model_type}_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        'main.py',
        'train',
        '--model_type', model_type,
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
    
    # Add model-specific arguments
    if model_type == 'DeepEnsemble':
        cmd.extend(['--num_models', '5'])
    elif model_type == 'MCDropout':
        cmd.extend(['--num_mc_samples', '50'])
    elif model_type == 'RandomForest':
        cmd.extend(['--n_estimators', '100', '--max_depth', '20'])
    
    # Run training
    log_file = exp_dir / 'training.log'
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
        
        # Check if model files were created
        model_path = Path('saved_models') / f'best_{model_type}_emb.pt'
        norm_stats_path = Path('saved_models') / f'best_{model_type}_emb_norm_stats.npz'
        
        success = model_path.exists() and norm_stats_path.exists()
        
        if success:
            # Copy model files to experiment directory
            import shutil
            if model_path.exists():
                shutil.copy(model_path, exp_dir / model_path.name)
            if norm_stats_path.exists():
                shutil.copy(norm_stats_path, exp_dir / norm_stats_path.name)
            
            # Check for calibrator
            calibrator_path = Path('saved_models') / f'best_{model_type}_emb_calibrator.pkl'
            if calibrator_path.exists():
                shutil.copy(calibrator_path, exp_dir / calibrator_path.name)
        
        result = {
            'success': success,
            'model_type': model_type,
            'seed': seed,
            'log_file': str(log_file),
            'model_path': str(model_path) if success else None,
            'exp_dir': str(exp_dir)
        }
        
        # Run test evaluation if training succeeded
        if success:
            test_results = run_test_evaluation(
                model_type=model_type,
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
            'model_type': model_type,
            'seed': seed,
            'log_file': str(log_file),
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run training experiments for multiple model types with multiple seeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run all models with default seeds (42-46)
  python run_multi_seed_experiments.py
  
  # Run specific models with custom seeds
  python run_multi_seed_experiments.py --model_types MoNIG NIG --seeds 42 43 44
  
  # Run with custom training parameters
  python run_multi_seed_experiments.py --epochs 200 --lr 1e-3
        """
    )
    
    # Model selection
    parser.add_argument('--model_types', nargs='+', 
                       choices=ALL_MODEL_TYPES + ['all'],
                       default=['all'],
                       help='Model types to train (default: all)')
    
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
    parser.add_argument('--risk_weight', type=float, default=0.005,
                       help='Risk regularization weight (for NIG/MoNIG)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for experiment results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Determine which models to train
    if 'all' in args.model_types:
        model_types = ALL_MODEL_TYPES
    else:
        model_types = args.model_types
    
    # Validate CSV path
    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary file
    summary_file = output_dir / f'experiment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    print("="*80)
    print("MULTI-SEED EXPERIMENT RUNNER")
    print("="*80)
    print(f"Model types: {', '.join(model_types)}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"CSV path: {args.csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Run experiments
    all_results = []
    total_experiments = len(model_types) * len(args.seeds)
    current_experiment = 0
    
    for model_type in model_types:
        for seed in args.seeds:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] ", end='')
            
            result = run_training(
                model_type=model_type,
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
            with open(summary_file, 'w') as f:
                json.dump({
                    'config': vars(args),
                    'results': all_results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n✅ Successful experiments:")
        for r in successful:
            print(f"  {r['model_type']} (seed {r['seed']}) -> {r['exp_dir']}")
    
    if failed:
        print("\n❌ Failed experiments:")
        for r in failed:
            print(f"  {r['model_type']} (seed {r['seed']})")
            if 'error' in r:
                print(f"    Error: {r['error']}")
            print(f"    Log: {r['log_file']}")
    
    # Summary by model type
    print("\n📊 Results by model type:")
    for model_type in model_types:
        model_results = [r for r in all_results if r['model_type'] == model_type]
        model_successful = [r for r in model_results if r['success']]
        print(f"  {model_type}: {len(model_successful)}/{len(model_results)} successful")
    
    print(f"\n📄 Full results saved to: {summary_file}")
    
    # Save results to CSV
    csv_file = output_dir / f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    save_results_to_csv(all_results, csv_file)
    print(f"📊 CSV results saved to: {csv_file}")
    print("="*80)
    
    # Exit with error code if any experiments failed
    if failed:
        sys.exit(1)


def save_results_to_csv(results, csv_file):
    """
    Save experiment results to CSV file.
    
    Args:
        results: List of result dictionaries
        csv_file: Path to output CSV file
    """
    # Prepare data for CSV
    rows = []
    for r in results:
        row = {
            'model_type': r['model_type'],
            'seed': r['seed'],
            'success': r['success'],
        }
        
        # Add test metrics if available
        test_metrics = ['test_mae', 'test_rmse', 'test_corr', 'test_r2',
                       'test_mean_epistemic', 'test_mean_aleatoric', 
                       'test_mean_total_uncertainty', 'test_mean_uncertainty',
                       'test_mean_std', 'test_samples']
        
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
    column_order = ['model_type', 'seed', 'success', 'test_samples',
                   'test_mae', 'test_rmse', 'test_corr', 'test_r2',
                   'test_mean_epistemic', 'test_mean_aleatoric', 
                   'test_mean_total_uncertainty', 'test_mean_uncertainty',
                   'test_mean_std', 'model_path', 'log_file', 'exp_dir', 'error']
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    df.to_csv(csv_file, index=False)
    
    # Print summary statistics
    print("\n📈 Summary Statistics by Model Type:")
    successful_results = [r for r in results if r['success']]
    if successful_results:
        successful_df = pd.DataFrame([r for r in successful_results])
        
        for model_type in successful_df['model_type'].unique():
            model_df = successful_df[successful_df['model_type'] == model_type]
            print(f"\n  {model_type}:")
            
            if 'test_mae' in model_df.columns:
                mae_values = model_df['test_mae'].dropna()
                if len(mae_values) > 0:
                    print(f"    MAE:  {mae_values.mean():.4f} ± {mae_values.std():.4f} (n={len(mae_values)})")
            
            if 'test_rmse' in model_df.columns:
                rmse_values = model_df['test_rmse'].dropna()
                if len(rmse_values) > 0:
                    print(f"    RMSE: {rmse_values.mean():.4f} ± {rmse_values.std():.4f} (n={len(rmse_values)})")
            
            if 'test_corr' in model_df.columns:
                corr_values = model_df['test_corr'].dropna()
                if len(corr_values) > 0:
                    print(f"    Corr: {corr_values.mean():.4f} ± {corr_values.std():.4f} (n={len(corr_values)})")
            
            if 'test_r2' in model_df.columns:
                r2_values = model_df['test_r2'].dropna()
                if len(r2_values) > 0:
                    print(f"    R²:   {r2_values.mean():.4f} ± {r2_values.std():.4f} (n={len(r2_values)})")


if __name__ == '__main__':
    main()

