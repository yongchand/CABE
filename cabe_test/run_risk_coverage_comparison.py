"""
Risk-Coverage Curve Comparison: CABE vs Baselines

Compares how well each model's uncertainty estimates enable selective prediction.
Tests the claim: "Low uncertainty predictions are more accurate"

Models compared:
- MoNIG (CABE)
- Gaussian
- NIG
- DeepEnsemble
- MCDropout
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import json


def load_and_extract_data(experiment_dir, model_name, seed):
    """
    Load results and extract predictions, errors, and uncertainties.
    
    Returns:
        tuple: (true_values, predictions, errors, uncertainties) or None
    """
    # Map display names to actual model directory names
    model_dir_name = model_name
    if model_name == 'CABE':
        model_dir_name = 'MoNIG'
    
    model_dir = Path(experiment_dir) / f"{model_dir_name}_seed{seed}"
    results_file = model_dir / "test_inference_results.csv"
    
    if not results_file.exists():
        return None
    
    df = pd.read_csv(results_file)
    
    true_values = df['True_Affinity'].values
    
    # Extract predictions and uncertainty based on model type
    if model_name in ['MoNIG', 'CABE']:
        predictions = df['MoNIG_Prediction'].values
        epistemic = df['MoNIG_Epistemic'].values
        aleatoric = df['MoNIG_Aleatoric'].values
        uncertainties = epistemic  # Use epistemic for selective prediction
    else:
        # Baseline models: Gaussian, NIG, DeepEnsemble, DeepEnsembleMVE, SVGP, etc.
        predictions = df['Prediction'].values
        uncertainties = df['Uncertainty'].values
    
    errors = np.abs(predictions - true_values)
    
    return true_values, predictions, errors, uncertainties


def calculate_risk_coverage_curve(errors, uncertainties, coverage_levels):
    """
    Calculate MAE and RMSE at different coverage levels.
    
    Args:
        errors: Prediction errors
        uncertainties: Uncertainty estimates
        coverage_levels: Array of coverage levels (e.g., [0.1, 0.2, ..., 1.0])
        
    Returns:
        dict: MAE, RMSE, and mean uncertainty at each coverage level
    """
    # Sort by uncertainty (ascending)
    sorted_indices = np.argsort(uncertainties)
    sorted_errors = errors[sorted_indices]
    sorted_uncertainties = uncertainties[sorted_indices]
    
    mae_at_coverage = []
    rmse_at_coverage = []
    mean_unc_at_coverage = []
    
    for coverage in coverage_levels:
        n_keep = int(len(sorted_errors) * coverage)
        if n_keep == 0:
            n_keep = 1
        
        kept_errors = sorted_errors[:n_keep]
        kept_uncertainties = sorted_uncertainties[:n_keep]
        
        mae = np.mean(kept_errors)
        rmse = np.sqrt(np.mean(kept_errors**2))
        mean_unc = np.mean(kept_uncertainties)
        
        mae_at_coverage.append(mae)
        rmse_at_coverage.append(rmse)
        mean_unc_at_coverage.append(mean_unc)
    
    return {
        'mae': np.array(mae_at_coverage),
        'rmse': np.array(rmse_at_coverage),
        'mean_uncertainty': np.array(mean_unc_at_coverage)
    }


def analyze_model_risk_coverage(experiment_dir, model_name, seeds, coverage_levels):
    """
    Analyze risk-coverage curves across multiple seeds.
    
    Returns:
        dict: Aggregated statistics
    """
    all_mae_curves = []
    all_rmse_curves = []
    mae_full_all = []
    rmse_full_all = []
    
    for seed in seeds:
        result = load_and_extract_data(experiment_dir, model_name, seed)
        if result is None:
            continue
        
        true_values, predictions, errors, uncertainties = result
        
        # Full set metrics
        mae_full = np.mean(errors)
        rmse_full = np.sqrt(np.mean(errors**2))
        mae_full_all.append(mae_full)
        rmse_full_all.append(rmse_full)
        
        # Risk-coverage curve
        curve = calculate_risk_coverage_curve(errors, uncertainties, coverage_levels)
        all_mae_curves.append(curve['mae'])
        all_rmse_curves.append(curve['rmse'])
    
    if len(all_mae_curves) == 0:
        return None
    
    # Aggregate across seeds
    mae_curves = np.array(all_mae_curves)
    rmse_curves = np.array(all_rmse_curves)
    
    # Calculate improvements at key thresholds
    improvements = {}
    for coverage_pct in [50, 70, 80, 90]:
        coverage_val = coverage_pct / 100.0
        idx = np.argmin(np.abs(coverage_levels - coverage_val))
        
        mae_at_cov = mae_curves[:, idx].mean()
        mae_full_mean = np.mean(mae_full_all)
        mae_improvement = (mae_full_mean - mae_at_cov) / mae_full_mean * 100
        
        rmse_at_cov = rmse_curves[:, idx].mean()
        rmse_full_mean = np.mean(rmse_full_all)
        rmse_improvement = (rmse_full_mean - rmse_at_cov) / rmse_full_mean * 100
        
        improvements[coverage_pct] = {
            'mae_kept': mae_at_cov,
            'mae_improvement': mae_improvement,
            'rmse_kept': rmse_at_cov,
            'rmse_improvement': rmse_improvement
        }
    
    return {
        'mae_mean': mae_curves.mean(axis=0),
        'mae_std': mae_curves.std(axis=0),
        'rmse_mean': rmse_curves.mean(axis=0),
        'rmse_std': rmse_curves.std(axis=0),
        'mae_full_mean': np.mean(mae_full_all),
        'rmse_full_mean': np.mean(rmse_full_all),
        'improvements': improvements,
        'num_seeds': len(all_mae_curves)
    }


def plot_risk_coverage_comparison(results_dict, coverage_levels, output_dir):
    """
    Plot risk-coverage curves for all models - Publication quality.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Publication-quality settings - BIGGER fonts
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
    })
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    
    # Clean, distinct colors
    colors = {
        'CABE': '#2E86AB',     # Blue
        'MoNIG': '#2E86AB',    # Blue (alias)
        'Evidential Regression': '#A23B72',  # Magenta
        'NIG': '#A23B72',      # Magenta (alias)
        'Gaussian': '#F18F01', # Orange
        'DeepEnsembleMVE': '#C73E1D',  # Red
        'SVGP': '#6B4E71'      # Purple
    }
    
    markers = {
        'CABE': 'o',
        'MoNIG': 'o',
        'Evidential Regression': '^',
        'NIG': '^',
        'Gaussian': 's',
        'DeepEnsembleMVE': 'D',
        'SVGP': 'v'
    }
    
    # Map model names for display
    display_names = {
        'CABE': 'CABE',
        'MoNIG': 'CABE',
        'NIG': 'Evidential Regression',
        'Gaussian': 'Gaussian',
        'DeepEnsembleMVE': 'DeepEnsemble MVE',
        'SVGP': 'SVGP'
    }
    
    # Plot 1: MAE vs Coverage (left panel)
    for model_name, results in results_dict.items():
        if results is None:
            continue
        
        display_name = display_names.get(model_name, model_name)
        color = colors.get(model_name, colors.get(display_name, '#888888'))
        marker = markers.get(model_name, markers.get(display_name, 'o'))
        
        mae_mean = results['mae_mean']
        mae_std = results['mae_std']
        
        # Plot line with markers at key points only - BIGGER markers and lines
        ax1.plot(coverage_levels * 100, mae_mean, 
                color=color, linewidth=3.5, alpha=0.9,
                label=display_name)
        
        # Add std shaded area
        ax1.fill_between(coverage_levels * 100, 
                         mae_mean - mae_std, 
                         mae_mean + mae_std,
                         color=color, alpha=0.15)
        
        # Add markers only at key coverage points (20, 40, 60, 80, 100)
        key_points = [i for i, c in enumerate(coverage_levels) if int(c * 100) in [20, 40, 60, 80, 100]]
        if len(key_points) > 0:
            ax1.scatter(coverage_levels[key_points] * 100, mae_mean[key_points],
                       color=color, marker=marker, s=140, zorder=5, edgecolors='white', linewidth=2)
    
    ax1.set_xlabel('Coverage (%)', fontsize=22)
    ax1.set_ylabel('MAE (pKd)', fontsize=22)
    ax1.legend(loc='lower right', fontsize=14, frameon=True, fancybox=False,
               edgecolor='gray', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xlim(10, 105)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    
    # Clean spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    # Plot 2: Improvement vs Discarded (right panel)
    for model_name, results in results_dict.items():
        if results is None:
            continue
        
        display_name = display_names.get(model_name, model_name)
        color = colors.get(model_name, colors.get(display_name, '#888888'))
        marker = markers.get(model_name, markers.get(display_name, 'o'))
        
        mae_mean = results['mae_mean']
        mae_std = results['mae_std']
        mae_full = results['mae_full_mean']
        
        # Calculate improvements
        improvements = np.array([(mae_full - mae) / mae_full * 100 for mae in mae_mean])
        # Calculate std for improvements (approximate)
        improvements_std = np.array([std / mae_full * 100 for std in mae_std])
        discard_pcts = 100 - coverage_levels * 100
        
        ax2.plot(discard_pcts, improvements,
                color=color, linewidth=3.5, alpha=0.9,
                label=display_name)
        
        # Add std shaded area
        ax2.fill_between(discard_pcts, 
                         improvements - improvements_std, 
                         improvements + improvements_std,
                         color=color, alpha=0.15)
        
        # Add markers at key discard points (20, 40, 60, 80)
        key_points = [i for i, d in enumerate(discard_pcts) if int(round(d)) in [20, 40, 60, 80]]
        if len(key_points) > 0:
            ax2.scatter(discard_pcts[key_points], improvements[key_points],
                       color=color, marker=marker, s=140, zorder=5, edgecolors='white', linewidth=2)
    
    ax2.set_xlabel('Discarded (%)', fontsize=22)
    ax2.set_ylabel('MAE Improvement (%)', fontsize=22)
    ax2.legend(loc='upper left', fontsize=14, frameon=True, fancybox=False,
               edgecolor='gray', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xlim(-2, 92)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    
    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    plot_path = output_path / 'risk_coverage_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plot_path_pdf = output_path / 'risk_coverage_comparison.pdf'
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"📊 Saved risk-coverage comparison to: {plot_path}")


def plot_improvement_heatmap(results_dict, output_dir):
    """
    Create heatmap showing improvement at different coverage levels - Publication quality.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Publication-quality settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })
    
    # Map model names for display
    display_names = {
        'CABE': 'CABE',
        'MoNIG': 'CABE',
        'NIG': 'Evidential Regression',
        'Gaussian': 'Gaussian',
        'DeepEnsembleMVE': 'DeepEnsemble MVE',
        'SVGP': 'SVGP'
    }
    
    coverage_levels = [50, 70, 80, 90]
    models = []
    display_models = []
    improvements = []
    
    for model_name, results in results_dict.items():
        if results is None:
            continue
        
        models.append(model_name)
        display_models.append(display_names.get(model_name, model_name))
        model_improvements = []
        for cov in coverage_levels:
            imp = results['improvements'].get(cov, {}).get('mae_improvement', 0)
            model_improvements.append(imp)
        improvements.append(model_improvements)
    
    improvements = np.array(improvements)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Use a cleaner colormap
    im = ax.imshow(improvements, cmap='YlGn', aspect='auto', vmin=0, vmax=max(20, improvements.max()))
    
    # Set ticks
    ax.set_xticks(np.arange(len(coverage_levels)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f'{c}%' for c in coverage_levels], fontsize=12)
    ax.set_yticklabels(display_models, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('MAE Improvement (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(coverage_levels)):
            val = improvements[i, j]
            # Use white text for dark backgrounds
            text_color = 'white' if val > 10 else 'black'
            ax.text(j, i, f'{val:.1f}%',
                   ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Coverage', fontsize=14)
    ax.set_ylabel('Method', fontsize=14)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    plot_path = output_path / 'improvement_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"📊 Saved improvement heatmap to: {plot_path}")


def print_comparison_report(results_dict):
    """
    Print comprehensive comparison report.
    """
    print("\n" + "="*100)
    print("RISK-COVERAGE COMPARISON: CABE vs BASELINES")
    print("="*100)
    
    # Full set performance
    print("\nFull Test Set Performance (100% coverage):")
    print(f"{'Model':<15} {'MAE (pKd)':<15} {'RMSE (pKd)':<15} {'Seeds':<10}")
    print("-"*100)
    
    for model_name, results in sorted(results_dict.items()):
        if results is None:
            print(f"{model_name:<15} {'N/A':<15} {'N/A':<15} {'0':<10}")
            continue
        
        mae_str = f"{results['mae_full_mean']:.4f}"
        rmse_str = f"{results['rmse_full_mean']:.4f}"
        
        print(f"{model_name:<15} {mae_str:<15} {rmse_str:<15} {results['num_seeds']:<10}")
    
    # Improvements at key thresholds
    for coverage_pct in [50, 70, 90]:
        print(f"\n" + "="*100)
        print(f"Performance at {coverage_pct}% Coverage ({100-coverage_pct}% most uncertain discarded)")
        print("="*100)
        print(f"{'Model':<15} {'MAE (pKd)':<15} {'MAE ↓%':<15} {'RMSE (pKd)':<15} {'RMSE ↓%':<15}")
        print("-"*100)
        
        for model_name, results in sorted(results_dict.items()):
            if results is None or coverage_pct not in results['improvements']:
                print(f"{model_name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                continue
            
            imp = results['improvements'][coverage_pct]
            mae_str = f"{imp['mae_kept']:.4f}"
            mae_imp_str = f"{imp['mae_improvement']:.1f}%"
            rmse_str = f"{imp['rmse_kept']:.4f}"
            rmse_imp_str = f"{imp['rmse_improvement']:.1f}%"
            
            print(f"{model_name:<15} {mae_str:<15} {mae_imp_str:<15} {rmse_str:<15} {rmse_imp_str:<15}")
    
    # Ranking by improvement at 70% coverage
    print("\n" + "="*100)
    print("RANKING: Best Uncertainty Estimates for Selective Prediction (70% coverage)")
    print("="*100)
    
    rankings = []
    for model_name, results in results_dict.items():
        if results is not None and 70 in results['improvements']:
            imp = results['improvements'][70]['mae_improvement']
            rankings.append((model_name, imp))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<8} {'Model':<15} {'MAE Improvement':<20} {'Interpretation'}")
    print("-"*100)
    
    for rank, (model_name, improvement) in enumerate(rankings, 1):
        if improvement > 10:
            interpretation = "✅ Excellent (>10%)"
        elif improvement > 5:
            interpretation = "✅ Good (5-10%)"
        elif improvement > 2:
            interpretation = "⚠️ Moderate (2-5%)"
        else:
            interpretation = "❌ Poor (<2%)"
        
        print(f"{rank:<8} {model_name:<15} {improvement:>6.1f}%{'':<13} {interpretation}")
    
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    
    if rankings:
        best_model, best_improvement = rankings[0]
        print(f"\n✅ Best Model: {best_model}")
        print(f"   → {best_improvement:.1f}% MAE improvement at 70% coverage")
        print(f"   → This model's uncertainty estimates are most useful for selective prediction")
        
        if len(rankings) > 1:
            worst_model, worst_improvement = rankings[-1]
            print(f"\n⚠️  Worst Model: {worst_model}")
            print(f"   → {worst_improvement:.1f}% MAE improvement at 70% coverage")
            print(f"   → This model's uncertainty estimates provide limited benefit")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Risk-Coverage Curve Comparison: CABE vs Baselines',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Experiment directory (default: experiments)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['CABE', 'NIG', 'Gaussian', 'DeepEnsembleMVE', 'SVGP'],
                       help='Models to compare')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                       help='Seeds to analyze (default: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51])')
    parser.add_argument('--output_dir', type=str, default='risk_coverage_comparison',
                       help='Output directory (default: risk_coverage_comparison)')
    
    args = parser.parse_args()
    
    print("="*100)
    print("RISK-COVERAGE CURVE COMPARISON")
    print("="*100)
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Seeds: {args.seeds}")
    print(f"Output directory: {args.output_dir}")
    print("="*100)
    
    # Coverage levels to test
    coverage_levels = np.arange(0.1, 1.01, 0.05)
    
    # Analyze each model
    results_dict = {}
    
    for model_name in args.models:
        print(f"\nAnalyzing {model_name}...")
        results = analyze_model_risk_coverage(
            args.experiment_dir,
            model_name,
            args.seeds,
            coverage_levels
        )
        results_dict[model_name] = results
        
        if results is not None:
            print(f"  ✅ Analyzed {results['num_seeds']} seeds")
            if 70 in results['improvements']:
                imp_70 = results['improvements'][70]['mae_improvement']
                print(f"  Improvement @ 70%: {imp_70:.1f}%")
        else:
            print(f"  ⚠️  No results found")
    
    # Generate plots
    print("\n" + "="*100)
    print("GENERATING VISUALIZATIONS")
    print("="*100)
    
    plot_risk_coverage_comparison(results_dict, coverage_levels, args.output_dir)
    plot_improvement_heatmap(results_dict, args.output_dir)
    
    # Print report
    print_comparison_report(results_dict)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {}
    for model_name, results in results_dict.items():
        if results is None:
            json_results[model_name] = None
            continue
        
        json_results[model_name] = {
            'mae_mean': results['mae_mean'].tolist(),
            'mae_std': results['mae_std'].tolist(),
            'rmse_mean': results['rmse_mean'].tolist(),
            'rmse_std': results['rmse_std'].tolist(),
            'mae_full_mean': float(results['mae_full_mean']),
            'rmse_full_mean': float(results['rmse_full_mean']),
            'improvements': {
                str(k): {
                    'mae_kept': float(v['mae_kept']),
                    'mae_improvement': float(v['mae_improvement']),
                    'rmse_kept': float(v['rmse_kept']),
                    'rmse_improvement': float(v['rmse_improvement'])
                }
                for k, v in results['improvements'].items()
            },
            'num_seeds': results['num_seeds']
        }
    
    json_path = output_path / 'risk_coverage_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Saved results to: {json_path}")
    print("\n✅ Risk-coverage comparison complete!")
    print("="*100)


if __name__ == '__main__':
    main()

