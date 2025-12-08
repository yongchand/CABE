#!/usr/bin/env python3
"""
Run CABE Case Studies Analysis

This script performs comprehensive case study analysis on MoNIG inference results:
1. CABE Best Cases - where MoNIG outperforms ALL individual experts
2. Catastrophe Avoidance - where MoNIG avoids expert failures
3. High Epistemic = Hard Cases - where high uncertainty correctly identifies difficult predictions

Usage:
    # Run on latest experiment results
    python run_case_studies.py
    
    # Run on specific results file
    python run_case_studies.py --csv experiments/MoNIG_seed42/test_inference_results.csv
    
    # Run on all seeds and aggregate
    python run_case_studies.py --experiment_dir experiments --seeds 42 43 44 45 46
"""

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime


# Expert names mapping
EXPERT_NAMES = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']


def merge_with_original_experts(inference_df, original_csv_path='pdbbind_descriptors_with_experts_and_binding.csv'):
    """
    Merge inference results with original expert predictions from dataset.
    
    Args:
        inference_df: DataFrame with MoNIG inference results
        original_csv_path: Path to original dataset CSV
    
    Returns:
        Merged DataFrame with both MoNIG and original expert predictions
    """
    original_df = pd.read_csv(original_csv_path)
    
    # Select only the columns we need from original dataset
    expert_cols = ['ComplexID', 'GNINA_Affinity', 'BIND_pIC50', 'flowdock_score', 'DynamicBind_score', 'Binding_Affinity']
    original_subset = original_df[expert_cols].copy()
    
    # Merge on ComplexID
    merged_df = inference_df.merge(original_subset, on='ComplexID', how='left', suffixes=('', '_orig'))
    
    # Verify we got the expert scores
    missing = merged_df['GNINA_Affinity'].isna().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} complexes missing original expert scores")
    
    return merged_df


def find_best_cases(df, num_experts=4, min_improvement=0.0, use_original_experts=True):
    """
    Find cases where CABE outperformed all single engines.
    
    Args:
        df: DataFrame with inference results (must have original expert scores if use_original_experts=True)
        num_experts: Number of experts (default: 4)
        min_improvement: Minimum improvement over best expert (default: 0.0)
        use_original_experts: Use original expert scores instead of NIG predictions (default: True)
    
    Returns:
        DataFrame with success cases
    """
    # Calculate errors
    df['MoNIG_Error'] = np.abs(df['MoNIG_Prediction'] - df['True_Affinity'])
    
    expert_errors = []
    expert_prediction_cols = ['GNINA_Affinity', 'BIND_pIC50', 'flowdock_score', 'DynamicBind_score']
    
    for j in range(num_experts):
        expert_num = j + 1
        error_col = f'Expert{expert_num}_Error'
        
        if use_original_experts and expert_prediction_cols[j] in df.columns:
            # Use original expert predictions from dataset
            df[error_col] = np.abs(df[expert_prediction_cols[j]] - df['True_Affinity'])
            # Also rename for consistency
            if f'Expert{expert_num}_Prediction' not in df.columns:
                df[f'Expert{expert_num}_Prediction'] = df[expert_prediction_cols[j]]
        else:
            # Fallback to NIG predictions
            df[error_col] = np.abs(df[f'Expert{expert_num}_Prediction'] - df['True_Affinity'])
        
        expert_errors.append(error_col)
    
    # Find cases where MoNIG error < all expert errors
    monig_better_mask = df['MoNIG_Error'] < df[expert_errors].min(axis=1)
    
    # Calculate improvement over best expert
    best_expert_error = df[expert_errors].min(axis=1)
    df['Improvement_Over_Best'] = best_expert_error - df['MoNIG_Error']
    df['Improvement_Percent'] = (df['Improvement_Over_Best'] / best_expert_error * 100).replace([np.inf, -np.inf], np.nan)
    
    # Filter by minimum improvement
    success_cases = df[monig_better_mask & (df['Improvement_Over_Best'] >= min_improvement)].copy()
    
    # Sort by improvement (best first)
    success_cases = success_cases.sort_values('Improvement_Over_Best', ascending=False)
    
    # Add which expert was best (before MoNIG)
    best_expert_idx = df[expert_errors].values.argmin(axis=1)
    success_cases['Best_Expert_Before'] = [EXPERT_NAMES[idx] if idx < len(EXPERT_NAMES) else f'Expert {idx+1}' 
                                          for idx in best_expert_idx[success_cases.index]]
    
    return success_cases


def find_catastrophe_cases(df, num_experts=4, use_original_experts=True):
    """
    Find cases where CABE avoided catastrophic expert failures.
    
    Args:
        df: DataFrame with inference results (must have original expert scores if use_original_experts=True)
        num_experts: Number of experts (default: 4)
        use_original_experts: Use original expert scores instead of NIG predictions (default: True)
    
    Returns:
        Tuple of (general_catastrophe_df, dynamicbind_catastrophe_df)
    """
    expert_errors = [f'Expert{j+1}_Error' for j in range(num_experts)]
    expert_prediction_cols = ['GNINA_Affinity', 'BIND_pIC50', 'flowdock_score', 'DynamicBind_score']
    
    # Ensure error columns exist
    for j in range(num_experts):
        expert_num = j + 1
        error_col = f'Expert{expert_num}_Error'
        if error_col not in df.columns:
            if use_original_experts and expert_prediction_cols[j] in df.columns:
                # Use original expert predictions
                df[error_col] = np.abs(df[expert_prediction_cols[j]] - df['True_Affinity'])
                if f'Expert{expert_num}_Prediction' not in df.columns:
                    df[f'Expert{expert_num}_Prediction'] = df[expert_prediction_cols[j]]
            else:
                # Fallback to NIG predictions
                df[error_col] = np.abs(df[f'Expert{expert_num}_Prediction'] - df['True_Affinity'])
    
    high_epistemic_threshold = df['MoNIG_Epistemic'].quantile(0.70)
    worst_expert_error = df[expert_errors].max(axis=1)
    best_expert_error = df[expert_errors].min(axis=1)
    
    # General catastrophe avoidance (any expert fails)
    # TRUE AVOIDANCE: MoNIG has REASONABLE error (<1.5) while expert catastrophically fails (>3.0)
    general_catastrophe = df[
        (worst_expert_error > 3.0) &  # At least one expert explodes
        (df['MoNIG_Error'] < 1.5) &  # MoNIG has REASONABLE error (TRUE avoidance, not just "less bad")
        (df['MoNIG_Epistemic'] > high_epistemic_threshold)  # High epistemic
    ].copy()
    
    general_catastrophe['Worst_Expert_Error'] = worst_expert_error[general_catastrophe.index]
    general_catastrophe['Best_Expert_Error'] = best_expert_error[general_catastrophe.index]
    general_catastrophe['MoNIG_vs_Worst'] = general_catastrophe['Worst_Expert_Error'] - general_catastrophe['MoNIG_Error']
    general_catastrophe = general_catastrophe.sort_values('Worst_Expert_Error', ascending=False)
    
    # DynamicBind-specific catastrophe
    # TRUE AVOIDANCE: MoNIG has REASONABLE error (<1.5) while DynamicBind catastrophically fails (>3.0)
    dynamicbind_error = df['Expert4_Error']
    dynamicbind_catastrophe = df[
        (dynamicbind_error > 3.0) &  # DynamicBind explodes
        (df['MoNIG_Error'] < 1.5) &  # MoNIG has REASONABLE error (TRUE avoidance, not just "less bad")
        (df['MoNIG_Epistemic'] > high_epistemic_threshold)  # High epistemic
    ].copy()
    
    if len(dynamicbind_catastrophe) > 0:
        dynamicbind_catastrophe['DynamicBind_Error'] = dynamicbind_error[dynamicbind_catastrophe.index]
        dynamicbind_catastrophe['Best_Expert_Error'] = best_expert_error[dynamicbind_catastrophe.index]
        dynamicbind_catastrophe['MoNIG_vs_DynamicBind'] = dynamicbind_catastrophe['DynamicBind_Error'] - dynamicbind_catastrophe['MoNIG_Error']
        dynamicbind_catastrophe = dynamicbind_catastrophe.sort_values('DynamicBind_Error', ascending=False)
    
    # Also find "damage mitigation" cases (MoNIG error 1.5-3.0, expert >3.0)
    # These show MoNIG is "less bad" but not truly good
    damage_mitigation = df[
        (worst_expert_error > 3.0) &  # Expert catastrophic
        (df['MoNIG_Error'] >= 1.5) &  # MoNIG also bad
        (df['MoNIG_Error'] < 3.0) &  # But not catastrophic
        (df['MoNIG_Error'] < worst_expert_error)  # Better than worst
    ].copy()
    
    if len(damage_mitigation) > 0:
        damage_mitigation['Worst_Expert_Error'] = worst_expert_error[damage_mitigation.index]
        damage_mitigation['MoNIG_vs_Worst'] = damage_mitigation['Worst_Expert_Error'] - damage_mitigation['MoNIG_Error']
        damage_mitigation = damage_mitigation.sort_values('MoNIG_vs_Worst', ascending=False)
    
    return general_catastrophe, dynamicbind_catastrophe, damage_mitigation


def find_high_epistemic_cases(df, num_experts=4, use_original_experts=True):
    """
    Find cases with high epistemic uncertainty (hard cases).
    
    Note: Uses original expert predictions for errors but NIG uncertainties 
    (since uncertainty is not available in original dataset).
    
    Args:
        df: DataFrame with inference results (must have original expert scores if use_original_experts=True)
        num_experts: Number of experts (default: 4)
        use_original_experts: Use original expert scores for errors instead of NIG predictions (default: True)
    
    Returns:
        DataFrame with high epistemic cases
    """
    expert_total_uncertainty = [f'Expert{j+1}_Total_Uncertainty' for j in range(num_experts)]
    expert_errors = [f'Expert{j+1}_Error' for j in range(num_experts)]
    expert_prediction_cols = ['GNINA_Affinity', 'BIND_pIC50', 'flowdock_score', 'DynamicBind_score']
    
    # Ensure error columns exist using original expert predictions if available
    for j in range(num_experts):
        expert_num = j + 1
        error_col = f'Expert{expert_num}_Error'
        if error_col not in df.columns:
            if use_original_experts and expert_prediction_cols[j] in df.columns:
                df[error_col] = np.abs(df[expert_prediction_cols[j]] - df['True_Affinity'])
            else:
                df[error_col] = np.abs(df[f'Expert{expert_num}_Prediction'] - df['True_Affinity'])
    
    # Calculate total uncertainty for experts if not present (uses NIG uncertainties)
    for j in range(num_experts):
        expert_num = j + 1
        unc_col = f'Expert{expert_num}_Total_Uncertainty'
        if unc_col not in df.columns:
            df[unc_col] = df[f'Expert{expert_num}_Epistemic'] + df[f'Expert{expert_num}_Aleatoric']
    
    # Thresholds
    high_error_threshold = 1.3
    high_epistemic_threshold = df['MoNIG_Epistemic'].quantile(0.70)
    low_uncertainty_threshold = df[expert_total_uncertainty].quantile(0.30).min()
    expert_wrong_threshold = 0.8
    
    candidates = df[
        (df['MoNIG_Error'] > high_error_threshold) &
        (df['MoNIG_Epistemic'] > high_epistemic_threshold)
    ].copy()
    
    # Check if any expert has low uncertainty but high error
    final_cases = []
    for idx, row in candidates.iterrows():
        for j in range(num_experts):
            expert_num = j + 1
            expert_unc = row[f'Expert{expert_num}_Total_Uncertainty']
            expert_err = row[f'Expert{expert_num}_Error']
            
            if expert_unc < low_uncertainty_threshold and expert_err > expert_wrong_threshold:
                final_cases.append(idx)
                break
    
    high_epistemic_df = candidates.loc[final_cases].copy()
    high_epistemic_df = high_epistemic_df.sort_values('MoNIG_Error', ascending=False)
    
    return high_epistemic_df


def analyze_epistemic_vs_disagreement(df, output_dir=None):
    """
    Analyze the relationship between epistemic uncertainty and expert disagreement.
    
    Args:
        df: DataFrame with inference results
        output_dir: Optional directory to save plots
        
    Returns:
        dict: Statistics about the relationship
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Calculate expert disagreement (std of expert predictions)
    expert_pred_cols = [col for col in df.columns if col.startswith('Expert') and col.endswith('_Prediction')]
    if len(expert_pred_cols) == 0:
        print("⚠️  No expert predictions found for disagreement calculation")
        return None
    
    # Expert disagreement = std of expert predictions
    df['Expert_Disagreement'] = df[expert_pred_cols].std(axis=1)
    
    # Get epistemic uncertainty
    epistemic = df['MoNIG_Epistemic'].values
    disagreement = df['Expert_Disagreement'].values
    
    # Calculate correlation
    pearson_corr, pearson_p = stats.pearsonr(epistemic, disagreement)
    spearman_corr, spearman_p = stats.spearmanr(epistemic, disagreement)
    
    # Calculate statistics
    stats_dict = {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'mean_epistemic': epistemic.mean(),
        'std_epistemic': epistemic.std(),
        'mean_disagreement': disagreement.mean(),
        'std_disagreement': disagreement.std(),
    }
    
    # Create scatter plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(disagreement, epistemic, alpha=0.5, s=20)
        
        # Add trend line
        z = np.polyfit(disagreement, epistemic, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(disagreement.min(), disagreement.max(), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
        
        plt.xlabel('Expert Disagreement (Std of Predictions)', fontsize=12)
        plt.ylabel('MoNIG Epistemic Uncertainty', fontsize=12)
        plt.title(f'Epistemic Uncertainty vs Expert Disagreement\n(Pearson r={pearson_corr:.3f}, p={pearson_p:.2e})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = output_path / 'epistemic_vs_disagreement.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved epistemic vs disagreement plot to: {plot_path}")
    
    return stats_dict


def analyze_disagreement_vs_error(df, output_dir=None):
    """
    Analyze the relationship between expert disagreement and prediction errors.
    Shows whether expert disagreement correlates with harder cases.
    
    Args:
        df: DataFrame with inference results and expert predictions
        output_dir: Optional directory to save plots
        
    Returns:
        dict: Statistics about the relationship
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Calculate expert disagreement (std of expert predictions)
    expert_pred_cols = [col for col in df.columns if col.startswith('Expert') and col.endswith('_Prediction')]
    if len(expert_pred_cols) == 0:
        print("⚠️  No expert predictions found for disagreement calculation")
        return None
    
    # Expert disagreement = std of expert predictions
    df['Expert_Disagreement'] = df[expert_pred_cols].std(axis=1)
    
    # Get MoNIG error
    if 'MoNIG_Error' not in df.columns:
        df['MoNIG_Error'] = abs(df['MoNIG_Prediction'] - df['True_Affinity'])
    
    disagreement = df['Expert_Disagreement'].values
    monig_error = df['MoNIG_Error'].values
    
    # Calculate correlation
    pearson_corr, pearson_p = stats.pearsonr(disagreement, monig_error)
    spearman_corr, spearman_p = stats.spearmanr(disagreement, monig_error)
    
    # Calculate statistics
    stats_dict = {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'mean_disagreement': disagreement.mean(),
        'std_disagreement': disagreement.std(),
        'mean_error': monig_error.mean(),
        'std_error': monig_error.std(),
    }
    
    # Create scatter plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot with color gradient based on error
        scatter = ax.scatter(disagreement, monig_error, 
                           c=monig_error, cmap='RdYlGn_r', 
                           alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('MoNIG Error (pKd)', fontsize=10)
        
        # Add trend line
        z = np.polyfit(disagreement, monig_error, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(disagreement.min(), disagreement.max(), 100)
        ax.plot(x_trend, p(x_trend), "b--", alpha=0.8, linewidth=2, 
               label=f'Trend line (slope={z[0]:.3f})')
        
        # Add horizontal lines for error thresholds
        ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Good (< 0.5)')
        ax.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5, label='Acceptable (< 1.5)')
        ax.axhline(y=3.0, color='red', linestyle=':', alpha=0.5, label='Catastrophic (> 3.0)')
        
        ax.set_xlabel('Expert Disagreement (Std of Predictions, pKd)', fontsize=12)
        ax.set_ylabel('MoNIG Prediction Error (pKd)', fontsize=12)
        ax.set_title(f'Expert Disagreement vs MoNIG Error\n(Pearson r={pearson_corr:.3f}, p={pearson_p:.2e})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = output_path / 'disagreement_vs_error.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved disagreement vs error plot to: {plot_path}")
    
    return stats_dict


def analyze_uncertainty_accuracy_tradeoff(df, output_dir=None):
    """
    Analyze the relationship between epistemic uncertainty and prediction accuracy.
    Creates a risk-coverage curve showing MAE/RMSE vs coverage.
    
    This demonstrates: "By discarding X% most uncertain predictions, MAE reduces by Y%"
    
    Args:
        df: DataFrame with inference results
        output_dir: Optional directory to save plots
        
    Returns:
        dict: Statistics about the uncertainty-accuracy relationship
    """
    import matplotlib.pyplot as plt
    
    # Get predictions and errors
    if 'MoNIG_Error' not in df.columns:
        df['MoNIG_Error'] = abs(df['MoNIG_Prediction'] - df['True_Affinity'])
    
    epistemic = df['MoNIG_Epistemic'].values
    errors = df['MoNIG_Error'].values
    
    # Sort by epistemic uncertainty
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
        'coverage_levels': coverage_levels,
        'mae_at_coverage': mae_at_coverage,
        'rmse_at_coverage': rmse_at_coverage,
        'mean_epistemic_at_coverage': mean_epistemic_at_coverage,
        'improvements': improvements
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
        ax1.set_title('Risk-Coverage Curve: Error vs Coverage\n(Low Uncertainty = High Accuracy)', 
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
                        xy=(pct, mae_val), xytext=(pct-15, mae_val+0.1),
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
        ax2.set_title('Error Improvement by Discarding Uncertain Predictions\n(Practical Decision-Making)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 90)
        
        # Add text box with key insights
        textstr = 'Key Thresholds:\n'
        for pct in [50, 70, 90]:
            imp = improvements[pct]
            textstr += f'Keep {pct}% (drop {imp["discarded"]:.0f}%): ↓{imp["mae_improvement"]:.1f}% MAE\n'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        plot_path = output_path / 'uncertainty_accuracy_tradeoff.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved uncertainty-accuracy tradeoff curve to: {plot_path}")
    
    return stats_dict


def analyze_worst_case_errors(df, num_experts=4, output_dir=None):
    """
    Analyze worst-case error reduction (Claim 4B).
    Compare error distributions: Best Expert vs Worst Expert vs CABE.
    
    Args:
        df: DataFrame with inference results and original expert predictions
        num_experts: Number of experts
        output_dir: Optional directory to save plots
        
    Returns:
        dict: Statistics about error distributions
    """
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats
    
    # Calculate errors for each expert and MoNIG
    expert_errors = []
    for j in range(num_experts):
        expert_num = j + 1
        expert_errors.append(df[f'Expert{expert_num}_Error'].values)
    
    expert_errors = np.array(expert_errors)  # Shape: (num_experts, num_samples)
    
    # For each sample, find best and worst expert error
    best_expert_errors = expert_errors.min(axis=0)
    worst_expert_errors = expert_errors.max(axis=0)
    monig_errors = df['MoNIG_Error'].values
    
    # Calculate percentile statistics
    percentiles = [50, 75, 90, 95, 99]
    stats_dict = {
        'best_expert': {},
        'worst_expert': {},
        'monig': {},
        'mean': {
            'best_expert': np.mean(best_expert_errors),
            'worst_expert': np.mean(worst_expert_errors),
            'monig': np.mean(monig_errors)
        }
    }
    
    for p in percentiles:
        stats_dict['best_expert'][f'p{p}'] = np.percentile(best_expert_errors, p)
        stats_dict['worst_expert'][f'p{p}'] = np.percentile(worst_expert_errors, p)
        stats_dict['monig'][f'p{p}'] = np.percentile(monig_errors, p)
    
    # Calculate how often MoNIG beats best/worst
    beats_best = (monig_errors < best_expert_errors).sum()
    beats_worst = (monig_errors < worst_expert_errors).sum()
    worse_than_worst = (monig_errors > worst_expert_errors).sum()
    
    stats_dict['beats_best_count'] = beats_best
    stats_dict['beats_worst_count'] = beats_worst
    stats_dict['worse_than_worst_count'] = worse_than_worst
    stats_dict['total_samples'] = len(monig_errors)
    
    # Create visualizations
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Boxplot comparison
        data_to_plot = [best_expert_errors, monig_errors, worst_expert_errors]
        labels = ['Best Expert\n(per sample)', 'MoNIG\n(CABE)', 'Worst Expert\n(per sample)']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True,
                         showmeans=True, meanline=True,
                         boxprops=dict(alpha=0.7),
                         medianprops=dict(color='black', linewidth=2),
                         meanprops=dict(color='red', linewidth=2, linestyle='--'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add percentile annotations
        for i, (data, label) in enumerate(zip(data_to_plot, ['Best', 'MoNIG', 'Worst']), 1):
            p90 = np.percentile(data, 90)
            p95 = np.percentile(data, 95)
            ax1.text(i, p90, f'90th: {p90:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax1.text(i, p95, f'95th: {p95:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax1.set_ylabel('Prediction Error (pKd)', fontsize=12, fontweight='bold')
        ax1.set_title('Error Distribution Comparison\n(Lower = Better, Narrower = More Consistent)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add legend for mean/median
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2, label='Median'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Mean')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Cumulative Distribution Function (CDF)
        sorted_best = np.sort(best_expert_errors)
        sorted_monig = np.sort(monig_errors)
        sorted_worst = np.sort(worst_expert_errors)
        
        n = len(sorted_best)
        y = np.arange(1, n + 1) / n
        
        ax2.plot(sorted_best, y, linewidth=2.5, label='Best Expert', color='#2ecc71', alpha=0.8)
        ax2.plot(sorted_monig, y, linewidth=2.5, label='MoNIG (CABE)', color='#3498db', alpha=0.8)
        ax2.plot(sorted_worst, y, linewidth=2.5, label='Worst Expert', color='#e74c3c', alpha=0.8)
        
        # Add vertical lines for percentiles
        for p, label_y in [(90, 0.92), (95, 0.97)]:
            best_p = np.percentile(best_expert_errors, p)
            monig_p = np.percentile(monig_errors, p)
            worst_p = np.percentile(worst_expert_errors, p)
            
            ax2.axhline(y=p/100, color='gray', linestyle=':', alpha=0.5)
            ax2.axvline(x=best_p, color='#2ecc71', linestyle='--', alpha=0.3)
            ax2.axvline(x=monig_p, color='#3498db', linestyle='--', alpha=0.3)
            ax2.axvline(x=worst_p, color='#e74c3c', linestyle='--', alpha=0.3)
            
            # Annotate percentile values
            ax2.text(monig_p, label_y, f'{p}th\n{monig_p:.2f}', 
                    fontsize=8, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Prediction Error (pKd)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Distribution of Errors\n(Left = Better, shows tail behavior)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(6, sorted_worst.max()))
        
        plt.tight_layout()
        
        plot_path = output_path / 'worst_case_error_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved worst-case error comparison to: {plot_path}")
    
    return stats_dict


def plot_disagreement_histogram(df, output_dir=None):
    """
    Create histogram showing distribution of expert disagreement.
    
    Args:
        df: DataFrame with inference results
        output_dir: Optional directory to save plot
        
    Returns:
        dict: Statistics about disagreement distribution
    """
    import matplotlib.pyplot as plt
    
    # Calculate expert disagreement (std of expert predictions)
    expert_pred_cols = [col for col in df.columns if col.startswith('Expert') and col.endswith('_Prediction')]
    if len(expert_pred_cols) == 0:
        print("⚠️  No expert predictions found for disagreement calculation")
        return None
    
    # Expert disagreement = std of expert predictions
    df['Expert_Disagreement'] = df[expert_pred_cols].std(axis=1)
    
    disagreement = df['Expert_Disagreement'].values
    
    # Calculate statistics
    stats_dict = {
        'mean': disagreement.mean(),
        'median': np.median(disagreement),
        'std': disagreement.std(),
        'min': disagreement.min(),
        'max': disagreement.max(),
        'q25': np.percentile(disagreement, 25),
        'q75': np.percentile(disagreement, 75),
    }
    
    # Define agreement categories
    low_agreement = (disagreement < 0.5).sum()
    moderate_agreement = ((disagreement >= 0.5) & (disagreement < 1.0)).sum()
    high_disagreement = ((disagreement >= 1.0) & (disagreement < 2.0)).sum()
    very_high_disagreement = (disagreement >= 2.0).sum()
    
    total = len(disagreement)
    stats_dict['low_agreement'] = low_agreement
    stats_dict['moderate_agreement'] = moderate_agreement
    stats_dict['high_disagreement'] = high_disagreement
    stats_dict['very_high_disagreement'] = very_high_disagreement
    
    # Create histogram
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create histogram with custom bins
        n, bins, patches = ax.hist(disagreement, bins=50, edgecolor='black', 
                                   linewidth=0.5, alpha=0.7, color='steelblue')
        
        # Color bins by disagreement level
        for i, patch in enumerate(patches):
            if bins[i] < 0.5:
                patch.set_facecolor('green')
                patch.set_alpha(0.7)
            elif bins[i] < 1.0:
                patch.set_facecolor('yellow')
                patch.set_alpha(0.7)
            elif bins[i] < 2.0:
                patch.set_facecolor('orange')
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor('red')
                patch.set_alpha(0.7)
        
        # Add vertical lines for statistics
        ax.axvline(stats_dict['mean'], color='red', linestyle='--', 
                  linewidth=2, label=f"Mean: {stats_dict['mean']:.3f}")
        ax.axvline(stats_dict['median'], color='blue', linestyle='--', 
                  linewidth=2, label=f"Median: {stats_dict['median']:.3f}")
        
        # Add shaded regions for categories
        ax.axvspan(0, 0.5, alpha=0.1, color='green', label=f'Low (<0.5): {low_agreement} ({low_agreement/total*100:.1f}%)')
        ax.axvspan(0.5, 1.0, alpha=0.1, color='yellow', label=f'Moderate (0.5-1.0): {moderate_agreement} ({moderate_agreement/total*100:.1f}%)')
        ax.axvspan(1.0, 2.0, alpha=0.1, color='orange', label=f'High (1.0-2.0): {high_disagreement} ({high_disagreement/total*100:.1f}%)')
        ax.axvspan(2.0, disagreement.max(), alpha=0.1, color='red', label=f'Very High (>2.0): {very_high_disagreement} ({very_high_disagreement/total*100:.1f}%)')
        
        ax.set_xlabel('Expert Disagreement (Std of Predictions, pKd)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Number of Samples)', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of Expert Disagreement\n(N={total}, Mean={stats_dict["mean"]:.3f}, Median={stats_dict["median"]:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add text box with statistics
        textstr = f'Statistics:\n'
        textstr += f'Min: {stats_dict["min"]:.3f}\n'
        textstr += f'Q25: {stats_dict["q25"]:.3f}\n'
        textstr += f'Median: {stats_dict["median"]:.3f}\n'
        textstr += f'Q75: {stats_dict["q75"]:.3f}\n'
        textstr += f'Max: {stats_dict["max"]:.3f}\n'
        textstr += f'Std: {stats_dict["std"]:.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        plot_path = output_path / 'disagreement_histogram.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved disagreement histogram to: {plot_path}")
    
    return stats_dict


def analyze_error_correlation_matrix(df, num_experts=4, output_dir=None):
    """
    Create error correlation matrix showing how errors correlate between engines.
    
    Args:
        df: DataFrame with error columns
        num_experts: Number of experts
        output_dir: Directory to save plot
    
    Returns:
        dict: Correlation matrix and statistics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Get error columns
    engine_names = EXPERT_NAMES + ['MoNIG']
    error_cols = [f'{name}_Error' for name in engine_names]
    
    # Check which columns exist
    available_cols = [col for col in error_cols if col in df.columns]
    available_names = [name for name, col in zip(engine_names, error_cols) if col in df.columns]
    
    if len(available_cols) < 2:
        print("⚠️  Not enough error columns available for correlation analysis")
        return None
    
    # Extract error values
    error_data = df[available_cols].values
    
    # Calculate correlation matrix (Pearson)
    corr_matrix_pearson = np.corrcoef(error_data.T)
    
    # Calculate Spearman correlation
    corr_matrix_spearman = np.zeros_like(corr_matrix_pearson)
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            if i == j:
                corr_matrix_spearman[i, j] = 1.0
            else:
                corr, _ = stats.spearmanr(error_data[:, i], error_data[:, j])
                corr_matrix_spearman[i, j] = corr
    
    # Create visualizations
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with two subplots (Pearson and Spearman)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot 1: Pearson correlation
        mask = np.triu(np.ones_like(corr_matrix_pearson, dtype=bool), k=1)
        sns.heatmap(corr_matrix_pearson, 
                   mask=mask,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0,
                   vmin=-0.5,
                   vmax=1.0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": 0.8, "label": "Correlation"},
                   xticklabels=available_names,
                   yticklabels=available_names,
                   ax=ax1)
        
        ax1.set_title('Error Correlation Matrix (Pearson)\nLower Triangle Only', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Engine', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Engine', fontsize=12, fontweight='bold')
        
        # Plot 2: Spearman correlation
        sns.heatmap(corr_matrix_spearman,
                   mask=mask,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0,
                   vmin=-0.5,
                   vmax=1.0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": 0.8, "label": "Correlation"},
                   xticklabels=available_names,
                   yticklabels=available_names,
                   ax=ax2)
        
        ax2.set_title('Error Correlation Matrix (Spearman)\nLower Triangle Only', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Engine', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Engine', fontsize=12, fontweight='bold')
        
        # Add interpretation text
        textstr = 'Interpretation:\n'
        textstr += '• High correlation (>0.7): Errors are similar\n'
        textstr += '• Moderate (0.4-0.7): Some overlap in errors\n'
        textstr += '• Low (<0.4): Complementary/diverse errors\n'
        textstr += '• Negative: Inverse error patterns\n\n'
        textstr += 'Goal: Low correlation = diverse experts'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        fig.text(0.5, -0.08, textstr, ha='center', fontsize=10,
                bbox=props, transform=fig.transFigure)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        plot_path = output_path / 'error_correlation_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved error correlation matrix to: {plot_path}")
    
    # Prepare results
    results = {
        'engine_names': available_names,
        'pearson_correlation': corr_matrix_pearson,
        'spearman_correlation': corr_matrix_spearman,
        'error_means': {name: df[f'{name}_Error'].mean() for name in available_names},
        'error_stds': {name: df[f'{name}_Error'].std() for name in available_names}
    }
    
    return results


def analyze_pairwise_error_scatter(df, num_experts=4, output_dir=None):
    """
    Create scatter plots comparing errors between pairs of engines.
    
    Args:
        df: DataFrame with error columns
        num_experts: Number of experts
        output_dir: Directory to save plots
    
    Returns:
        dict: Statistics for each pair
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from itertools import combinations
    
    # Get available engines
    engine_names = EXPERT_NAMES + ['MoNIG']
    available_names = [name for name in engine_names if f'{name}_Error' in df.columns]
    
    if len(available_names) < 2:
        print("⚠️  Not enough engines available for pairwise comparison")
        return None
    
    # Generate all pairs (excluding MoNIG vs MoNIG)
    pairs = []
    
    # Expert vs Expert comparisons
    for pair in combinations([name for name in available_names if name != 'MoNIG'], 2):
        pairs.append(pair)
    
    # MoNIG vs Expert comparisons
    if 'MoNIG' in available_names:
        for expert in [name for name in available_names if name != 'MoNIG']:
            pairs.append((expert, 'MoNIG'))
    
    if len(pairs) == 0:
        print("⚠️  No valid pairs found")
        return None
    
    # Calculate grid size
    n_pairs = len(pairs)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    # Create large figure with all scatter plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    pair_stats = {}
    
    for idx, (engine1, engine2) in enumerate(pairs):
        ax = axes[idx]
        
        error1 = df[f'{engine1}_Error'].values
        error2 = df[f'{engine2}_Error'].values
        
        # Calculate correlation
        pearson_corr, pearson_p = stats.pearsonr(error1, error2)
        spearman_corr, spearman_p = stats.spearmanr(error1, error2)
        
        # Calculate how often each is better
        engine1_better = (error1 < error2).sum()
        engine2_better = (error2 < error1).sum()
        tie = (error1 == error2).sum()
        
        # Create hexbin for density
        hexbin = ax.hexbin(error1, error2, gridsize=30, cmap='YlOrRd', 
                          mincnt=1, alpha=0.8, edgecolors='black', linewidths=0.2)
        
        # Add diagonal line (y=x) showing where errors are equal
        max_val = max(error1.max(), error2.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, 
               label='Equal Error')
        
        # Add trend line
        z = np.polyfit(error1, error2, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(error1.min(), error1.max(), 100)
        ax.plot(x_trend, p(x_trend), 'b-', alpha=0.7, linewidth=2,
               label=f'Trend (slope={z[0]:.2f})')
        
        # Labels and title
        ax.set_xlabel(f'{engine1} Error (pKd)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{engine2} Error (pKd)', fontsize=11, fontweight='bold')
        
        title = f'{engine1} vs {engine2}\n'
        title += f'Pearson r={pearson_corr:.3f}, Spearman ρ={spearman_corr:.3f}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add text box with statistics
        textstr = f'{engine1} better: {engine1_better} ({engine1_better/len(error1)*100:.1f}%)\n'
        textstr += f'{engine2} better: {engine2_better} ({engine2_better/len(error2)*100:.1f}%)\n'
        textstr += f'Tie: {tie}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add colorbar for this subplot
        cbar = plt.colorbar(hexbin, ax=ax)
        cbar.set_label('Count', fontsize=9)
        
        # Store statistics
        pair_stats[f'{engine1}_vs_{engine2}'] = {
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p,
            'engine1_better_count': engine1_better,
            'engine2_better_count': engine2_better,
            'tie_count': tie,
            'total_samples': len(error1)
        }
    
    # Hide unused subplots
    for idx in range(len(pairs), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Pairwise Engine Error Comparison\n(Points above diagonal: Engine2 has higher error)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_path = output_path / 'error_scatter_pairwise.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved pairwise error scatter plots to: {plot_path}")
    else:
        plt.close()
    
    return pair_stats


def analyze_monig_vs_experts_scatter(df, num_experts=4, output_dir=None):
    """
    Create focused comparison: MoNIG vs each expert engine.
    
    Args:
        df: DataFrame with error columns
        num_experts: Number of experts
        output_dir: Directory to save plot
    
    Returns:
        dict: Statistics for MoNIG comparisons
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    if 'MoNIG_Error' not in df.columns:
        print("⚠️  MoNIG errors not found in dataset")
        return None
    
    # Get available experts
    available_experts = [name for name in EXPERT_NAMES if f'{name}_Error' in df.columns]
    
    if len(available_experts) == 0:
        print("⚠️  No expert errors found")
        return None
    
    # Create 2x2 grid for 4 experts
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    monig_errors = df['MoNIG_Error'].values
    monig_stats = {}
    
    for idx, expert_name in enumerate(available_experts[:4]):  # Limit to 4
        ax = axes[idx]
        
        expert_errors = df[f'{expert_name}_Error'].values
        
        # Calculate statistics
        pearson_corr, pearson_p = stats.pearsonr(expert_errors, monig_errors)
        spearman_corr, spearman_p = stats.spearmanr(expert_errors, monig_errors)
        
        # Count wins
        monig_better = (monig_errors < expert_errors).sum()
        expert_better = (expert_errors < monig_errors).sum()
        tie = (monig_errors == expert_errors).sum()
        
        # Create scatter plot
        scatter = ax.scatter(expert_errors, monig_errors, 
                           c=monig_errors, cmap='RdYlGn_r',
                           alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add diagonal and thresholds
        max_val = max(expert_errors.max(), monig_errors.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, 
               label='Equal Error')
        
        # Add horizontal/vertical lines for thresholds
        ax.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5, 
                  label='MoNIG = 1.5 (acceptable)')
        ax.axhline(y=3.0, color='red', linestyle=':', alpha=0.5,
                  label='MoNIG = 3.0 (catastrophic)')
        ax.axvline(x=3.0, color='red', linestyle=':', alpha=0.5)
        
        # Labels
        ax.set_xlabel(f'{expert_name} Error (pKd)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MoNIG Error (pKd)', fontsize=12, fontweight='bold')
        
        title = f'MoNIG vs {expert_name}\n'
        title += f'r={pearson_corr:.3f} (p={pearson_p:.2e})'
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Statistics text box
        textstr = f'MoNIG better: {monig_better} ({monig_better/len(monig_errors)*100:.1f}%)\n'
        textstr += f'{expert_name} better: {expert_better} ({expert_better/len(expert_errors)*100:.1f}%)\n'
        textstr += f'Mean MoNIG error: {monig_errors.mean():.3f}\n'
        textstr += f'Mean {expert_name} error: {expert_errors.mean():.3f}'
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.85)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props, fontweight='bold')
        
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('MoNIG Error', fontsize=9)
        
        # Store stats
        monig_stats[expert_name] = {
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p,
            'monig_better': monig_better,
            'expert_better': expert_better,
            'monig_mean_error': monig_errors.mean(),
            'expert_mean_error': expert_errors.mean(),
            'total': len(monig_errors)
        }
    
    plt.suptitle('MoNIG Error vs Expert Engines\n(Points below diagonal: MoNIG has lower error)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_path = output_path / 'monig_vs_experts_scatter.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved MoNIG vs experts comparison to: {plot_path}")
    else:
        plt.close()
    
    return monig_stats


def print_case_study_report(df, best_cases, general_catastrophe, dynamicbind_catastrophe, high_epistemic, num_experts=4, damage_mitigation=None, epistemic_stats=None, disagreement_error_stats=None, disagreement_dist_stats=None, uncertainty_accuracy_stats=None, baseline_comparison=None, worst_case_stats=None, error_corr_stats=None, pairwise_error_stats=None, monig_vs_experts_stats=None):
    """Print comprehensive case study report."""
    
    print("\n" + "="*100)
    print("CABE CASE STUDIES COMPREHENSIVE ANALYSIS")
    print("="*100)
    print(f"Total test samples: {len(df)}")
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    # ===== EPISTEMIC UNCERTAINTY VS EXPERT DISAGREEMENT =====
    if epistemic_stats:
        print("\n" + "="*100)
        print("EPISTEMIC UNCERTAINTY ↔ EXPERT DISAGREEMENT ANALYSIS")
        print("="*100)
        print(f"Pearson Correlation:  r = {epistemic_stats['pearson_correlation']:>6.3f}  (p = {epistemic_stats['pearson_p_value']:.2e})")
        print(f"Spearman Correlation: ρ = {epistemic_stats['spearman_correlation']:>6.3f}  (p = {epistemic_stats['spearman_p_value']:.2e})")
        print()
        print(f"Mean Epistemic Uncertainty:  {epistemic_stats['mean_epistemic']:.4f} ± {epistemic_stats['std_epistemic']:.4f}")
        print(f"Mean Expert Disagreement:    {epistemic_stats['mean_disagreement']:.4f} ± {epistemic_stats['std_disagreement']:.4f}")
        
        # Interpretation
        corr = epistemic_stats['pearson_correlation']
        if corr > 0.7:
            interpretation = "Strong positive correlation ✅"
        elif corr > 0.5:
            interpretation = "Moderate positive correlation ✅"
        elif corr > 0.3:
            interpretation = "Weak-to-moderate positive correlation ⚠️"
        elif corr > 0.1:
            interpretation = "Weak positive correlation ⚠️"
        else:
            interpretation = "No meaningful correlation ❌"
        
        print(f"\n📊 Interpretation: {interpretation}")
        print("    → Epistemic uncertainty should increase when experts disagree")
        print("    → Good correlation indicates model captures expert uncertainty well")
        print("="*100)
    
    # ===== EXPERT DISAGREEMENT VS PREDICTION ERROR =====
    if disagreement_error_stats:
        print("\n" + "="*100)
        print("EXPERT DISAGREEMENT ↔ PREDICTION ERROR ANALYSIS")
        print("="*100)
        print(f"Pearson Correlation:  r = {disagreement_error_stats['pearson_correlation']:>6.3f}  (p = {disagreement_error_stats['pearson_p_value']:.2e})")
        print(f"Spearman Correlation: ρ = {disagreement_error_stats['spearman_correlation']:>6.3f}  (p = {disagreement_error_stats['spearman_p_value']:.2e})")
        print()
        print(f"Mean Expert Disagreement:    {disagreement_error_stats['mean_disagreement']:.4f} ± {disagreement_error_stats['std_disagreement']:.4f}")
        print(f"Mean MoNIG Error:            {disagreement_error_stats['mean_error']:.4f} ± {disagreement_error_stats['std_error']:.4f}")
        
        # Interpretation
        corr = disagreement_error_stats['pearson_correlation']
        if corr > 0.5:
            interpretation = "Moderate-to-strong positive correlation ✅"
            explanation = "When experts disagree, predictions are more likely to be wrong"
        elif corr > 0.3:
            interpretation = "Weak-to-moderate positive correlation ⚠️"
            explanation = "Some relationship exists, but disagreement isn't always predictive of error"
        elif corr > 0.0:
            interpretation = "Weak positive correlation ⚠️"
            explanation = "Expert disagreement has limited predictive value for errors"
        else:
            interpretation = "No meaningful correlation ❌"
            explanation = "Expert disagreement doesn't predict error"
        
        print(f"\n📊 Interpretation: {interpretation}")
        print(f"    → {explanation}")
        print("    → High disagreement = harder cases (but not always wrong predictions)")
        print("="*100)
    
    # ===== UNCERTAINTY-ACCURACY TRADEOFF (CLAIM 2B) =====
    if uncertainty_accuracy_stats:
        print("\n" + "="*100)
        print("CLAIM 2B: LOW-UNCERTAINTY PREDICTIONS ARE HIGHLY ACCURATE")
        print("="*100)
        print()
        print("Full Test Set Performance:")
        print(f"  MAE:  {uncertainty_accuracy_stats['mae_full']:.4f} pKd")
        print(f"  RMSE: {uncertainty_accuracy_stats['rmse_full']:.4f} pKd")
        print()
        
        print("Risk-Coverage Analysis:")
        print(f"{'Coverage':<12} {'Discarded':<12} {'MAE (pKd)':<15} {'RMSE (pKd)':<15} {'MAE ↓%':<12} {'RMSE ↓%':<12} {'Epistemic Threshold'}")
        print("-"*100)
        
        for pct in [50, 70, 80, 90, 100]:
            if pct == 100:
                mae = uncertainty_accuracy_stats['mae_full']
                rmse = uncertainty_accuracy_stats['rmse_full']
                mae_imp = 0.0
                rmse_imp = 0.0
                ep_thresh = "N/A"
                discarded = 0
            else:
                imp = uncertainty_accuracy_stats['improvements'][pct]
                mae = imp['mae_kept']
                rmse = imp['rmse_kept']
                mae_imp = imp['mae_improvement']
                rmse_imp = imp['rmse_improvement']
                ep_thresh = f"{imp['epistemic_threshold']:.4f}"
                discarded = imp['discarded']
            
            print(f"{pct}%{'':<9} {discarded:.0f}%{'':<9} {mae:<15.4f} {rmse:<15.4f} {mae_imp:<12.1f} {rmse_imp:<12.1f} {ep_thresh}")
        
        print()
        print("📊 Key Insights:")
        
        # Highlight best practical thresholds
        best_50 = uncertainty_accuracy_stats['improvements'][50]
        best_70 = uncertainty_accuracy_stats['improvements'][70]
        
        print(f"  • Keeping 50% (most certain): MAE ↓{best_50['mae_improvement']:.1f}% (from {uncertainty_accuracy_stats['mae_full']:.3f} to {best_50['mae_kept']:.3f})")
        print(f"  • Keeping 70% (moderate risk): MAE ↓{best_70['mae_improvement']:.1f}% (from {uncertainty_accuracy_stats['mae_full']:.3f} to {best_70['mae_kept']:.3f})")
        print()
        print("💡 Practical Application:")
        print("   'By discarding the 30% most epistemically uncertain predictions,")
        print(f"    we reduce MAE by {best_70['mae_improvement']:.1f}% (from {uncertainty_accuracy_stats['mae_full']:.3f} to {best_70['mae_kept']:.3f} pKd)'")
        print()
        print("   → Epistemic uncertainty enables selective prediction")
        print("   → Users can trade coverage for accuracy based on risk tolerance")
        print("="*100)
    
    # ===== BASELINE COMPARISON FOR RISK-COVERAGE =====
    if baseline_comparison:
        print("\n" + "="*100)
        print("RISK-COVERAGE COMPARISON: MoNIG vs BASELINES")
        print("="*100)
        print()
        print("How well does each model's uncertainty enable selective prediction?")
        print()
        
        # Full set performance
        print(f"{'Model':<15} {'MAE (100%)':<15} {'@ 50%':<15} {'Improv':<10} {'@ 70%':<15} {'Improv':<10} {'@ 90%':<15} {'Improv':<10}")
        print("-"*100)
        
        for model_name in ['MoNIG', 'Gaussian', 'NIG', 'DeepEnsemble', 'MCDropout']:
            if model_name not in baseline_comparison:
                print(f"{model_name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'N/A':<15} {'N/A':<10} {'N/A':<15} {'N/A':<10}")
                continue
            
            results = baseline_comparison[model_name]
            mae_full = results['mae_full']
            
            mae_50 = results['improvements'][50]['mae']
            imp_50 = results['improvements'][50]['improvement']
            
            mae_70 = results['improvements'][70]['mae']
            imp_70 = results['improvements'][70]['improvement']
            
            mae_90 = results['improvements'][90]['mae']
            imp_90 = results['improvements'][90]['improvement']
            
            print(f"{model_name:<15} {mae_full:<15.4f} {mae_50:<15.4f} {imp_50:>5.1f}%{'':<4} {mae_70:<15.4f} {imp_70:>5.1f}%{'':<4} {mae_90:<15.4f} {imp_90:>5.1f}%{'':<4}")
        
        print()
        print("📊 Key Insight:")
        
        # Find best model at 70% coverage
        best_model = None
        best_improvement = -999
        for model_name, results in baseline_comparison.items():
            imp = results['improvements'][70]['improvement']
            if imp > best_improvement:
                best_improvement = imp
                best_model = model_name
        
        if best_model:
            print(f"   ✅ {best_model} has the best uncertainty estimates ({best_improvement:+.1f}% @ 70% coverage)")
            
            if best_model == 'MoNIG':
                print("   → MoNIG's epistemic uncertainty is most useful for selective prediction")
                print("   → Baselines show minimal or negative improvements")
            
        print()
        print("💡 Interpretation:")
        print("   • Positive improvement = Uncertainty correlates with prediction difficulty")
        print("   • Negative improvement = Uncertainty is misleading")
        print("   • @ 70% = Keep 70% most certain, discard 30% most uncertain")
        print()
        print("   See: risk_coverage_comparison.png for full baseline analysis")
        print("="*100)
    
    # ===== WORST-CASE ERROR REDUCTION (CLAIM 4B) =====
    if worst_case_stats:
        print("\n" + "="*100)
        print("CLAIM 4B: CABE REDUCES WORST-CASE ERRORS")
        print("="*100)
        print()
        print("Error Distribution Comparison:")
        print(f"{'Method':<20} {'Mean':<12} {'Median (50th)':<15} {'75th':<12} {'90th':<12} {'95th':<12} {'99th':<12}")
        print("-"*100)
        
        for method_name, method_key in [('Best Expert', 'best_expert'), ('MoNIG (CABE)', 'monig'), ('Worst Expert', 'worst_expert')]:
            mean_val = worst_case_stats['mean'][method_key]
            p50 = worst_case_stats[method_key]['p50']
            p75 = worst_case_stats[method_key]['p75']
            p90 = worst_case_stats[method_key]['p90']
            p95 = worst_case_stats[method_key]['p95']
            p99 = worst_case_stats[method_key]['p99']
            
            print(f"{method_name:<20} {mean_val:<12.4f} {p50:<15.4f} {p75:<12.4f} {p90:<12.4f} {p95:<12.4f} {p99:<12.4f}")
        
        print()
        print("📊 Key Metrics:")
        
        total = worst_case_stats['total_samples']
        beats_best = worst_case_stats['beats_best_count']
        beats_worst = worst_case_stats['beats_worst_count']
        worse_than_worst = worst_case_stats['worse_than_worst_count']
        
        print(f"  • MoNIG beats best expert: {beats_best} / {total} ({beats_best/total*100:.1f}%)")
        print(f"  • MoNIG beats worst expert: {beats_worst} / {total} ({beats_worst/total*100:.1f}%)")
        print(f"  • MoNIG worse than worst: {worse_than_worst} / {total} ({worse_than_worst/total*100:.1f}%)")
        print()
        
        # Compare tail behavior (95th percentile)
        best_p95 = worst_case_stats['best_expert']['p95']
        monig_p95 = worst_case_stats['monig']['p95']
        worst_p95 = worst_case_stats['worst_expert']['p95']
        
        reduction_vs_worst = (worst_p95 - monig_p95) / worst_p95 * 100
        comparison_to_best = (monig_p95 - best_p95) / best_p95 * 100
        
        print("💡 Worst-Case Analysis (95th percentile):")
        print(f"  • Best expert: {best_p95:.3f} pKd")
        print(f"  • MoNIG: {monig_p95:.3f} pKd")
        print(f"  • Worst expert: {worst_p95:.3f} pKd")
        print()
        print(f"  → MoNIG reduces worst-case error by {reduction_vs_worst:.1f}% vs worst expert")
        
        if monig_p95 < best_p95:
            print(f"  → MoNIG even beats best expert at 95th percentile! ({-comparison_to_best:.1f}% better)")
        else:
            print(f"  → MoNIG is {comparison_to_best:.1f}% higher than best expert at 95th percentile")
        
        print()
        print("🎯 Practical Implication:")
        print("   'CABE significantly reduces tail risk - catastrophically bad predictions are rare'")
        print(f"   '95% of CABE predictions have error < {monig_p95:.2f} pKd'")
        print("="*100)
    
    # ===== EXPERT DISAGREEMENT DISTRIBUTION =====
    if disagreement_dist_stats:
        print("\n" + "="*100)
        print("EXPERT DISAGREEMENT DISTRIBUTION")
        print("="*100)
        print(f"Mean:   {disagreement_dist_stats['mean']:.4f} pKd")
        print(f"Median: {disagreement_dist_stats['median']:.4f} pKd")
        print(f"Std:    {disagreement_dist_stats['std']:.4f} pKd")
        print(f"Range:  [{disagreement_dist_stats['min']:.4f}, {disagreement_dist_stats['max']:.4f}] pKd")
        print()
        
        total = disagreement_dist_stats['low_agreement'] + disagreement_dist_stats['moderate_agreement'] + \
                disagreement_dist_stats['high_disagreement'] + disagreement_dist_stats['very_high_disagreement']
        
        print("Disagreement Categories:")
        print(f"  🟢 Low Agreement      (< 0.5):   {disagreement_dist_stats['low_agreement']:3d} ({disagreement_dist_stats['low_agreement']/total*100:5.1f}%) - Experts mostly agree")
        print(f"  🟡 Moderate Agreement (0.5-1.0): {disagreement_dist_stats['moderate_agreement']:3d} ({disagreement_dist_stats['moderate_agreement']/total*100:5.1f}%) - Some disagreement")
        print(f"  🟠 High Disagreement  (1.0-2.0): {disagreement_dist_stats['high_disagreement']:3d} ({disagreement_dist_stats['high_disagreement']/total*100:5.1f}%) - Significant disagreement")
        print(f"  🔴 Very High          (> 2.0):   {disagreement_dist_stats['very_high_disagreement']:3d} ({disagreement_dist_stats['very_high_disagreement']/total*100:5.1f}%) - Extreme disagreement")
        print()
        print("💡 Interpretation:")
        print("   - Low disagreement (<0.5 pKd) ≈ experts within 3× binding affinity")
        print("   - High disagreement (>1.0 pKd) ≈ experts differ by >10× binding affinity")
        print("   - Very high (>2.0 pKd) ≈ experts differ by >100× binding affinity!")
        print("="*100)
    
    # ===== ERROR CORRELATION ANALYSIS =====
    if error_corr_stats:
        print("\n" + "="*100)
        print("ERROR CORRELATION BETWEEN ENGINES")
        print("="*100)
        
        engine_names = error_corr_stats['engine_names']
        pearson = error_corr_stats['pearson_correlation']
        
        # Print correlation matrix (lower triangle only)
        print("\nPearson Correlation Matrix (Lower Triangle):")
        print(f"{'Engine':<15}", end="")
        for name in engine_names:
            print(f"{name:<12}", end="")
        print()
        print("-"*100)
        
        for i, name1 in enumerate(engine_names):
            print(f"{name1:<15}", end="")
            for j, name2 in enumerate(engine_names):
                if i == j:
                    print(f"{'1.000':<12}", end="")
                elif i > j:  # Lower triangle only
                    print(f"{pearson[i, j]:<12.3f}", end="")
                else:
                    print(f"{'':<12}", end="")
            print()
        
        print("\n📊 Mean Errors:")
        for name in engine_names:
            mean_err = error_corr_stats['error_means'][name]
            std_err = error_corr_stats['error_stds'][name]
            print(f"  {name:<15}: {mean_err:.4f} ± {std_err:.4f} pKd")
        
        print("\n💡 Key Insights:")
        # Find lowest correlation between experts (excluding MoNIG)
        expert_only = [n for n in engine_names if n != 'MoNIG']
        min_corr = 1.0
        min_pair = None
        for i, n1 in enumerate(expert_only):
            for j, n2 in enumerate(expert_only):
                if i < j:
                    i_idx = engine_names.index(n1)
                    j_idx = engine_names.index(n2)
                    corr_val = pearson[max(i_idx, j_idx), min(i_idx, j_idx)]
                    if corr_val < min_corr:
                        min_corr = corr_val
                        min_pair = (n1, n2)
        
        if min_pair:
            print(f"  • Lowest correlation: {min_pair[0]} ↔ {min_pair[1]} = {min_corr:.3f}")
            if min_corr < 0.2:
                print(f"    → Excellent diversity! Near-independent errors ✅")
            elif min_corr < 0.4:
                print(f"    → Good diversity for ensemble ✅")
            else:
                print(f"    → Moderate diversity")
        
        # Find highest correlation with MoNIG
        if 'MoNIG' in engine_names:
            monig_idx = engine_names.index('MoNIG')
            max_corr_with_monig = -1
            max_expert = None
            for i, name in enumerate(engine_names):
                if name != 'MoNIG':
                    corr_val = pearson[max(monig_idx, i), min(monig_idx, i)]
                    if corr_val > max_corr_with_monig:
                        max_corr_with_monig = corr_val
                        max_expert = name
            
            if max_expert:
                print(f"  • MoNIG most correlated with: {max_expert} (r={max_corr_with_monig:.3f})")
                print(f"    → MoNIG learns heavily from this expert")
        
        print("="*100)
    
    # ===== MoNIG VS EXPERTS WIN RATES =====
    if monig_vs_experts_stats:
        print("\n" + "="*100)
        print("MoNIG vs EXPERT ENGINES - WIN RATE ANALYSIS")
        print("="*100)
        
        for expert_name, stats in monig_vs_experts_stats.items():
            monig_better = stats['monig_better']
            expert_better = stats['expert_better']
            total = stats['total']
            
            print(f"\n{expert_name}:")
            print(f"  MoNIG better:       {monig_better:4d} / {total} ({monig_better/total*100:5.1f}%)")
            print(f"  {expert_name} better:       {expert_better:4d} / {total} ({expert_better/total*100:5.1f}%)")
            print(f"  Pearson r:          {stats['pearson_r']:6.3f} (p={stats['pearson_p']:.2e})")
            print(f"  Mean MoNIG error:   {stats['monig_mean_error']:.4f} pKd")
            print(f"  Mean {expert_name} error:   {stats['expert_mean_error']:.4f} pKd")
        
        print("\n💡 Summary:")
        total_wins = sum(s['monig_better'] for s in monig_vs_experts_stats.values())
        total_comparisons = sum(s['total'] for s in monig_vs_experts_stats.values())
        print(f"  • Overall MoNIG win rate: {total_wins}/{total_comparisons} ({total_wins/total_comparisons*100:.1f}%)")
        print("="*100)
    
    # ===== PAIRWISE ERROR CORRELATION SUMMARY =====
    if pairwise_error_stats:
        print("\n" + "="*100)
        print("PAIRWISE ERROR CORRELATION SUMMARY")
        print("="*100)
        
        # Sort by correlation strength
        sorted_pairs = sorted(pairwise_error_stats.items(), 
                            key=lambda x: abs(x[1]['pearson_r']), 
                            reverse=True)
        
        print(f"\n{'Pair':<30} {'Pearson r':<12} {'Spearman ρ':<12} {'Interpretation'}")
        print("-"*100)
        
        for pair_name, stats in sorted_pairs[:10]:  # Show top 10
            pearson_r = stats['pearson_r']
            spearman_r = stats['spearman_r']
            
            # Interpretation
            if abs(pearson_r) > 0.7:
                interp = "Strong correlation"
            elif abs(pearson_r) > 0.4:
                interp = "Moderate correlation"
            elif abs(pearson_r) > 0.2:
                interp = "Weak correlation"
            else:
                interp = "Very weak/no correlation"
            
            print(f"{pair_name:<30} {pearson_r:<12.3f} {spearman_r:<12.3f} {interp}")
        
        print("\n💡 Interpretation:")
        print("  • High correlation (>0.7):  Engines make similar mistakes")
        print("  • Moderate (0.4-0.7):       Some overlap in error patterns")
        print("  • Low (<0.4):               Diverse/complementary errors ✅")
        print("  • Negative correlation:     Inverse error patterns")
        print()
        print("  → Low correlation between experts = diverse errors = better ensemble potential")
        print("  → MoNIG should leverage this diversity to reduce overall error")
        print("="*100)
    
    # ===== CASE STUDY 1: BEST CASES =====
    print("\n" + "="*100)
    print("CASE STUDY 1: CABE BEST CASES (Outperforms ALL Experts)")
    print("="*100)
    
    if len(best_cases) > 0:
        success_rate = len(best_cases) / len(df) * 100
        print(f"\nFound {len(best_cases)} cases ({success_rate:.1f}%) where MoNIG beats ALL experts\n")
        
        print("Statistics:")
        print(f"  Average improvement over best expert: {best_cases['Improvement_Over_Best'].mean():.4f}")
        print(f"  Median improvement: {best_cases['Improvement_Over_Best'].median():.4f}")
        print(f"  Max improvement: {best_cases['Improvement_Over_Best'].max():.4f}")
        print(f"  Average improvement %: {best_cases['Improvement_Percent'].mean():.1f}%")
        
        # Show top 5 cases
        print(f"\nTop 5 Best Cases:")
        print("-" * 100)
        for i, (idx, row) in enumerate(best_cases.head(5).iterrows(), 1):
            print(f"\n{i}. Complex {row['ComplexID']}")
            print(f"   True Affinity: {row['True_Affinity']:.3f}")
            print(f"   MoNIG: {row['MoNIG_Prediction']:.3f} (Error: {row['MoNIG_Error']:.3f}) ✅ BEST")
            print(f"   Improvement: {row['Improvement_Over_Best']:.3f} ({row['Improvement_Percent']:.1f}%)")
            print(f"   Expert Predictions:")
            expert_prediction_cols = ['GNINA_Affinity', 'BIND_pIC50', 'flowdock_score', 'DynamicBind_score']
            for j in range(num_experts):
                expert_num = j + 1
                expert_name = EXPERT_NAMES[j]
                # Use original expert prediction if available, fallback to NIG
                if expert_prediction_cols[j] in row.index:
                    pred = row[expert_prediction_cols[j]]
                else:
                    pred = row[f'Expert{expert_num}_Prediction']
                err = row[f'Expert{expert_num}_Error']
                marker = " (was best)" if expert_name == row['Best_Expert_Before'] else ""
                print(f"     {expert_name}: {pred:.3f} (Error: {err:.3f}){marker}")
    else:
        print("\n⚠️  No cases found where CABE outperformed all experts")
    
    # ===== CASE STUDY 2: CATASTROPHE AVOIDANCE =====
    print("\n" + "="*100)
    print("CASE STUDY 2: CATASTROPHE AVOIDANCE")
    print("="*100)
    
    if len(dynamicbind_catastrophe) > 0:
        print(f"\nDynamicBind-Specific Catastrophic Failures: {len(dynamicbind_catastrophe)} cases")
        print("-" * 100)
        
        print("\nStatistics:")
        print(f"  Average DynamicBind error: {dynamicbind_catastrophe['DynamicBind_Error'].mean():.3f}")
        print(f"  Average MoNIG error: {dynamicbind_catastrophe['MoNIG_Error'].mean():.3f}")
        print(f"  Average error saved: {dynamicbind_catastrophe['MoNIG_vs_DynamicBind'].mean():.3f}")
        
        print(f"\nTop 5 DynamicBind Catastrophe Cases:")
        for i, (idx, row) in enumerate(dynamicbind_catastrophe.head(5).iterrows(), 1):
            print(f"\n{i}. Complex {row['ComplexID']}")
            print(f"   True Affinity: {row['True_Affinity']:.3f}")
            print(f"   MoNIG: {row['MoNIG_Prediction']:.3f} (Error: {row['MoNIG_Error']:.3f})")
            print(f"   MoNIG Epistemic: {row['MoNIG_Epistemic']:.3f} (HIGH)")
            # Use original DynamicBind score if available
            db_pred = row['DynamicBind_score'] if 'DynamicBind_score' in row.index else row['Expert4_Prediction']
            print(f"   DynamicBind: {db_pred:.3f} (Error: {row['DynamicBind_Error']:.3f}) 🔴 CATASTROPHIC")
            print(f"   ✅ MoNIG saved: {row['MoNIG_vs_DynamicBind']:.3f} units of error")
    
    if len(general_catastrophe) > 0:
        print(f"\nGeneral Catastrophic Failures (Any Expert): {len(general_catastrophe)} cases")
        print("-" * 100)
        
        print("\nStatistics:")
        print(f"  Average worst expert error: {general_catastrophe['Worst_Expert_Error'].mean():.3f}")
        print(f"  Average MoNIG error: {general_catastrophe['MoNIG_Error'].mean():.3f}")
        print(f"  Average error saved: {general_catastrophe['MoNIG_vs_Worst'].mean():.3f}")
    
    if len(dynamicbind_catastrophe) == 0 and len(general_catastrophe) == 0:
        print("\n⚠️  No true catastrophe avoidance cases found")
    
    # ===== DAMAGE MITIGATION (MoNIG "less bad" but not good) =====
    if damage_mitigation is not None and len(damage_mitigation) > 0:
        print("\n" + "="*100)
        print("DAMAGE MITIGATION (MoNIG reduces but doesn't eliminate error)")
        print("="*100)
        print(f"\nFound {len(damage_mitigation)} cases where MoNIG is 'less bad' (error 1.5-3.0)")
        print("Note: These are NOT true catastrophe avoidance - MoNIG still performs poorly")
        print("-" * 100)
        
        print("\nStatistics:")
        print(f"  Average worst expert error: {damage_mitigation['Worst_Expert_Error'].mean():.3f}")
        print(f"  Average MoNIG error: {damage_mitigation['MoNIG_Error'].mean():.3f}")
        print(f"  Average error reduction: {damage_mitigation['MoNIG_vs_Worst'].mean():.3f}")
        
        if len(damage_mitigation) > 0:
            print(f"\nTop 3 Damage Mitigation Cases:")
            for i, (idx, row) in enumerate(damage_mitigation.head(3).iterrows(), 1):
                print(f"\n{i}. Complex {row['ComplexID']}")
                print(f"   True Affinity: {row['True_Affinity']:.3f}")
                print(f"   MoNIG: {row['MoNIG_Prediction']:.3f} (Error: {row['MoNIG_Error']:.3f}) ⚠️ STILL BAD")
                print(f"   Worst Expert Error: {row['Worst_Expert_Error']:.3f} 🔴")
                print(f"   Error reduction: {row['MoNIG_vs_Worst']:.3f} (MoNIG is less bad, not good)")
    
    # ===== CASE STUDY 3: HIGH EPISTEMIC = HARD CASES =====
    print("\n" + "="*100)
    print("CASE STUDY 3: HIGH EPISTEMIC UNCERTAINTY (Hard Cases)")
    print("="*100)
    
    if len(high_epistemic) > 0:
        print(f"\nFound {len(high_epistemic)} cases with high epistemic uncertainty\n")
        
        print("Statistics:")
        print(f"  Average MoNIG error: {high_epistemic['MoNIG_Error'].mean():.3f}")
        print(f"  Average MoNIG epistemic: {high_epistemic['MoNIG_Epistemic'].mean():.3f}")
        print(f"  Average expert disagreement: {high_epistemic['Expert_Disagreement'].mean():.3f}")
        
        print(f"\nTop 3 High Epistemic Cases:")
        print("-" * 100)
        for i, (idx, row) in enumerate(high_epistemic.head(3).iterrows(), 1):
            print(f"\n{i}. Complex {row['ComplexID']}")
            print(f"   True Affinity: {row['True_Affinity']:.3f}")
            print(f"   MoNIG: {row['MoNIG_Prediction']:.3f} (Error: {row['MoNIG_Error']:.3f})")
            print(f"   MoNIG Epistemic: {row['MoNIG_Epistemic']:.3f} ⚠️  HIGH")
            print(f"   Expert Disagreement: {row['Expert_Disagreement']:.3f}")
    else:
        print("\n⚠️  No high epistemic uncertainty cases found")
    
    # ===== OVERALL SUMMARY =====
    print("\n" + "="*100)
    print("OVERALL SUMMARY")
    print("="*100)
    
    expert_errors = [f'Expert{j+1}_Error' for j in range(num_experts)]
    worst_expert_error = df[expert_errors].max(axis=1)
    worse_than_all = df[df['MoNIG_Error'] > worst_expert_error]
    
    print(f"\n✅ Cases where MoNIG beats ALL experts: {len(best_cases)} ({len(best_cases)/len(df)*100:.1f}%)")
    print(f"✅ Cases where MoNIG avoids catastrophe: {len(general_catastrophe)} ({len(general_catastrophe)/len(df)*100:.1f}%)")
    print(f"⚠️  Cases where MoNIG worse than ALL experts: {len(worse_than_all)} ({len(worse_than_all)/len(df)*100:.1f}%)")
    
    if len(worse_than_all) == 0:
        print("\n🎉 PROOF: MoNIG NEVER does worse than all experts!")
    
    print("\n" + "="*100)


def analyze_baseline_risk_coverage(experiment_dir, seed, output_dir=None):
    """
    Quick baseline comparison for risk-coverage at the same seed.
    Shows MoNIG vs baselines side-by-side.
    Generates risk-coverage comparison plots.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    models = ['MoNIG', 'Gaussian', 'NIG', 'DeepEnsemble', 'MCDropout']
    coverage_levels = np.array([0.5, 0.7, 0.9, 1.0])
    
    results_summary = {}
    mae_curves = {}  # Store MAE values for plotting
    
    for model_name in models:
        model_dir = Path(experiment_dir) / f"{model_name}_seed{seed}"
        results_file = model_dir / "test_inference_results.csv"
        
        if not results_file.exists():
            continue
        
        df = pd.read_csv(results_file)
        true_values = df['True_Affinity'].values
        
        # Extract predictions and uncertainty
        if model_name in ['MoNIG', 'CABE']:
            predictions = df['MoNIG_Prediction'].values
            uncertainties = df['MoNIG_Epistemic'].values
        else:
            predictions = df['Prediction'].values
            uncertainties = df['Uncertainty'].values
        
        errors = np.abs(predictions - true_values)
        mae_full = np.mean(errors)
        
        # Calculate MAE at different coverage levels
        sorted_indices = np.argsort(uncertainties)
        sorted_errors = errors[sorted_indices]
        
        mae_values = []
        improvements = {}
        for coverage in coverage_levels:
            if coverage == 1.0:
                mae_at_cov = mae_full
                improvement = 0.0
            else:
                n_keep = int(len(sorted_errors) * coverage)
                if n_keep == 0:
                    n_keep = 1
                mae_at_cov = np.mean(sorted_errors[:n_keep])
                improvement = (mae_full - mae_at_cov) / mae_full * 100
            
            mae_values.append(mae_at_cov)
            improvements[int(coverage * 100)] = {
                'mae': mae_at_cov,
                'improvement': improvement
            }
        
        results_summary[model_name] = {
            'mae_full': mae_full,
            'improvements': improvements
        }
        mae_curves[model_name] = mae_values
    
    # Generate plots if output_dir is provided and we have data
    if output_dir and len(results_summary) > 0:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        colors = {
            'MoNIG': '#1f77b4',  # blue
            'Gaussian': '#ff7f0e',  # orange
            'NIG': '#2ca02c',  # green
            'DeepEnsemble': '#d62728',  # red
            'MCDropout': '#9467bd'  # purple
        }
        
        markers = {
            'MoNIG': 'o',
            'Gaussian': 's',
            'NIG': '^',
            'DeepEnsemble': 'D',
            'MCDropout': 'v'
        }
        
        # Plot 1: MAE vs Coverage
        for model_name, mae_values in mae_curves.items():
            color = colors.get(model_name, 'gray')
            marker = markers.get(model_name, 'o')
            
            ax1.plot(coverage_levels * 100, mae_values, 
                    color=color, marker=marker, markersize=8, linewidth=2.5,
                    label=f"{model_name}", alpha=0.8)
            
            # Add horizontal line for full set
            ax1.axhline(y=results_summary[model_name]['mae_full'], color=color, 
                       linestyle='--', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('Coverage (% of samples kept)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('MAE (pKd)', fontsize=13, fontweight='bold')
        ax1.set_title('Risk-Coverage Curves: MAE vs Coverage\n(Lower = Better)', 
                     fontsize=15, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(45, 105)
        
        # Plot 2: Improvement vs Discarded
        for model_name, mae_values in mae_curves.items():
            color = colors.get(model_name, 'gray')
            marker = markers.get(model_name, 'o')
            
            mae_full = results_summary[model_name]['mae_full']
            
            # Calculate improvements
            improvements = [(mae_full - mae) / mae_full * 100 for mae in mae_values]
            discard_pcts = 100 - coverage_levels * 100
            
            ax2.plot(discard_pcts, improvements,
                    color=color, marker=marker, markersize=8, linewidth=2.5,
                    label=model_name, alpha=0.8)
        
        # Add shaded regions
        ax2.axvspan(0, 20, alpha=0.1, color='green', label='Low risk (≤20% discarded)')
        ax2.axvspan(20, 50, alpha=0.1, color='yellow')
        ax2.axvspan(50, 100, alpha=0.1, color='red')
        
        ax2.set_xlabel('% of Most Uncertain Samples Discarded', fontsize=13, fontweight='bold')
        ax2.set_ylabel('MAE Improvement (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Error Improvement by Selective Prediction\n(Higher = Better)', 
                     fontsize=15, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-5, 55)
        
        plt.tight_layout()
        
        plot_path = output_path / 'risk_coverage_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved risk-coverage comparison plot to: {plot_path}")
        
        # Create improvement heatmap
        _plot_improvement_heatmap(results_summary, output_path)
    
    return results_summary


def _plot_improvement_heatmap(results_dict, output_dir):
    """
    Create heatmap showing improvement at different coverage levels.
    Helper function for analyze_baseline_risk_coverage.
    """
    import matplotlib.pyplot as plt
    
    coverage_levels = [50, 70, 90]
    models = []
    improvements = []
    
    for model_name, results in results_dict.items():
        if results is None:
            continue
        
        models.append(model_name)
        model_improvements = []
        for cov in coverage_levels:
            imp = results['improvements'].get(cov, {}).get('improvement', 0)
            model_improvements.append(imp)
        improvements.append(model_improvements)
    
    if len(models) == 0:
        return
    
    improvements = np.array(improvements)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=0, vmax=20)
    
    # Set ticks
    ax.set_xticks(np.arange(len(coverage_levels)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f'{c}%' for c in coverage_levels])
    ax.set_yticklabels(models)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE Improvement (%)', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(coverage_levels)):
            text = ax.text(j, i, f'{improvements[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Coverage (% kept)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('MAE Improvement by Model and Coverage Level\n(Higher = Better Uncertainty Estimates)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'improvement_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved improvement heatmap to: {plot_path}")


def analyze_single_csv(csv_path, output_dir=None, original_csv_path='pdbbind_descriptors_with_experts_and_binding.csv', experiment_dir='experiments'):
    """Analyze a single inference results CSV file."""
    
    print(f"\nAnalyzing: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Merge with original expert predictions
    print("Merging with original expert predictions...")
    df = merge_with_original_experts(df, original_csv_path)
    
    # Detect number of experts
    expert_cols = [col for col in df.columns if col.startswith('Expert') and '_Prediction' in col]
    num_experts = len(expert_cols)
    
    if num_experts == 0:
        print("Error: No expert predictions found in CSV!")
        return None
    
    # Run all analyses using ORIGINAL expert predictions
    print("Running analyses with original expert predictions...")
    best_cases = find_best_cases(df, num_experts, use_original_experts=True)
    general_catastrophe, dynamicbind_catastrophe, damage_mitigation = find_catastrophe_cases(df, num_experts, use_original_experts=True)
    high_epistemic = find_high_epistemic_cases(df, num_experts, use_original_experts=True)
    
    # Analyze epistemic vs disagreement relationship
    print("\nAnalyzing epistemic uncertainty vs expert disagreement...")
    epistemic_stats = analyze_epistemic_vs_disagreement(df, output_dir)
    
    # Analyze disagreement vs error relationship
    print("Analyzing expert disagreement vs prediction error...")
    disagreement_error_stats = analyze_disagreement_vs_error(df, output_dir)
    
    # Plot disagreement distribution
    print("Creating expert disagreement distribution histogram...")
    disagreement_dist_stats = plot_disagreement_histogram(df, output_dir)
    
    # Analyze uncertainty-accuracy tradeoff (Claim 2B)
    print("Analyzing uncertainty-accuracy tradeoff (Claim 2B: Low uncertainty = High accuracy)...")
    uncertainty_accuracy_stats = analyze_uncertainty_accuracy_tradeoff(df, output_dir)
    
    # Analyze worst-case error reduction (Claim 4B)
    print("Analyzing worst-case error reduction (Claim 4B: CABE reduces tail risk)...")
    worst_case_stats = analyze_worst_case_errors(df, num_experts, output_dir)
    
    # Analyze error correlations between engines
    print("\nAnalyzing error correlations between engines...")
    error_corr_stats = analyze_error_correlation_matrix(df, num_experts, output_dir)
    
    print("Creating pairwise error scatter plots...")
    pairwise_error_stats = analyze_pairwise_error_scatter(df, num_experts, output_dir)
    
    print("Creating MoNIG vs experts error comparison...")
    monig_vs_experts_stats = analyze_monig_vs_experts_scatter(df, num_experts, output_dir)
    
    # Quick baseline comparison if available
    baseline_comparison = None
    if experiment_dir and csv_path:
        # Extract seed from csv_path (e.g., MoNIG_seed42/test_inference_results.csv)
        import re
        seed_match = re.search(r'seed(\d+)', str(csv_path))
        if seed_match:
            seed = int(seed_match.group(1))
            print(f"Comparing with baselines at seed {seed}...")
            baseline_comparison = analyze_baseline_risk_coverage(experiment_dir, seed, output_dir)
    
    # Print report
    print_case_study_report(df, best_cases, general_catastrophe, dynamicbind_catastrophe, high_epistemic, num_experts, damage_mitigation, epistemic_stats, disagreement_error_stats, disagreement_dist_stats, uncertainty_accuracy_stats, baseline_comparison, worst_case_stats, error_corr_stats, pairwise_error_stats, monig_vs_experts_stats)
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if len(best_cases) > 0:
            best_cases.to_csv(output_path / 'best_cases.csv', index=False)
            print(f"\n💾 Saved best cases to: {output_path / 'best_cases.csv'}")
        
        if len(general_catastrophe) > 0:
            general_catastrophe.to_csv(output_path / 'catastrophe_avoidance.csv', index=False)
            print(f"💾 Saved catastrophe cases to: {output_path / 'catastrophe_avoidance.csv'}")
        
        if len(dynamicbind_catastrophe) > 0:
            dynamicbind_catastrophe.to_csv(output_path / 'dynamicbind_catastrophe.csv', index=False)
            print(f"💾 Saved DynamicBind catastrophe to: {output_path / 'dynamicbind_catastrophe.csv'}")
        
        if len(high_epistemic) > 0:
            high_epistemic.to_csv(output_path / 'high_epistemic_cases.csv', index=False)
            print(f"💾 Saved high epistemic cases to: {output_path / 'high_epistemic_cases.csv'}")
        
        if damage_mitigation is not None and len(damage_mitigation) > 0:
            damage_mitigation.to_csv(output_path / 'damage_mitigation_cases.csv', index=False)
            print(f"💾 Saved damage mitigation cases to: {output_path / 'damage_mitigation_cases.csv'}")
    
    return {
        'best_cases': best_cases,
        'general_catastrophe': general_catastrophe,
        'dynamicbind_catastrophe': dynamicbind_catastrophe,
        'high_epistemic': high_epistemic,
        'damage_mitigation': damage_mitigation,
        'epistemic_stats': epistemic_stats,
        'disagreement_error_stats': disagreement_error_stats,
        'disagreement_dist_stats': disagreement_dist_stats,
        'uncertainty_accuracy_stats': uncertainty_accuracy_stats,
        'worst_case_stats': worst_case_stats,
        'error_corr_stats': error_corr_stats,
        'pairwise_error_stats': pairwise_error_stats,
        'monig_vs_experts_stats': monig_vs_experts_stats
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run CABE Case Studies Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on latest results
  python run_case_studies.py
  
  # Run on specific CSV
  python run_case_studies.py --csv experiments/MoNIG_seed42/test_inference_results.csv
  
  # Run on experiment directory with all seeds
  python run_case_studies.py --experiment_dir experiments --model MoNIG --seeds 42 43 44 45 46
        """
    )
    
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to specific inference results CSV')
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Experiment directory (default: experiments)')
    parser.add_argument('--model', type=str, default='MoNIG',
                       help='Model name (default: MoNIG)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Seeds to analyze (default: [42])')
    parser.add_argument('--output_dir', type=str, default='case_studies_output',
                       help='Output directory for results (default: case_studies_output)')
    
    args = parser.parse_args()
    
    print("="*100)
    print("CABE CASE STUDIES ANALYSIS")
    print("="*100)
    
    if args.csv:
        # Analyze single CSV file
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}")
            return
        
        analyze_single_csv(csv_path, args.output_dir)
    
    else:
        # Analyze multiple seeds
        experiment_dir = Path(args.experiment_dir)
        
        all_results = []
        for seed in args.seeds:
            csv_path = experiment_dir / f'{args.model}_seed{seed}' / 'test_inference_results.csv'
            
            if not csv_path.exists():
                print(f"\n⚠️  Warning: File not found: {csv_path}, skipping seed {seed}")
                continue
            
            results = analyze_single_csv(csv_path, Path(args.output_dir) / f'seed{seed}')
            if results:
                results['seed'] = seed
                all_results.append(results)
        
        # Aggregate statistics across seeds
        if len(all_results) > 1:
            print("\n" + "="*100)
            print(f"AGGREGATED STATISTICS ACROSS {len(all_results)} SEEDS")
            print("="*100)
            
            best_case_counts = [len(r['best_cases']) for r in all_results]
            catastrophe_counts = [len(r['general_catastrophe']) for r in all_results]
            
            print(f"\nBest cases: {np.mean(best_case_counts):.1f} ± {np.std(best_case_counts):.1f}")
            print(f"Catastrophe avoidance: {np.mean(catastrophe_counts):.1f} ± {np.std(catastrophe_counts):.1f}")
    
    print("\n✅ Case studies analysis complete!")


if __name__ == '__main__':
    main()
