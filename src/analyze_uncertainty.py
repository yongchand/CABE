"""
Comprehensive Uncertainty Analysis for MoNIG Inference Results
Uses uncertainty_toolbox for rigorous uncertainty quantification metrics
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Import uncertainty_toolbox
import uncertainty_toolbox as uct


def compute_interval_ece(y_pred, y_std, y_true):
    """Expected calibration error from interval coverage."""
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
        y_pred, y_std, y_true, prop_type='interval'
    )
    return float(np.mean(np.abs(exp_props - obs_props)))


def compute_picp(y_pred, y_std, y_true, coverage=0.95):
    """Prediction interval coverage probability for symmetric intervals."""
    z = norm.ppf(0.5 + coverage / 2.0)
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))

def analyze_uncertainty(csv_path):
    """
    Comprehensive uncertainty analysis using uncertainty_toolbox
    """
    df = pd.read_csv(csv_path)
    
    # Calculate prediction errors
    df['MoNIG_Error'] = np.abs(df['MoNIG_Prediction'] - df['True_Affinity'])
    
    # Detect number of experts from columns
    expert_cols = [col for col in df.columns if col.startswith('Expert') and '_Prediction' in col]
    num_experts = len(expert_cols)
    
    # Calculate errors and uncertainties for all experts dynamically
    for j in range(num_experts):
        expert_num = j + 1
        df[f'Expert{expert_num}_Error'] = np.abs(df[f'Expert{expert_num}_Prediction'] - df['True_Affinity'])
        df[f'Expert{expert_num}_Total_Uncertainty'] = df[f'Expert{expert_num}_Epistemic'] + df[f'Expert{expert_num}_Aleatoric']
        df[f'Expert{expert_num}_Std'] = np.sqrt(df[f'Expert{expert_num}_Total_Uncertainty'])
    
    # Total uncertainty (epistemic + aleatoric) = predicted standard deviation
    df['MoNIG_Total_Uncertainty'] = df['MoNIG_Epistemic'] + df['MoNIG_Aleatoric']
    
    # Predicted std (square root of total uncertainty for uncertainty_toolbox)
    df['MoNIG_Std'] = np.sqrt(df['MoNIG_Total_Uncertainty'])
    
    print("="*80)
    print("UNCERTAINTY ANALYSIS REPORT (using uncertainty_toolbox)")
    print("="*80)
    
    # ===== 1. Uncertainty Toolbox Metrics - MoNIG =====
    print("\n📊 UNCERTAINTY_TOOLBOX METRICS - MoNIG")
    print("-" * 80)
    
    y_pred = df['MoNIG_Prediction'].values
    y_true = df['True_Affinity'].values
    y_std = df['MoNIG_Std'].values
    
    # Get all metrics at once
    metrics = uct.metrics.get_all_metrics(y_pred, y_std, y_true, verbose=False)
    
    # Extract nested values
    accuracy_metrics = metrics.get('accuracy', {})
    calibration_metrics = metrics.get('avg_calibration', {})
    sharpness_metrics = metrics.get('sharpness', {})
    scoring_metrics = metrics.get('scoring_rule', {})
    
    print("\n🎯 Prediction Quality:")
    print(f"  MAE:  {accuracy_metrics.get('mae', accuracy_metrics.get('ma', 'N/A'))}")
    print(f"  RMSE: {accuracy_metrics.get('rmse', 'N/A')}")
    print(f"  MDAE: {accuracy_metrics.get('mdae', 'N/A')} (Median Absolute Error)")
    
    print("\n📐 Calibration Metrics:")
    print(f"  Mean Absolute Calibration:  {calibration_metrics.get('ma_cal', 'N/A'):.4f}")
    print(f"  RMS Calibration Error:      {calibration_metrics.get('rms_cal', 'N/A'):.4f}")
    print(f"  Miscalibration Area:        {calibration_metrics.get('miscal_area', 'N/A'):.4f}")
    
    print("\n🎲 Sharpness (how confident is the model?):")
    print(f"  Mean Predicted Std: {sharpness_metrics.get('sharp', 'N/A'):.4f}")
    
    print("\n📈 Scoring Rules (lower is better):")
    print(f"  Negative Log-Likelihood: {scoring_metrics.get('nll', 'N/A')}")
    print(f"  CRPS: {scoring_metrics.get('crps', 'N/A')}")
    print(f"  Check Score: {scoring_metrics.get('check', 'N/A')}")
    print(f"  Interval Score: {scoring_metrics.get('interval', 'N/A')}")
    
    # Additional calibration diagnostics
    ece = compute_interval_ece(y_pred, y_std, y_true)
    picp_90 = compute_picp(y_pred, y_std, y_true, coverage=0.90)
    picp_95 = compute_picp(y_pred, y_std, y_true, coverage=0.95)
    print("\n📊 Additional Calibration Diagnostics:")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  PICP@90%: {picp_90:.4f} (target 0.9000)")
    print(f"  PICP@95%: {picp_95:.4f} (target 0.9500)")
    
    # ===== 2. Expert Comparison with uncertainty_toolbox =====
    print("\n\n👥 EXPERT-WISE CALIBRATION METRICS")
    print("-" * 80)
    
    expert_names = ['GNINA', 'BIND', 'flowdock']
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        pred_col = f'Expert{expert_num}_Prediction'
        std_col = f'Expert{expert_num}_Std'
        
        y_pred_exp = df[pred_col].values
        y_std_exp = df[std_col].values
        metrics_exp = uct.metrics.get_all_metrics(y_pred_exp, y_std_exp, y_true, verbose=False)
        
        exp_accuracy = metrics_exp.get('accuracy', {})
        exp_calibration = metrics_exp.get('avg_calibration', {})
        exp_sharpness = metrics_exp.get('sharpness', {})
        exp_scoring = metrics_exp.get('scoring_rule', {})
        
        print(f"\nExpert {expert_num} ({expert_name}):")
        print(f"  MAE:             {exp_accuracy.get('mae', exp_accuracy.get('ma', 'N/A'))}")
        print(f"  Avg Calibration: {exp_calibration.get('ma_cal', 'N/A'):.4f}")
        print(f"  Sharpness:       {exp_sharpness.get('sharp', 'N/A'):.4f}")
        print(f"  NLL:             {exp_scoring.get('nll', 'N/A')}")
    
    # ===== 3. Epistemic vs Aleatoric Decomposition =====
    print("\n\n🔬 EPISTEMIC vs ALEATORIC DECOMPOSITION")
    print("-" * 80)
    
    print(f"\n{'Metric':<30} {'Mean':<12} {'Median':<12} {'Std':<12}")
    print("-" * 80)
    
    metrics_list = [
        ('MoNIG_Epistemic', 'MoNIG Epistemic'),
        ('MoNIG_Aleatoric', 'MoNIG Aleatoric'),
        ('MoNIG_Total_Uncertainty', 'MoNIG Total'),
    ]
    
    # Add expert-specific metrics dynamically
    expert_names = ['GNINA', 'BIND', 'flowdock']
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        metrics_list.extend([
            (f'Expert{expert_num}_Epistemic', f'Expert{expert_num} ({expert_name}) Epistemic'),
            (f'Expert{expert_num}_Confidence_nu', f'Expert{expert_num} ({expert_name}) Confidence (ν)'),
        ])
    
    for col, label in metrics_list:
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        print(f"{label:<30} {mean:<12.4f} {median:<12.4f} {std:<12.4f}")
    
    epistemic_ratio = df['MoNIG_Epistemic'] / (df['MoNIG_Epistemic'] + df['MoNIG_Aleatoric'])
    aleatoric_ratio = df['MoNIG_Aleatoric'] / (df['MoNIG_Epistemic'] + df['MoNIG_Aleatoric'])
    
    print(f"\nAverage epistemic ratio: {epistemic_ratio.mean():.2%}")
    print(f"Average aleatoric ratio: {aleatoric_ratio.mean():.2%}")
    
    epistemic_dominant = (epistemic_ratio > 0.5).sum()
    aleatoric_dominant = (epistemic_ratio <= 0.5).sum()
    
    print(f"\nEpistemic-dominant cases: {epistemic_dominant} ({epistemic_dominant/len(df)*100:.1f}%)")
    print(f"Aleatoric-dominant cases: {aleatoric_dominant} ({aleatoric_dominant/len(df)*100:.1f}%)")
    
    # ===== 4. High vs Low Uncertainty Predictions =====
    print("\n\n📍 HIGH vs LOW UNCERTAINTY CASES")
    print("-" * 80)
    
    # Split by uncertainty quartiles
    q25 = df['MoNIG_Total_Uncertainty'].quantile(0.25)
    q75 = df['MoNIG_Total_Uncertainty'].quantile(0.75)
    
    low_unc = df[df['MoNIG_Total_Uncertainty'] <= q25]
    high_unc = df[df['MoNIG_Total_Uncertainty'] >= q75]
    
    print(f"Low uncertainty (Q1, n={len(low_unc)}):")
    print(f"  Avg Total Uncertainty: {low_unc['MoNIG_Total_Uncertainty'].mean():.4f}")
    print(f"  Avg Error:             {low_unc['MoNIG_Error'].mean():.4f}")
    print(f"  Avg Epistemic:         {low_unc['MoNIG_Epistemic'].mean():.4f}")
    print(f"  Avg Aleatoric:         {low_unc['MoNIG_Aleatoric'].mean():.4f}")
    
    print(f"\nHigh uncertainty (Q4, n={len(high_unc)}):")
    print(f"  Avg Total Uncertainty: {high_unc['MoNIG_Total_Uncertainty'].mean():.4f}")
    print(f"  Avg Error:             {high_unc['MoNIG_Error'].mean():.4f}")
    print(f"  Avg Epistemic:         {high_unc['MoNIG_Epistemic'].mean():.4f}")
    print(f"  Avg Aleatoric:         {high_unc['MoNIG_Aleatoric'].mean():.4f}")
    
    error_increase = (high_unc['MoNIG_Error'].mean() / low_unc['MoNIG_Error'].mean() - 1) * 100
    print(f"\n  ⚠️  Error increase from low→high uncertainty: {error_increase:+.1f}%")
    
    # ===== 5. Most/Least Certain Predictions =====
    print("\n\n🎯 MOST CERTAIN PREDICTIONS (Top 5)")
    print("-" * 80)
    most_certain = df.nsmallest(5, 'MoNIG_Total_Uncertainty')
    print(most_certain[['ComplexID', 'True_Affinity', 'MoNIG_Prediction', 'MoNIG_Error',
                        'MoNIG_Epistemic', 'MoNIG_Aleatoric', 'MoNIG_Total_Uncertainty']].to_string(index=False))
    
    print("\n\n⚠️  MOST UNCERTAIN PREDICTIONS (Top 5)")
    print("-" * 80)
    most_uncertain = df.nlargest(5, 'MoNIG_Total_Uncertainty')
    print(most_uncertain[['ComplexID', 'True_Affinity', 'MoNIG_Prediction', 'MoNIG_Error',
                          'MoNIG_Epistemic', 'MoNIG_Aleatoric', 'MoNIG_Total_Uncertainty']].to_string(index=False))
    
    # ===== 6. Expert Confidence Analysis =====
    print("\n\n👥 EXPERT CONFIDENCE ANALYSIS")
    print("-" * 80)
    
    # When experts agree vs disagree (use existing column if available, otherwise compute)
    if 'Expert_Disagreement' not in df.columns:
        # Compute max pairwise disagreement
        predictions = [df[f'Expert{j+1}_Prediction'].values for j in range(num_experts)]
        max_disagreements = []
        for i in range(len(df)):
            max_disag = max(abs(predictions[j][i] - predictions[k][i]) 
                          for j in range(num_experts) 
                          for k in range(j+1, num_experts)) if num_experts >= 2 else 0.0
            max_disagreements.append(max_disag)
        df['Expert_Disagreement'] = max_disagreements
    
    high_disagreement = df[df['Expert_Disagreement'] > df['Expert_Disagreement'].median()]
    low_disagreement = df[df['Expert_Disagreement'] <= df['Expert_Disagreement'].median()]
    
    print(f"When experts AGREE (low disagreement, n={len(low_disagreement)}):")
    print(f"  Avg MoNIG Epistemic:  {low_disagreement['MoNIG_Epistemic'].mean():.4f}")
    print(f"  Avg MoNIG Error:      {low_disagreement['MoNIG_Error'].mean():.4f}")
    
    print(f"\nWhen experts DISAGREE (high disagreement, n={len(high_disagreement)}):")
    print(f"  Avg MoNIG Epistemic:  {high_disagreement['MoNIG_Epistemic'].mean():.4f}")
    print(f"  Avg MoNIG Error:      {high_disagreement['MoNIG_Error'].mean():.4f}")
    
    # ===== 7. Practical Recommendations =====
    print("\n\n💡 PRACTICAL RECOMMENDATIONS")
    print("-" * 80)
    
    # Define thresholds
    high_epistemic_thresh = df['MoNIG_Epistemic'].quantile(0.75)
    high_aleatoric_thresh = df['MoNIG_Aleatoric'].quantile(0.75)
    
    high_epistemic = df[df['MoNIG_Epistemic'] > high_epistemic_thresh]
    high_aleatoric = df[df['MoNIG_Aleatoric'] > high_aleatoric_thresh]
    
    print(f"High Epistemic Uncertainty cases (n={len(high_epistemic)}, >{high_epistemic_thresh:.4f}):")
    print(f"  → Model is uncertain. Consider:")
    print(f"    • Collecting more training data similar to these cases")
    print(f"    • Active learning: prioritize these for expert annotation")
    print(f"    • Model may need more capacity or better features")
    
    print(f"\nHigh Aleatoric Uncertainty cases (n={len(high_aleatoric)}, >{high_aleatoric_thresh:.4f}):")
    print(f"  → Inherent data noise. Consider:")
    print(f"    • These predictions have fundamental limits")
    print(f"    • May need better experimental measurements")
    print(f"    • Ensemble or repeated measurements may help")
    
    # Calibration quality assessment
    print(f"\n🎯 Calibration Assessment:")
    avg_cal = calibration_metrics.get('ma_cal', None)
    if avg_cal is not None:
        if avg_cal < 0.1:
            print(f"  ✅ Excellent calibration (ma_cal={avg_cal:.4f})")
        elif avg_cal < 0.3:
            print(f"  ⚠️  Moderate calibration (ma_cal={avg_cal:.4f})")
        else:
            print(f"  ❌ Poor calibration (ma_cal={avg_cal:.4f}) - uncertainties need adjustment")
    else:
        print(f"  ⚠️  Calibration metric not available")
    
    print("\n" + "="*80)
    
    return df, metrics


def visualize_uncertainty(df, output_prefix='uncertainty'):
    """
    Create comprehensive uncertainty visualizations using uncertainty_toolbox
    """
    # Detect number of experts from columns
    expert_cols = [col for col in df.columns if col.startswith('Expert') and '_Prediction' in col]
    num_experts = len(expert_cols)
    
    y_pred = df['MoNIG_Prediction'].values
    y_true = df['True_Affinity'].values
    y_std = df['MoNIG_Std'].values
    
    # Define colors for experts
    colors = ['coral', 'skyblue', 'lightgreen', 'orange', 'purple']
    
    base_path = Path(output_prefix)
    base_dir = base_path.parent if str(base_path.parent) != '.' else Path('.')
    stem = base_path.name
    
    # Create single subdirectory
    output_dir = base_dir / f"{stem}_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. Use uncertainty_toolbox's main plot =====
    print("\n📊 Generating uncertainty_toolbox visualizations...")
    
    try:
        _ = uct.viz.plot_calibration(y_pred, y_std, y_true)
        cal_path = output_dir / f'{stem}_uct_calibration.png'
        plt.savefig(cal_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {cal_path}")
        plt.close()
    except Exception as e:
        print(f"  ⚠ Skipped calibration plot: {e}")
    
    try:
        _ = uct.viz.plot_intervals(y_pred, y_std, y_true, n_subset=min(200, len(y_pred)))
        intervals_path = output_dir / f'{stem}_uct_intervals.png'
        plt.savefig(intervals_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {intervals_path}")
        plt.close()
    except Exception as e:
        print(f"  ⚠ Skipped intervals plot: {e}")
    
    try:
        _ = uct.viz.plot_intervals_ordered(y_pred, y_std, y_true, n_subset=min(200, len(y_pred)))
        ordered_path = output_dir / f'{stem}_uct_intervals_ordered.png'
        plt.savefig(ordered_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {ordered_path}")
        plt.close()
    except Exception as e:
        print(f"  ⚠ Skipped ordered intervals plot: {e}")
    
    try:
        x_vals = np.arange(len(y_true))
        _ = uct.viz.plot_xy(
            y_pred,
            y_std,
            y_true,
            x_vals,
            num_stds_confidence_bound=2
        )
        conf_path = output_dir / f'{stem}_uct_confidence_band.png'
        plt.savefig(conf_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {conf_path}")
        plt.close()
    except Exception as e:
        print(f"  ⚠ Skipped confidence band plot: {e}")
    
    # ===== 2. Custom plots for epistemic/aleatoric and expert analysis =====
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Epistemic vs Aleatoric
    ax1 = plt.subplot(3, 4, 1)
    scatter = ax1.scatter(df['MoNIG_Epistemic'], df['MoNIG_Aleatoric'], 
                         c=df['MoNIG_Error'], alpha=0.6, cmap='viridis', s=30)
    ax1.set_xlabel('Epistemic Uncertainty')
    ax1.set_ylabel('Aleatoric Uncertainty')
    ax1.set_title('Epistemic vs Aleatoric (colored by error)')
    plt.colorbar(scatter, ax=ax1, label='Prediction Error')
    ax1.plot([0, max(df['MoNIG_Epistemic'].max(), df['MoNIG_Aleatoric'].max())],
             [0, max(df['MoNIG_Epistemic'].max(), df['MoNIG_Aleatoric'].max())],
             'r--', alpha=0.3, label='Equal')
    ax1.legend()
    
    # Epistemic/Aleatoric Ratio Distribution
    ax2 = plt.subplot(3, 4, 2)
    epistemic_ratio = df['MoNIG_Epistemic'] / (df['MoNIG_Epistemic'] + df['MoNIG_Aleatoric'])
    ax2.hist(epistemic_ratio, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Equal split')
    ax2.set_xlabel('Epistemic / (Epistemic + Aleatoric)')
    ax2.set_ylabel('Count')
    ax2.set_title('Uncertainty Type Distribution')
    ax2.legend()
    
    # Uncertainty by Error Quantile
    ax3 = plt.subplot(3, 4, 3)
    df['Error_Quantile'] = pd.qcut(df['MoNIG_Error'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    error_groups = df.groupby('Error_Quantile', observed=True)[['MoNIG_Epistemic', 'MoNIG_Aleatoric']].mean()
    error_groups.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
    ax3.set_xlabel('Error Quantile')
    ax3.set_ylabel('Average Uncertainty')
    ax3.set_title('Uncertainty by Error Quantile')
    ax3.legend(['Epistemic', 'Aleatoric'])
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Per-Expert Epistemic Uncertainty
    ax4 = plt.subplot(3, 4, 4)
    expert_epistemic_data = [df[f'Expert{j+1}_Epistemic'].values for j in range(num_experts)]
    expert_epistemic_data.append(df['MoNIG_Epistemic'].values)
    positions = list(range(1, num_experts + 2))
    expert_labels = [f'Expert {j+1}' for j in range(num_experts)] + ['MoNIG']
    _ = ax4.boxplot(expert_epistemic_data, 
                     positions=positions, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax4.set_xticklabels(expert_labels)
    ax4.set_ylabel('Epistemic Uncertainty')
    ax4.set_title('Epistemic Uncertainty Comparison')
    ax4.grid(axis='y', alpha=0.3)
    
    # Row 2: Expert Analysis
    # Expert Confidence Distribution
    ax5 = plt.subplot(3, 4, 5)
    colors = ['coral', 'skyblue', 'lightgreen', 'orange', 'purple']
    for j in range(num_experts):
        expert_num = j + 1
        ax5.hist(df[f'Expert{expert_num}_Confidence_nu'], bins=50, alpha=0.6, 
                label=f'Expert {expert_num}', density=True, color=colors[j % len(colors)])
    ax5.set_xlabel('Confidence (ν)')
    ax5.set_ylabel('Density')
    ax5.set_title('Expert Confidence Distribution')
    ax5.legend()
    ax5.set_yscale('log')
    
    # Expert Weights Distribution
    ax6 = plt.subplot(3, 4, 6)
    for j in range(num_experts):
        expert_num = j + 1
        ax6.hist(df[f'Expert{expert_num}_Weight'], bins=50, alpha=0.6, 
                label=f'Expert {expert_num}', color=colors[j % len(colors)])
    ax6.set_xlabel('Expert Weight')
    ax6.set_ylabel('Count')
    ax6.set_title('MoNIG Expert Weight Distribution')
    ax6.legend()
    if num_experts > 1:
        ax6.axvline(1.0/num_experts, color='red', linestyle='--', alpha=0.5, label='Equal weight')
    
    # Expert Disagreement vs MoNIG Epistemic
    ax7 = plt.subplot(3, 4, 7)
    if 'Expert_Disagreement' not in df.columns:
        # Compute if not already present
        predictions = [df[f'Expert{j+1}_Prediction'].values for j in range(num_experts)]
        max_disagreements = []
        for i in range(len(df)):
            max_disag = max(abs(predictions[j][i] - predictions[k][i]) 
                          for j in range(num_experts) 
                          for k in range(j+1, num_experts)) if num_experts >= 2 else 0.0
            max_disagreements.append(max_disag)
        df['Expert_Disagreement'] = max_disagreements
    scatter = ax7.scatter(df['Expert_Disagreement'], df['MoNIG_Epistemic'], 
                         c=df['MoNIG_Error'], alpha=0.5, s=30, cmap='plasma')
    ax7.set_xlabel('Expert Disagreement (max pairwise)')
    ax7.set_ylabel('MoNIG Epistemic Uncertainty')
    ax7.set_title('Expert Disagreement vs Epistemic')
    plt.colorbar(scatter, ax=ax7, label='Error')
    
    # Confidence vs Error by Expert
    ax8 = plt.subplot(3, 4, 8)
    for j in range(num_experts):
        expert_num = j + 1
        ax8.scatter(df[f'Expert{expert_num}_Confidence_nu'], df[f'Expert{expert_num}_Error'], 
                   alpha=0.4, s=20, label=f'Expert {expert_num}', color=colors[j % len(colors)])
    ax8.set_xlabel('Confidence (ν)')
    ax8.set_ylabel('Prediction Error')
    ax8.set_title('Confidence vs Error (by Expert)')
    ax8.legend()
    ax8.set_xscale('log')
    
    # Row 3: More analysis
    # MAE comparison
    ax9 = plt.subplot(3, 4, 9)
    mae_data = [df[f'Expert{j+1}_Error'].mean() for j in range(num_experts)]
    mae_data.append(df['MoNIG_Error'].mean())
    mae_labels = [f'Expert {j+1}' for j in range(num_experts)] + ['MoNIG']
    mae_colors = colors[:num_experts] + ['lightgreen']
    bars = ax9.bar(mae_labels, mae_data, color=mae_colors)
    ax9.set_ylabel('Mean Absolute Error')
    ax9.set_title('Prediction Accuracy Comparison')
    ax9.grid(axis='y', alpha=0.3)
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Prediction scatter plots for experts (show first 2 experts)
    min_val = df['True_Affinity'].min()
    max_val = df['True_Affinity'].max()
    for j in range(min(2, num_experts)):
        expert_num = j + 1
        ax_idx = 10 + j
        if ax_idx <= 11:  # Only plot in ax10 and ax11
            ax = plt.subplot(3, 4, ax_idx)
            ax.scatter(df['True_Affinity'], df[f'Expert{expert_num}_Prediction'], 
                      alpha=0.4, s=20, color=colors[j % len(colors)])
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            ax.set_xlabel('True Affinity')
            ax.set_ylabel(f'Expert {expert_num} Prediction')
            ax.set_title(f'Expert {expert_num}: Predicted vs True')
            ax.grid(alpha=0.3)
    
    # Prediction scatter: MoNIG vs True
    ax12 = plt.subplot(3, 4, 12)
    ax12.scatter(df['True_Affinity'], df['MoNIG_Prediction'], alpha=0.4, s=20, color='lightgreen')
    ax12.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax12.set_xlabel('True Affinity')
    ax12.set_ylabel('MoNIG Prediction')
    ax12.set_title('MoNIG: Predicted vs True')
    ax12.grid(alpha=0.3)
    
    plt.tight_layout()
    custom_path = output_dir / f'{stem}_custom_analysis.png'
    plt.savefig(custom_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved: {custom_path}")
    plt.close()
    
    # ===== 3. Focused expert diagnostics =====
    fig_diag, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Compute bins from all experts
    all_confidences = np.concatenate([df[f'Expert{j+1}_Confidence_nu'].values for j in range(num_experts)])
    bins = np.logspace(np.log10(max(all_confidences.min(), 1e-3)),
                       np.log10(all_confidences.max()), 40)
    expert_names = ['GNINA', 'BIND', 'flowdock']
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        axes[0].hist(df[f'Expert{expert_num}_Confidence_nu'], bins=bins, alpha=0.6, 
                    label=f'Expert {expert_num} ({expert_name})', color=colors[j % len(colors)])
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Confidence (ν)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Expert Confidence Distribution')
    axes[0].legend()
    
    for j in range(num_experts):
        expert_num = j + 1
        axes[1].hist(df[f'Expert{expert_num}_Weight'], bins=40, alpha=0.7, 
                    label=f'Expert {expert_num}', color=colors[j % len(colors)])
    if num_experts > 1:
        axes[1].axvline(1.0/num_experts, color='red', linestyle='--', linewidth=1.5, label='Equal weight')
    axes[1].set_xlabel('Expert Weight')
    axes[1].set_ylabel('Count')
    axes[1].set_title('MoNIG Expert Weight Distribution')
    axes[1].legend()
    
    maes = [df[f'Expert{j+1}_Error'].mean() for j in range(num_experts)]
    maes.append(df['MoNIG_Error'].mean())
    mae_labels = [f'Expert {j+1}' for j in range(num_experts)] + ['MoNIG']
    mae_colors = colors[:num_experts] + ['#81c784']
    axes[2].bar(mae_labels, maes, color=mae_colors)
    axes[2].set_ylabel('Mean Absolute Error')
    axes[2].set_title('Prediction Accuracy Comparison')
    for idx, mae in enumerate(maes):
        axes[2].text(idx, mae + 0.01, f"{mae:.3f}", ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    diag_path = output_dir / f'{stem}_expert_stats.png'
    plt.savefig(diag_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {diag_path}")
    plt.close(fig_diag)
    
    print(f"\n✅ All visualizations saved under: {output_dir.resolve()}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Uncertainty from Inference Results using uncertainty_toolbox')
    parser.add_argument('--csv', type=str, default='test_inference_results.csv',
                       help='Path to inference results CSV (default: test_inference_results.csv from test split)')
    parser.add_argument('--output_prefix', type=str, default='test_uncertainty',
                       help='Prefix for output files (default: test_uncertainty)')
    
    args = parser.parse_args()
    
    # Run analysis
    df, metrics = analyze_uncertainty(args.csv)
    
    # Create visualizations
    visualize_uncertainty(df, args.output_prefix)
    
    base = Path(args.output_prefix)
    stem = base.name
    base_dir = base.parent if str(base.parent) != '.' else Path('.')
    output_dir = base_dir / f"{stem}_figures"
    print("\n" + "="*80)
    print("✅ Uncertainty analysis complete!")
    print("="*80)
    print("\nGenerated directories/files:")
    print(f"  • {output_dir} ->")
    print(f"      - {stem}_uct_calibration.png")
    print(f"      - {stem}_uct_intervals.png")
    print(f"      - {stem}_uct_intervals_ordered.png")
    print(f"      - {stem}_uct_confidence_band.png")
    print(f"      - {stem}_custom_analysis.png")
    print(f"      - {stem}_expert_stats.png")
    print("="*80)

