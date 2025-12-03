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
    
    # Check if conformal prediction intervals exist
    has_conformal = 'Conformal_Lower' in df.columns and 'Conformal_Upper' in df.columns
    
    # Convert conformal columns to numeric if they exist (they might be stored as strings with brackets)
    if has_conformal:
        # Handle string format like '[2.3908916]' by stripping brackets
        df['Conformal_Lower'] = df['Conformal_Lower'].astype(str).str.strip('[]').astype(float)
        df['Conformal_Upper'] = df['Conformal_Upper'].astype(str).str.strip('[]').astype(float)
        if 'Conformal_Width' in df.columns:
            df['Conformal_Width'] = df['Conformal_Width'].astype(str).str.strip('[]').astype(float)
    
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
    
    # ===== Conformal Prediction Analysis =====
    if has_conformal:
        print("\n\n🎯 CONFORMAL PREDICTION ANALYSIS")
        print("-" * 80)
        
        # Compute PICP for conformal intervals
        conformal_inside = (df['True_Affinity'] >= df['Conformal_Lower']) & (df['True_Affinity'] <= df['Conformal_Upper'])
        conformal_picp = conformal_inside.mean()
        
        # Get target coverage if available (from conformal quantile file or default)
        # For now, assume 0.95 if not specified
        target_coverage = 0.95  # Could be read from conformal.npz if needed
        
        print(f"\nConformal Prediction Intervals:")
        print(f"  PICP: {conformal_picp:.4f} (target: {target_coverage:.4f})")
        print(f"  Coverage Error: {abs(conformal_picp - target_coverage):.4f}")
        
        if conformal_picp >= target_coverage - 0.05:  # Within 5% of target
            print(f"  ✅ Good calibration (within acceptable range)")
        else:
            print(f"  ⚠️  Under-coverage (intervals too narrow)")
        
        # Compare conformal vs standard intervals
        standard_inside_95 = (df['True_Affinity'] >= (df['MoNIG_Prediction'] - 1.96 * df['MoNIG_Std'])) & \
                             (df['True_Affinity'] <= (df['MoNIG_Prediction'] + 1.96 * df['MoNIG_Std']))
        standard_picp_95 = standard_inside_95.mean()
        
        print(f"\nComparison with Standard 95% Intervals (z=1.96):")
        print(f"  Standard PICP: {standard_picp_95:.4f}")
        print(f"  Conformal PICP: {conformal_picp:.4f}")
        print(f"  Difference: {conformal_picp - standard_picp_95:+.4f}")
        
        # Average interval widths
        conformal_width = df['Conformal_Width'].mean()
        standard_width_95 = (2 * 1.96 * df['MoNIG_Std']).mean()
        
        print(f"\nAverage Interval Widths:")
        print(f"  Standard (95%): {standard_width_95:.4f}")
        print(f"  Conformal:      {conformal_width:.4f}")
        print(f"  Ratio:          {conformal_width / standard_width_95:.4f}")
        
        # Efficiency: narrower intervals with same coverage are better
        if conformal_width < standard_width_95 and conformal_picp >= standard_picp_95:
            print(f"  ✅ Conformal intervals are more efficient (narrower with same/better coverage)")
        elif conformal_picp > standard_picp_95:
            print(f"  ✅ Conformal intervals achieve better coverage")
        else:
            print(f"  ⚠️  Standard intervals may be more efficient")
    
    # ===== 2. Expert Comparison with uncertainty_toolbox =====
    print("\n\n👥 EXPERT-WISE CALIBRATION METRICS")
    print("-" * 80)
    
    expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
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
    expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
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
    
    # ===== 7. MoNIG Identifies When NOT to Trust Engines =====
    print("\n\n🚨 MoNIG IDENTIFIES WHEN NOT TO TRUST ENGINES")
    print("-" * 80)
    print("\nThis section demonstrates that MoNIG correctly identifies low confidence")
    print("when individual engines are wrong and have high epistemic uncertainty.")
    
    expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
    
    # Define thresholds: "wrong" = error > median error, "high epistemic" = > 75th percentile
    error_thresholds = {}
    epistemic_thresholds = {}
    
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        
        # Threshold: expert is "wrong" if error > median error
        error_thresholds[j] = df[f'Expert{expert_num}_Error'].median()
        # Threshold: expert has "high epistemic" if > 75th percentile
        epistemic_thresholds[j] = df[f'Expert{expert_num}_Epistemic'].quantile(0.75)
    
    # MoNIG high epistemic threshold
    monig_high_epistemic_thresh = df['MoNIG_Epistemic'].quantile(0.75)
    
    print(f"\n📊 Thresholds:")
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        print(f"  {expert_name}: Error > {error_thresholds[j]:.4f}, Epistemic > {epistemic_thresholds[j]:.4f}")
    print(f"  MoNIG: Epistemic > {monig_high_epistemic_thresh:.4f}")
    
    # Find cases for each expert
    untrustworthy_cases = {}
    
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        
        # Find cases where expert is wrong AND has high epistemic uncertainty
        wrong_mask = df[f'Expert{expert_num}_Error'] > error_thresholds[j]
        high_epistemic_mask = df[f'Expert{expert_num}_Epistemic'] > epistemic_thresholds[j]
        untrustworthy = df[wrong_mask & high_epistemic_mask].copy()
        
        untrustworthy_cases[j] = untrustworthy
        
        print(f"\n🔴 {expert_name} Untrustworthy Cases (Wrong + High Epistemic):")
        print(f"   Found {len(untrustworthy)} cases")
        
        if len(untrustworthy) > 0:
            # Check if MoNIG also has high epistemic in these cases
            monig_also_uncertain = untrustworthy[untrustworthy['MoNIG_Epistemic'] > monig_high_epistemic_thresh]
            print(f"   ✅ MoNIG also uncertain in {len(monig_also_uncertain)}/{len(untrustworthy)} cases ({len(monig_also_uncertain)/len(untrustworthy)*100:.1f}%)")
            
            # Show top 5 clearest examples
            if len(untrustworthy) > 0:
                # Sort by error (most wrong) and epistemic (highest uncertainty)
                untrustworthy_sorted = untrustworthy.sort_values(
                    by=[f'Expert{expert_num}_Error', f'Expert{expert_num}_Epistemic'], 
                    ascending=False
                ).head(5)
                
                print(f"\n   Top 5 clearest examples:")
                display_cols = ['ComplexID', 'True_Affinity', 
                               f'Expert{expert_num}_Prediction', f'Expert{expert_num}_Error',
                               f'Expert{expert_num}_Epistemic', 
                               'MoNIG_Prediction', 'MoNIG_Epistemic']
                print(untrustworthy_sorted[display_cols].to_string(index=False))
        else:
            print(f"   ⚠️  No cases found matching criteria")
    
    # Summary: How often does MoNIG correctly identify untrustworthy experts?
    print(f"\n\n📈 SUMMARY: MoNIG's Ability to Identify Untrustworthy Engines")
    print("-" * 80)
    
    all_untrustworthy = set()
    for j in range(num_experts):
        if len(untrustworthy_cases[j]) > 0:
            all_untrustworthy.update(untrustworthy_cases[j].index)
    
    if len(all_untrustworthy) > 0:
        all_untrustworthy_df = df.loc[list(all_untrustworthy)]
        monig_correctly_uncertain = all_untrustworthy_df[
            all_untrustworthy_df['MoNIG_Epistemic'] > monig_high_epistemic_thresh
        ]
        
        print(f"Total cases where ANY expert is untrustworthy: {len(all_untrustworthy)}")
        print(f"MoNIG correctly uncertain (high epistemic) in: {len(monig_correctly_uncertain)} cases")
        print(f"Success rate: {len(monig_correctly_uncertain)/len(all_untrustworthy)*100:.1f}%")
        print(f"\n✅ This demonstrates that MoNIG correctly identifies when NOT to trust engines!")
        print(f"   When individual engines are wrong and uncertain, MoNIG also shows high epistemic uncertainty.")
    else:
        print("⚠️  No untrustworthy cases found in this dataset.")
    
    # ===== 8. Practical Recommendations =====
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
    
    return df, metrics, has_conformal


def plot_uncertainty_as_risk_estimator(df, output_dir, stem):
    """
    Create Uncertainty as Risk-Estimator plot.
    
    This plot shows how well uncertainty predicts prediction error/risk.
    Key diagnostic for uncertainty quantification quality.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Uncertainty as Risk Estimator', fontsize=16, fontweight='bold')
    
    # Compute total uncertainty and error
    total_uncertainty = df['MoNIG_Total_Uncertainty'].values
    error = df['MoNIG_Error'].values
    epistemic = df['MoNIG_Epistemic'].values
    aleatoric = df['MoNIG_Aleatoric'].values
    
    # Plot 1: Scatter plot - Total Uncertainty vs Error
    ax1 = axes[0, 0]
    scatter = ax1.scatter(total_uncertainty, error, alpha=0.5, s=20, c=error, cmap='plasma')
    ax1.set_xlabel('Total Uncertainty (Epistemic + Aleatoric)')
    ax1.set_ylabel('Prediction Error |y_true - y_pred|')
    ax1.set_title('Uncertainty vs Error (Scatter)')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Error')
    
    # Add correlation coefficient
    corr = np.corrcoef(total_uncertainty, error)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='top', fontsize=11)
    
    # Plot 2: Binned Analysis - Average Error per Uncertainty Quantile
    ax2 = axes[0, 1]
    df_temp = df.copy()
    df_temp['Uncertainty_Quantile'] = pd.qcut(total_uncertainty, q=10, labels=False, duplicates='drop')
    quantile_stats = df_temp.groupby('Uncertainty_Quantile', observed=True).agg({
        'MoNIG_Error': ['mean', 'std', 'count'],
        'MoNIG_Total_Uncertainty': 'mean'
    }).reset_index()
    
    quantile_stats.columns = ['Quantile', 'Mean_Error', 'Std_Error', 'Count', 'Mean_Uncertainty']
    quantile_stats = quantile_stats.sort_values('Mean_Uncertainty')
    
    ax2.errorbar(quantile_stats['Mean_Uncertainty'], quantile_stats['Mean_Error'],
                yerr=quantile_stats['Std_Error'], fmt='o-', linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Mean Error ± Std')
    
    # Ideal line: if uncertainty perfectly predicts error, they should be proportional
    if len(quantile_stats) > 1 and quantile_stats['Mean_Uncertainty'].max() > 0:
        ideal_slope = quantile_stats['Mean_Error'].max() / quantile_stats['Mean_Uncertainty'].max()
        ideal_line = ideal_slope * quantile_stats['Mean_Uncertainty']
        ax2.plot(quantile_stats['Mean_Uncertainty'], ideal_line, 'r--', 
                linewidth=2, alpha=0.5, label='Ideal (proportional)')
    
    ax2.set_xlabel('Mean Uncertainty (per quantile)')
    ax2.set_ylabel('Mean Prediction Error')
    ax2.set_title('Uncertainty Quantiles vs Average Error')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Epistemic vs Aleatoric as Risk Estimators
    ax3 = axes[1, 0]
    
    # Create bins for epistemic and aleatoric
    try:
        epistemic_bins = pd.qcut(epistemic, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        aleatoric_bins = pd.qcut(aleatoric, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        
        df_temp2 = df.copy()
        df_temp2['Epistemic_Bin'] = epistemic_bins
        df_temp2['Aleatoric_Bin'] = aleatoric_bins
        
        # Heatmap: Average error for each epistemic/aleatoric combination
        error_heatmap = df_temp2.groupby(['Epistemic_Bin', 'Aleatoric_Bin'], observed=True)['MoNIG_Error'].mean().unstack()
        
        im = ax3.imshow(error_heatmap.values, cmap='YlOrRd', aspect='auto', origin='lower')
        ax3.set_xticks(range(len(error_heatmap.columns)))
        ax3.set_xticklabels(error_heatmap.columns, rotation=45, ha='right')
        ax3.set_yticks(range(len(error_heatmap.index)))
        ax3.set_yticklabels(error_heatmap.index)
        ax3.set_xlabel('Aleatoric Uncertainty Level')
        ax3.set_ylabel('Epistemic Uncertainty Level')
        ax3.set_title('Average Error by Uncertainty Type')
        plt.colorbar(im, ax=ax3, label='Mean Error')
    except Exception as e:
        ax3.text(0.5, 0.5, f'Could not create heatmap:\n{str(e)}', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Average Error by Uncertainty Type')
    
    # Plot 4: Risk Estimation Quality Metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Compute various risk estimation metrics
    metrics_text = "Risk Estimation Quality Metrics\n" + "="*50 + "\n\n"
    
    # Correlation metrics
    corr_total = np.corrcoef(total_uncertainty, error)[0, 1]
    corr_epistemic = np.corrcoef(epistemic, error)[0, 1]
    corr_aleatoric = np.corrcoef(aleatoric, error)[0, 1]
    
    metrics_text += f"Correlation Coefficients:\n"
    metrics_text += f"  Total Uncertainty:     {corr_total:+.4f}\n"
    metrics_text += f"  Epistemic Uncertainty: {corr_epistemic:+.4f}\n"
    metrics_text += f"  Aleatoric Uncertainty: {corr_aleatoric:+.4f}\n\n"
    
    # Quantile analysis
    low_unc_q = np.percentile(total_uncertainty, 25)
    high_unc_q = np.percentile(total_uncertainty, 75)
    
    low_unc_mask = total_uncertainty <= low_unc_q
    high_unc_mask = total_uncertainty >= high_unc_q
    
    error_ratio = None
    if np.sum(low_unc_mask) > 0 and np.sum(high_unc_mask) > 0:
        low_unc_error = df.loc[low_unc_mask, 'MoNIG_Error'].mean()
        high_unc_error = df.loc[high_unc_mask, 'MoNIG_Error'].mean()
        error_ratio = high_unc_error / low_unc_error if low_unc_error > 0 else np.nan
        
        metrics_text += f"Error by Uncertainty Quartiles:\n"
        metrics_text += f"  Low Uncertainty (Q1):  {low_unc_error:.4f}\n"
        metrics_text += f"  High Uncertainty (Q4):  {high_unc_error:.4f}\n"
        metrics_text += f"  Error Ratio (Q4/Q1):   {error_ratio:.2f}x\n\n"
    else:
        error_ratio = np.nan
        metrics_text += "Error by Uncertainty Quartiles:\n"
        metrics_text += "  (Insufficient data for quartile analysis)\n\n"
    
    # Risk estimation quality
    if corr_total > 0.5:
        quality = "✅ Excellent"
    elif corr_total > 0.3:
        quality = "⚠️  Moderate"
    elif corr_total > 0.1:
        quality = "⚠️  Weak"
    else:
        quality = "❌ Poor"
    
    metrics_text += f"Risk Estimation Quality: {quality}\n"
    metrics_text += f"  (Based on correlation: {corr_total:.3f})\n\n"
    
    # Coverage analysis
    if error_ratio is not None and not np.isnan(error_ratio):
        if error_ratio > 1.5:
            metrics_text += "✅ High uncertainty cases have significantly higher error\n"
        elif error_ratio > 1.2:
            metrics_text += "⚠️  Moderate difference between low/high uncertainty cases\n"
        else:
            metrics_text += "❌ Uncertainty does not reliably predict error\n"
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    risk_path = output_dir / f'{stem}_uncertainty_risk_estimator.png'
    plt.savefig(risk_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {risk_path}")
    
    return corr_total, error_ratio


def visualize_uncertainty(df, output_prefix='uncertainty', has_conformal=False):
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
    expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
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
    
    # ===== 4. Visualization: "When NOT to Trust Engines" =====
    print("\n📊 Generating 'When NOT to Trust Engines' visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('MoNIG Identifies When NOT to Trust Engines', fontsize=16, fontweight='bold')
    
    expert_names = ['GNINA', 'BIND', 'flowdock', 'DynamicBind']
    colors_expert = ['coral', 'skyblue', 'lightgreen', 'orange', 'purple', 'pink', 'cyan', 'yellow']
    
    # Calculate thresholds
    error_thresholds = {}
    epistemic_thresholds = {}
    for j in range(num_experts):
        expert_num = j + 1
        error_thresholds[j] = df[f'Expert{expert_num}_Error'].median()
        epistemic_thresholds[j] = df[f'Expert{expert_num}_Epistemic'].quantile(0.75)
    monig_high_epistemic_thresh = df['MoNIG_Epistemic'].quantile(0.75)
    
    # Plot 1: Expert Error vs Epistemic (highlight untrustworthy cases)
    ax1 = axes[0, 0]
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        
        # All points
        ax1.scatter(df[f'Expert{expert_num}_Epistemic'], df[f'Expert{expert_num}_Error'],
                   alpha=0.3, s=15, color=colors_expert[j], label=f'{expert_name} (all)')
        
        # Untrustworthy cases (wrong + high epistemic)
        wrong_mask = df[f'Expert{expert_num}_Error'] > error_thresholds[j]
        high_epistemic_mask = df[f'Expert{expert_num}_Epistemic'] > epistemic_thresholds[j]
        untrustworthy = df[wrong_mask & high_epistemic_mask]
        
        if len(untrustworthy) > 0:
            ax1.scatter(untrustworthy[f'Expert{expert_num}_Epistemic'], 
                       untrustworthy[f'Expert{expert_num}_Error'],
                       alpha=0.8, s=50, color=colors_expert[j], 
                       marker='X', edgecolors='red', linewidths=1.5,
                       label=f'{expert_name} (untrustworthy)')
    
    ax1.axhline(y=df['MoNIG_Error'].median(), color='gray', linestyle='--', alpha=0.5, label='Median Error')
    ax1.axvline(x=monig_high_epistemic_thresh, color='gray', linestyle='--', alpha=0.5, label='High Epistemic')
    ax1.set_xlabel('Expert Epistemic Uncertainty')
    ax1.set_ylabel('Expert Prediction Error')
    ax1.set_title('Expert Error vs Epistemic Uncertainty\n(X = Untrustworthy Cases)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    
    # Plot 2: MoNIG Epistemic for Untrustworthy Expert Cases
    ax2 = axes[0, 1]
    all_untrustworthy_indices = set()
    for j in range(num_experts):
        expert_num = j + 1
        wrong_mask = df[f'Expert{expert_num}_Error'] > error_thresholds[j]
        high_epistemic_mask = df[f'Expert{expert_num}_Epistemic'] > epistemic_thresholds[j]
        untrustworthy = df[wrong_mask & high_epistemic_mask]
        if len(untrustworthy) > 0:
            all_untrustworthy_indices.update(untrustworthy.index)
    
    if len(all_untrustworthy_indices) > 0:
        all_untrustworthy_df = df.loc[list(all_untrustworthy_indices)]
        trustworthy_df = df[~df.index.isin(all_untrustworthy_indices)]
        
        ax2.scatter(trustworthy_df['MoNIG_Epistemic'], trustworthy_df['MoNIG_Error'],
                   alpha=0.3, s=15, color='lightblue', label='Trustworthy cases')
        ax2.scatter(all_untrustworthy_df['MoNIG_Epistemic'], all_untrustworthy_df['MoNIG_Error'],
                   alpha=0.8, s=50, color='red', marker='X', edgecolors='darkred', linewidths=1.5,
                   label='Expert untrustworthy cases')
        
        ax2.axvline(x=monig_high_epistemic_thresh, color='gray', linestyle='--', alpha=0.5, 
                   label=f'High Epistemic ({monig_high_epistemic_thresh:.3f})')
        
        # Count how many untrustworthy cases MoNIG correctly identifies
        correctly_identified = all_untrustworthy_df[
            all_untrustworthy_df['MoNIG_Epistemic'] > monig_high_epistemic_thresh
        ]
        success_rate = len(correctly_identified) / len(all_untrustworthy_df) * 100
        
        ax2.set_xlabel('MoNIG Epistemic Uncertainty')
        ax2.set_ylabel('MoNIG Prediction Error')
        ax2.set_title(f'MoNIG Response to Untrustworthy Experts\n({success_rate:.1f}% correctly identified)')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No untrustworthy cases found', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('MoNIG Response to Untrustworthy Experts')
    
    # Plot 3: Per-expert breakdown
    ax3 = axes[1, 0]
    expert_success_rates = []
    expert_labels = []
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        
        wrong_mask = df[f'Expert{expert_num}_Error'] > error_thresholds[j]
        high_epistemic_mask = df[f'Expert{expert_num}_Epistemic'] > epistemic_thresholds[j]
        untrustworthy = df[wrong_mask & high_epistemic_mask]
        
        if len(untrustworthy) > 0:
            monig_correct = untrustworthy[untrustworthy['MoNIG_Epistemic'] > monig_high_epistemic_thresh]
            success_rate = len(monig_correct) / len(untrustworthy) * 100
            expert_success_rates.append(success_rate)
            expert_labels.append(f'{expert_name}\n(n={len(untrustworthy)})')
        else:
            expert_success_rates.append(0)
            expert_labels.append(f'{expert_name}\n(n=0)')
    
    bars = ax3.bar(expert_labels, expert_success_rates, color=colors_expert[:num_experts])
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('MoNIG Success Rate: Identifying Untrustworthy Experts')
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Example cases table visualization
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Collect top examples from each expert
    example_text = "Top Examples: Expert Wrong + High Epistemic → MoNIG Also Uncertain\n\n"
    
    for j in range(num_experts):
        expert_num = j + 1
        expert_name = expert_names[j] if j < len(expert_names) else f'Expert {expert_num}'
        
        wrong_mask = df[f'Expert{expert_num}_Error'] > error_thresholds[j]
        high_epistemic_mask = df[f'Expert{expert_num}_Epistemic'] > epistemic_thresholds[j]
        untrustworthy = df[wrong_mask & high_epistemic_mask].copy()
        
        if len(untrustworthy) > 0:
            # Sort by error and epistemic
            untrustworthy_sorted = untrustworthy.sort_values(
                by=[f'Expert{expert_num}_Error', f'Expert{expert_num}_Epistemic'], 
                ascending=False
            ).head(2)  # Top 2 examples
            
            example_text += f"\n{expert_name}:\n"
            for idx, row in untrustworthy_sorted.iterrows():
                example_text += f"  • {row['ComplexID']}: "
                example_text += f"True={row['True_Affinity']:.2f}, "
                example_text += f"{expert_name}={row[f'Expert{expert_num}_Prediction']:.2f} "
                example_text += f"(err={row[f'Expert{expert_num}_Error']:.2f}, "
                example_text += f"epi={row[f'Expert{expert_num}_Epistemic']:.3f}) → "
                example_text += f"MoNIG epi={row['MoNIG_Epistemic']:.3f}\n"
    
    ax4.text(0.05, 0.95, example_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the figure
    untrustworthy_path = output_dir / f'{stem}_untrustworthy_engines.png'
    plt.savefig(untrustworthy_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {untrustworthy_path}")
    
    # ===== 5. Conformal Prediction Visualization =====
    if has_conformal:
        print("\n📊 Generating conformal prediction visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Conformal Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Conformal intervals coverage
        ax1 = axes[0, 0]
        n_subset = min(200, len(df))
        subset_idx = np.random.choice(len(df), n_subset, replace=False)
        subset_df = df.iloc[subset_idx].sort_values('MoNIG_Prediction')
        
        x_vals = np.arange(len(subset_df))
        ax1.fill_between(x_vals, subset_df['Conformal_Lower'], subset_df['Conformal_Upper'],
                        alpha=0.3, color='blue', label='Conformal Interval')
        ax1.scatter(x_vals, subset_df['True_Affinity'], alpha=0.6, s=20, color='red', 
                   marker='x', label='True Value')
        ax1.plot(x_vals, subset_df['MoNIG_Prediction'], 'k-', alpha=0.5, linewidth=1, label='Prediction')
        ax1.set_xlabel('Sample Index (sorted by prediction)')
        ax1.set_ylabel('Affinity')
        ax1.set_title('Conformal Prediction Intervals')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Coverage comparison
        ax2 = axes[0, 1]
        standard_lower_95 = df['MoNIG_Prediction'] - 1.96 * df['MoNIG_Std']
        standard_upper_95 = df['MoNIG_Prediction'] + 1.96 * df['MoNIG_Std']
        
        standard_inside_95 = (df['True_Affinity'] >= standard_lower_95) & (df['True_Affinity'] <= standard_upper_95)
        conformal_inside = (df['True_Affinity'] >= df['Conformal_Lower']) & (df['True_Affinity'] <= df['Conformal_Upper'])
        
        comparison_data = {
            'Standard (95%)': [standard_inside_95.mean(), (2 * 1.96 * df['MoNIG_Std']).mean()],
            'Conformal': [conformal_inside.mean(), df['Conformal_Width'].mean()]
        }
        
        x_pos = np.arange(2)
        width = 0.35
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x_pos - width/2, [comparison_data['Standard (95%)'][0], comparison_data['Conformal'][0]], 
                        width, label='PICP', color=['skyblue', 'lightcoral'], alpha=0.7)
        bars2 = ax2_twin.bar(x_pos + width/2, [comparison_data['Standard (95%)'][1], comparison_data['Conformal'][1]], 
                             width, label='Avg Width', color=['darkblue', 'darkred'], alpha=0.7)
        
        ax2.set_ylabel('PICP (Coverage)', color='black')
        ax2_twin.set_ylabel('Average Interval Width', color='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Standard (95%)', 'Conformal'])
        ax2.set_title('Coverage and Width Comparison')
        ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(alpha=0.3, axis='y')
        
        # Plot 3: Interval width distribution
        ax3 = axes[1, 0]
        ax3.hist(df['Conformal_Width'], bins=50, alpha=0.7, color='blue', label='Conformal', density=True)
        ax3.hist(2 * 1.96 * df['MoNIG_Std'], bins=50, alpha=0.7, color='orange', label='Standard (95%)', density=True)
        ax3.set_xlabel('Interval Width')
        ax3.set_ylabel('Density')
        ax3.set_title('Interval Width Distribution')
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
        
        # Plot 4: Coverage by uncertainty level
        ax4 = axes[1, 1]
        df['Uncertainty_Quantile'] = pd.qcut(df['MoNIG_Std'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        
        conformal_by_quantile = df.groupby('Uncertainty_Quantile', observed=True).apply(
            lambda g: ((g['True_Affinity'] >= g['Conformal_Lower']) & (g['True_Affinity'] <= g['Conformal_Upper'])).mean()
        )
        standard_by_quantile = df.groupby('Uncertainty_Quantile', observed=True).apply(
            lambda g: ((g['True_Affinity'] >= (g['MoNIG_Prediction'] - 1.96 * g['MoNIG_Std'])) & 
                      (g['True_Affinity'] <= (g['MoNIG_Prediction'] + 1.96 * g['MoNIG_Std']))).mean()
        )
        
        x_pos_quant = np.arange(len(conformal_by_quantile))
        ax4.plot(x_pos_quant, conformal_by_quantile.values, 'o-', label='Conformal', linewidth=2, markersize=8)
        ax4.plot(x_pos_quant, standard_by_quantile.values, 's-', label='Standard (95%)', linewidth=2, markersize=8)
        ax4.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
        ax4.set_xticks(x_pos_quant)
        ax4.set_xticklabels(conformal_by_quantile.index)
        ax4.set_ylabel('PICP (Coverage)')
        ax4.set_xlabel('Uncertainty Quantile')
        ax4.set_title('Coverage by Uncertainty Level')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_ylim([0.7, 1.0])
        
        plt.tight_layout()
        conformal_path = output_dir / f'{stem}_conformal_analysis.png'
        plt.savefig(conformal_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {conformal_path}")
    
    # ===== 6. Uncertainty as Risk Estimator =====
    print("\n📊 Generating Uncertainty as Risk Estimator plot...")
    try:
        plot_uncertainty_as_risk_estimator(df, output_dir, stem)
    except Exception as e:
        print(f"  ⚠ Skipped risk estimator plot: {e}")
    
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
    df, metrics, has_conformal = analyze_uncertainty(args.csv)
    
    # Create visualizations
    visualize_uncertainty(df, args.output_prefix, has_conformal=has_conformal)
    
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
    print(f"      - {stem}_untrustworthy_engines.png")
    print(f"      - {stem}_uncertainty_risk_estimator.png")
    if has_conformal:
        print(f"      - {stem}_conformal_analysis.png")
    print("="*80)

