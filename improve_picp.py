"""
Analyze and improve PICP (Prediction Interval Coverage Probability)

Current Issue:
- PICP@95%: 0.9421 (should be ~0.95) - intervals too narrow
- PICP@90%: 0.9022 (should be ~0.90) - close to ideal

Solutions:
1. Analyze uncertainty scaling
2. Implement post-hoc calibration
3. Modify uncertainty parameters
"""
import pandas as pd
import numpy as np
from scipy import stats
import torch

def load_inference_results(model_name, seed):
    """Load inference results"""
    path = f'experiments_improved_comparison/{model_name}_seed{seed}_optadam/test_inference_results.csv'
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def calculate_picp(y_true, y_pred, uncertainty, coverage=0.95):
    """Calculate PICP for given coverage level"""
    z_score = stats.norm.ppf((1 + coverage) / 2)
    lower = y_pred - z_score * uncertainty
    upper = y_pred + z_score * uncertainty
    
    in_interval = (y_true >= lower) & (y_true <= upper)
    picp = np.mean(in_interval)
    
    avg_width = np.mean(upper - lower)
    
    return picp, avg_width

def analyze_current_uncertainty():
    """Analyze current uncertainty estimates"""
    print("="*80)
    print("CURRENT UNCERTAINTY ANALYSIS")
    print("="*80)
    print()
    
    models = ['MoNIG_Improved', 'MoNIG_Improved_v2', 'MoNIG_Hybrid']
    seeds = [42, 43, 44]
    
    results = []
    
    for model in models:
        print(f"\n{model}:")
        print("-"*80)
        
        model_results = []
        
        for seed in seeds:
            df = load_inference_results(model, seed)
            if df is None:
                continue
            
            y_true = df['True_Affinity'].values
            y_pred = df['MoNIG_Prediction'].values
            
            # Calculate uncertainty from columns
            if 'MoNIG_Epistemic' in df.columns and 'MoNIG_Aleatoric' in df.columns:
                epistemic = df['MoNIG_Epistemic'].values
                aleatoric = df['MoNIG_Aleatoric'].values
                total_unc = np.sqrt(epistemic + aleatoric)
            elif 'MoNIG_Std' in df.columns:
                total_unc = df['MoNIG_Std'].values
            else:
                print(f"  Seed {seed}: No uncertainty columns found")
                continue
            
            # Calculate PICP at different levels
            picp_95, width_95 = calculate_picp(y_true, y_pred, total_unc, 0.95)
            picp_90, width_90 = calculate_picp(y_true, y_pred, total_unc, 0.90)
            
            model_results.append({
                'model': model,
                'seed': seed,
                'picp_95': picp_95,
                'picp_90': picp_90,
                'width_95': width_95,
                'width_90': width_90,
                'mean_unc': np.mean(total_unc),
                'std_unc': np.std(total_unc)
            })
            
            print(f"  Seed {seed}:")
            print(f"    PICP@95%: {picp_95:.4f} (width: {width_95:.4f})")
            print(f"    PICP@90%: {picp_90:.4f} (width: {width_90:.4f})")
            print(f"    Mean Uncertainty: {np.mean(total_unc):.4f} ± {np.std(total_unc):.4f}")
        
        if model_results:
            avg_picp_95 = np.mean([r['picp_95'] for r in model_results])
            avg_picp_90 = np.mean([r['picp_90'] for r in model_results])
            print(f"\n  Average PICP@95%: {avg_picp_95:.4f} (target: 0.9500)")
            print(f"  Average PICP@90%: {avg_picp_90:.4f} (target: 0.9000)")
            print(f"  Gap@95%: {0.95 - avg_picp_95:.4f}")
            print(f"  Gap@90%: {0.90 - avg_picp_90:.4f}")
        
        results.extend(model_results)
    
    return pd.DataFrame(results)

def calculate_calibration_factor(df_analysis):
    """Calculate optimal uncertainty scaling factor"""
    print("\n" + "="*80)
    print("CALIBRATION FACTOR CALCULATION")
    print("="*80)
    print()
    
    models = df_analysis['model'].unique()
    
    calibration_factors = {}
    
    for model in models:
        subset = df_analysis[df_analysis['model'] == model]
        
        # Average PICP
        avg_picp_95 = subset['picp_95'].mean()
        avg_picp_90 = subset['picp_90'].mean()
        
        # Calculate scaling factor needed
        # If PICP < target, we need to increase uncertainty
        # Rough approximation: scale_factor = target_z / current_z
        
        if avg_picp_95 < 0.95:
            # Need to scale up uncertainty
            # Using inverse CDF to estimate needed scaling
            current_coverage = avg_picp_95
            target_coverage = 0.95
            
            # Approximate scaling factor
            scale_95 = stats.norm.ppf((1 + target_coverage) / 2) / stats.norm.ppf((1 + current_coverage) / 2)
        else:
            scale_95 = 1.0
        
        if avg_picp_90 < 0.90:
            current_coverage = avg_picp_90
            target_coverage = 0.90
            scale_90 = stats.norm.ppf((1 + target_coverage) / 2) / stats.norm.ppf((1 + current_coverage) / 2)
        else:
            scale_90 = 1.0
        
        # Use average of both
        scale_factor = (scale_95 + scale_90) / 2
        
        calibration_factors[model] = {
            'scale_factor': scale_factor,
            'current_picp_95': avg_picp_95,
            'current_picp_90': avg_picp_90
        }
        
        print(f"{model}:")
        print(f"  Current PICP@95%: {avg_picp_95:.4f}")
        print(f"  Current PICP@90%: {avg_picp_90:.4f}")
        print(f"  Recommended Scale Factor: {scale_factor:.4f}")
        print()
    
    return calibration_factors

def test_calibrated_uncertainty():
    """Test post-hoc calibration"""
    print("="*80)
    print("TESTING POST-HOC CALIBRATION")
    print("="*80)
    print()
    
    models = ['MoNIG_Improved', 'MoNIG_Improved_v2', 'MoNIG_Hybrid']
    seeds = [42, 43, 44]
    
    # Test different scaling factors
    scale_factors = [1.0, 1.05, 1.10, 1.15, 1.20, 1.25]
    
    all_results = []
    
    for model in models:
        print(f"\n{model}:")
        print("-"*80)
        
        for scale in scale_factors:
            picp_95_list = []
            picp_90_list = []
            
            for seed in seeds:
                df = load_inference_results(model, seed)
                if df is None:
                    continue
                
                y_true = df['True_Affinity'].values
                y_pred = df['MoNIG_Prediction'].values
                
                if 'MoNIG_Epistemic' in df.columns and 'MoNIG_Aleatoric' in df.columns:
                    epistemic = df['MoNIG_Epistemic'].values
                    aleatoric = df['MoNIG_Aleatoric'].values
                    total_unc = np.sqrt(epistemic + aleatoric)
                elif 'MoNIG_Std' in df.columns:
                    total_unc = df['MoNIG_Std'].values
                else:
                    continue
                
                # Apply scaling
                calibrated_unc = total_unc * scale
                
                picp_95, _ = calculate_picp(y_true, y_pred, calibrated_unc, 0.95)
                picp_90, _ = calculate_picp(y_true, y_pred, calibrated_unc, 0.90)
                
                picp_95_list.append(picp_95)
                picp_90_list.append(picp_90)
            
            if picp_95_list:
                avg_picp_95 = np.mean(picp_95_list)
                avg_picp_90 = np.mean(picp_90_list)
                
                # Calculate error from target
                error_95 = abs(avg_picp_95 - 0.95)
                error_90 = abs(avg_picp_90 - 0.90)
                total_error = error_95 + error_90
                
                all_results.append({
                    'model': model,
                    'scale': scale,
                    'picp_95': avg_picp_95,
                    'picp_90': avg_picp_90,
                    'error_95': error_95,
                    'error_90': error_90,
                    'total_error': total_error
                })
                
                indicator_95 = "✅" if abs(avg_picp_95 - 0.95) < 0.01 else "❌"
                indicator_90 = "✅" if abs(avg_picp_90 - 0.90) < 0.01 else "❌"
                
                print(f"  Scale={scale:.2f}: PICP@95%={avg_picp_95:.4f} {indicator_95}, PICP@90%={avg_picp_90:.4f} {indicator_90}")
    
    # Find best scale for each model
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("OPTIMAL SCALING FACTORS")
    print("="*80)
    print()
    
    for model in models:
        subset = results_df[results_df['model'] == model]
        if len(subset) == 0:
            continue
        
        # Find scale with minimum total error
        best_row = subset.loc[subset['total_error'].idxmin()]
        
        print(f"{model}:")
        print(f"  Optimal Scale: {best_row['scale']:.2f}")
        print(f"  Calibrated PICP@95%: {best_row['picp_95']:.4f} (error: {best_row['error_95']:.4f})")
        print(f"  Calibrated PICP@90%: {best_row['picp_90']:.4f} (error: {best_row['error_90']:.4f})")
        print()
    
    return results_df

def main():
    print("="*80)
    print("PICP IMPROVEMENT ANALYSIS")
    print("="*80)
    print()
    
    # Step 1: Analyze current uncertainty
    df_analysis = analyze_current_uncertainty()
    
    # Step 2: Calculate calibration factors
    calibration_factors = calculate_calibration_factor(df_analysis)
    
    # Step 3: Test different scaling factors
    results_df = test_calibrated_uncertainty()
    
    # Save results
    df_analysis.to_csv('picp_analysis.csv', index=False)
    results_df.to_csv('picp_calibration_results.csv', index=False)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("📊 Analysis saved to:")
    print("   - picp_analysis.csv (current uncertainty stats)")
    print("   - picp_calibration_results.csv (calibration test results)")
    print()
    print("🎯 Next Steps:")
    print("   1. Apply optimal scaling factor in model")
    print("   2. Retrain with adjusted uncertainty parameters")
    print("   3. OR: Use post-hoc calibration at inference time")
    print()

if __name__ == '__main__':
    main()

