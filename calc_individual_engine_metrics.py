#!/usr/bin/env python3
"""
Calculate individual engine metrics (MAE, RMSE, Corr, R²) on test data.
Uses test_pdbs.csv to filter test samples from pdbbind_descriptors_with_experts_and_binding.csv.
"""

import numpy as np
import pandas as pd
import re


def parse_affinity(affinity_strings):
    """
    Parse binding affinity from strings like 'Kd=6.67uM', 'Ki=19uM'
    Convert to pKd/pKi values (negative log of molar concentration)
    Same as src/drug_dataset_emb.py
    """
    labels = []
    valid_mask = []
    
    for s in affinity_strings:
        try:
            s_str = str(s)
            
            # Skip inequality values
            if '>' in s_str or '<' in s_str:
                labels.append(np.nan)
                valid_mask.append(False)
                continue
            
            # Extract numeric value and unit
            match = re.search(r'([0-9.]+)([a-zA-Z]+)', s_str)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                # Convert to Molar (check longer units first)
                if 'fm' in unit:
                    molar = value * 1e-15
                elif 'pm' in unit:
                    molar = value * 1e-12
                elif 'nm' in unit:
                    molar = value * 1e-9
                elif 'um' in unit or 'μm' in unit:
                    molar = value * 1e-6
                elif 'mm' in unit:
                    molar = value * 1e-3
                elif unit == 'm':
                    molar = value
                else:
                    labels.append(np.nan)
                    valid_mask.append(False)
                    continue
                
                p_value = -np.log10(molar)
                labels.append(p_value)
                valid_mask.append(True)
            else:
                labels.append(np.nan)
                valid_mask.append(False)
        except Exception:
            labels.append(np.nan)
            valid_mask.append(False)
    
    return np.array(labels, dtype=np.float32), np.array(valid_mask, dtype=bool)


def compute_metrics(y_true, y_pred, name=""):
    """Compute regression metrics (same as run_multi_seed_experiments.py)."""
    # MAE
    mae = np.mean(np.abs(y_pred - y_true))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Correlation (Pearson)
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        'Engine': name,
        'MAE': mae,
        'RMSE': rmse,
        'Corr': corr,
        'R2': r2
    }


def main():
    # Load test PDB IDs
    test_pdbs = pd.read_csv('data/test_pdbs.csv', header=None)[0].str.lower().tolist()
    print(f"Loaded {len(test_pdbs)} test PDB IDs")
    
    # Load main dataset
    df = pd.read_csv('pdbbind_descriptors_with_experts_and_binding.csv')
    print(f"Loaded {len(df)} total samples")
    
    # Filter to test set
    df['ComplexID_lower'] = df['ComplexID'].str.lower()
    test_df = df[df['ComplexID_lower'].isin(test_pdbs)].copy()
    print(f"Found {len(test_df)} test samples")
    
    # Parse binding affinity strings to numeric pKd values
    y_true_parsed, affinity_valid = parse_affinity(test_df['Binding_Affinity'].values)
    test_df['pKd'] = y_true_parsed
    test_df['affinity_valid'] = affinity_valid
    print(f"Parsed binding affinity: {affinity_valid.sum()} valid out of {len(affinity_valid)}")
    
    # Engine columns and their names
    engines = {
        'GNINA': 'GNINA_Affinity',
        'BIND': 'BIND_pIC50',
        'FlowDock': 'flowdock_score',
        'BALM': 'balm_prediction',
        'DynamicBind': 'DynamicBind_score'
    }
    
    results = []
    
    for engine_name, col_name in engines.items():
        # Convert engine score to numeric (coerce errors to NaN)
        test_df[col_name] = pd.to_numeric(test_df[col_name], errors='coerce')
        
        # Filter out NaN values for this engine AND valid binding affinity
        valid_mask = ~test_df[col_name].isna() & test_df['affinity_valid']
        valid_df = test_df[valid_mask]
        
        if len(valid_df) == 0:
            print(f"WARNING: No valid samples for {engine_name}")
            continue
        
        y_true = valid_df['pKd'].values.astype(np.float64)
        y_pred = valid_df[col_name].values.astype(np.float64)
        
        metrics = compute_metrics(y_true, y_pred, engine_name)
        metrics['N_samples'] = len(valid_df)
        results.append(metrics)
        
        print(f"\n{engine_name}:")
        print(f"  N samples: {len(valid_df)}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  Corr: {metrics['Corr']:.4f}")
        print(f"  R²:   {metrics['R2']:.4f}")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = 'individual_engine_metrics_test_pdbs.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to {output_path}")
    
    # Print formatted table
    print("\n" + "="*70)
    print("Individual Engine Metrics on Test Set (test_pdbs.csv)")
    print("="*70)
    print(f"{'Engine':<15} {'MAE':>10} {'RMSE':>10} {'Corr':>10} {'R²':>10} {'N':>8}")
    print("-"*70)
    for _, row in results_df.iterrows():
        print(f"{row['Engine']:<15} {row['MAE']:>10.4f} {row['RMSE']:>10.4f} {row['Corr']:>10.4f} {row['R2']:>10.4f} {int(row['N_samples']):>8}")
    print("="*70)


if __name__ == '__main__':
    main()

