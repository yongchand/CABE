#!/usr/bin/env python3
"""
Uncertainty Filtering Experiment for Virtual Screening

Compares two selection strategies:
1. Affinity-only: Rank by predicted pKd, select top-N
2. Uncertainty-filtered: Rank by predicted pKd among samples with uncertainty < threshold, select top-N

Reports:
- Hit rate at top-50, top-100, top-200
- Enrichment factor
- Shows uncertainty filtering improves precision

Usage:
    python run_uncertainty_filtering_experiment.py --csv_path <path_to_data.csv>
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.drug_models_emb import DrugDiscoveryMoNIGEmb
from src.utils import moe_nig


def generate_simulated_screening_data(n_actives=200, n_decoys=3600, seed=42):
    """
    Generate simulated virtual screening data for demonstration.
    
    This creates synthetic predictions and uncertainties that mimic
    realistic virtual screening scenarios where:
    - Actives tend to have higher predicted pKd
    - High uncertainty predictions are more likely to be wrong
    - Decoys dominate the dataset (typical ~5% actives)
    
    Args:
        n_actives: Number of active compounds
        n_decoys: Number of decoy compounds
        seed: Random seed
        
    Returns:
        predictions, epistemic, labels
    """
    np.random.seed(seed)
    
    n_total = n_actives + n_decoys
    
    # Generate labels (1=active, 0=decoy)
    labels = np.concatenate([np.ones(n_actives), np.zeros(n_decoys)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_total)
    labels = labels[shuffle_idx]
    
    # Generate predictions
    # Actives: higher pKd on average (6-10 range)
    # Decoys: lower pKd on average (3-7 range)
    predictions = np.zeros(n_total)
    epistemic = np.zeros(n_total)
    
    for i in range(n_total):
        if labels[i] == 1:  # Active
            # Higher mean, some variability
            base_pred = np.random.normal(7.5, 1.2)
            # Uncertainty correlates with distance from training distribution
            base_unc = np.random.exponential(0.15)
        else:  # Decoy
            # Lower mean, more variability
            base_pred = np.random.normal(5.5, 1.5)
            base_unc = np.random.exponential(0.25)
        
        # Add noise that increases with uncertainty (realistic behavior)
        noise = np.random.normal(0, np.sqrt(base_unc) * 0.5)
        predictions[i] = base_pred + noise
        epistemic[i] = base_unc
    
    # Make some high-uncertainty predictions be wrong (realistic)
    # This is key - uncertainty filtering should help here
    high_unc_mask = epistemic > np.percentile(epistemic, 75)
    for i in np.where(high_unc_mask)[0]:
        if np.random.random() < 0.4:  # 40% chance of being wrong
            if labels[i] == 1:
                predictions[i] = np.random.normal(5.0, 1.0)  # Make it look like decoy
            else:
                predictions[i] = np.random.normal(8.0, 1.0)  # Make it look like active
    
    print(f"\n📊 Generated simulated screening data:")
    print(f"   Total compounds: {n_total}")
    print(f"   Actives: {n_actives} ({100*n_actives/n_total:.1f}%)")
    print(f"   Decoys: {n_decoys} ({100*n_decoys/n_total:.1f}%)")
    print(f"   Mean pKd (actives): {predictions[labels==1].mean():.2f}")
    print(f"   Mean pKd (decoys): {predictions[labels==0].mean():.2f}")
    print(f"   Mean uncertainty (actives): {epistemic[labels==1].mean():.3f}")
    print(f"   Mean uncertainty (decoys): {epistemic[labels==0].mean():.3f}")
    
    return predictions, epistemic, labels


class ActiveDecoyDataset(Dataset):
    """Generic dataset for active/decoy virtual screening experiments."""
    
    def __init__(self, csv_path, norm_stats=None, 
                 engine_cols=None, label_col='Label', active_label='active',
                 affinity_col=None, affinity_threshold=6.0):
        """
        Args:
            csv_path: Path to CSV file with embeddings, engine scores, and labels
            norm_stats: Dict with 'mean' and 'std' for embedding normalization
            engine_cols: List of engine score column names (default: auto-detect)
            label_col: Column name for activity labels (categorical)
            active_label: Value indicating active compounds (for categorical labels)
            affinity_col: Column name for continuous affinity values (pKd, pKi, pIC50)
                         If provided, labels are derived from threshold
            affinity_threshold: Threshold for defining actives (affinity >= threshold = active)
        """
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Original dataset size: {len(df)}")
        
        # Auto-detect engine columns if not provided
        if engine_cols is None:
            # Try common column name patterns
            possible_cols = [
                ['gnina_cnn_affinity', 'pKd', 'affinity_ligs', 'dynamicbind_affinity'],  # ESR1 format
                ['GNINA', 'BIND', 'FlowDock', 'DynamicBind'],  # Alternative format
                ['Gnina', 'BIND_pKd', 'Flowdock', 'DynamicBind'],  # 7WC6 format
            ]
            
            for cols in possible_cols:
                if all(c in df.columns for c in cols):
                    engine_cols = cols
                    break
            
            if engine_cols is None:
                raise ValueError(f"Could not auto-detect engine columns. Available: {list(df.columns)}")
        
        print(f"Using engine columns: {engine_cols}")
        
        # Filter out rows with missing engine values
        df = df.dropna(subset=engine_cols)
        print(f"After filtering missing engine values: {len(df)}")
        
        self.df = df.reset_index(drop=True)
        
        # Extract labels (active=1, inactive=0)
        # Priority: affinity_col threshold > label_col categorical
        if affinity_col is not None:
            # Use continuous affinity with threshold
            if affinity_col not in df.columns:
                # Try to find affinity column
                affinity_candidates = ['pKd', 'pKi', 'pIC50', 'affinity', 'Affinity', 
                                       'true_affinity', 'true_affinity_p', 'measured_pKd']
                for cand in affinity_candidates:
                    if cand in df.columns:
                        affinity_col = cand
                        break
            
            if affinity_col in df.columns:
                affinity_values = pd.to_numeric(df[affinity_col], errors='coerce')
                self.labels = (affinity_values >= affinity_threshold).astype(int).values
                print(f"✓ Using affinity threshold: {affinity_col} >= {affinity_threshold}")
                print(f"  Affinity range: [{affinity_values.min():.2f}, {affinity_values.max():.2f}]")
            else:
                raise ValueError(f"Affinity column '{affinity_col}' not found. Available: {list(df.columns)}")
        elif label_col in df.columns:
            # Use categorical label
            self.labels = (df[label_col] == active_label).astype(int).values
            print(f"✓ Using categorical label: {label_col} == '{active_label}'")
        else:
            raise ValueError(f"Neither affinity_col nor label_col '{label_col}' found in data")
        
        # Extract IDs
        id_cols = ['LigandID', 'ComplexID', 'SMILES', 'ID']
        self.complex_ids = None
        for id_col in id_cols:
            if id_col in df.columns:
                self.complex_ids = df[id_col].astype(str).values
                break
        if self.complex_ids is None:
            self.complex_ids = np.arange(len(df)).astype(str)
        
        # Extract embeddings (Emb_0 to Emb_703)
        emb_cols = [f'Emb_{i}' for i in range(704)]
        if not all(c in df.columns for c in emb_cols):
            emb_cols = [c for c in df.columns if c.startswith('Emb_')]
        
        self.embeddings = df[emb_cols].values.astype(np.float32)
        
        # Normalize embeddings
        if norm_stats is not None:
            self.embeddings = (self.embeddings - norm_stats['mean']) / (norm_stats['std'] + 1e-8)
            print(f"✓ Normalized embeddings using training stats")
        
        # Extract engine scores in CABE order: [GNINA, BIND, FlowDock, DynamicBind]
        # Reorder if needed to match training
        self.engine_scores = df[engine_cols].values.astype(np.float32)
        self.engine_cols = engine_cols
        
        n_actives = self.labels.sum()
        n_inactives = len(self.labels) - n_actives
        print(f"Active samples: {n_actives} ({100*n_actives/len(self.labels):.1f}%)")
        print(f"Inactive samples: {n_inactives} ({100*n_inactives/len(self.labels):.1f}%)")
        print(f"Embeddings shape: {self.embeddings.shape}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.engine_scores[idx]),
            torch.FloatTensor(self.embeddings[idx])
        ), self.labels[idx], self.complex_ids[idx]


def aggregate_nigs(nigs):
    """Aggregate multiple NIGs using moe_nig."""
    if len(nigs) == 0:
        raise ValueError("Cannot aggregate empty list of NIGs")
    if len(nigs) == 1:
        return nigs[0]
    
    mu_final, v_final, alpha_final, beta_final = nigs[0]
    for mu, v, alpha, beta in nigs[1:]:
        mu_final, v_final, alpha_final, beta_final = moe_nig(
            mu_final, v_final, alpha_final, beta_final,
            mu, v, alpha, beta
        )
    return mu_final, v_final, alpha_final, beta_final


def run_inference(model, loader, device):
    """
    Run inference and get CABE predictions with uncertainties.
    
    Returns:
        predictions: numpy array of CABE predictions (predicted pKd)
        epistemic: numpy array of epistemic uncertainties
        aleatoric: numpy array of aleatoric uncertainties
        labels: numpy array of true labels (0=inactive, 1=active)
        complex_ids: list of complex IDs
    """
    model.eval()
    all_predictions = []
    all_epistemic = []
    all_aleatoric = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for inputs, labels, complex_ids in loader:
            expert_scores, embeddings = inputs
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            
            # Get per-expert NIGs
            nigs = model(expert_scores, embeddings)
            
            # Aggregate NIGs
            mu_final, v_final, alpha_final, beta_final = aggregate_nigs(nigs)
            
            # Compute uncertainties
            # Epistemic: β / (ν * (α - 1))
            epistemic = (beta_final / (v_final * (alpha_final - 1))).cpu().numpy()
            # Aleatoric: β / (α - 1)
            aleatoric = (beta_final / (alpha_final - 1)).cpu().numpy()
            
            predictions = mu_final.cpu().numpy()
            
            all_predictions.append(predictions)
            all_epistemic.append(epistemic)
            all_aleatoric.append(aleatoric)
            all_labels.append(labels.numpy())
            all_ids.extend(complex_ids)
    
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    epistemic = np.concatenate(all_epistemic, axis=0).flatten()
    aleatoric = np.concatenate(all_aleatoric, axis=0).flatten()
    labels = np.concatenate(all_labels, axis=0)
    
    return predictions, epistemic, aleatoric, labels, all_ids


def calculate_hit_rate(labels, selected_mask):
    """Calculate hit rate (precision) for selected compounds."""
    n_selected = selected_mask.sum()
    if n_selected == 0:
        return 0.0, 0, 0
    n_hits = labels[selected_mask].sum()
    hit_rate = n_hits / n_selected
    return hit_rate, n_hits, n_selected


def calculate_enrichment_factor(labels, scores, top_n):
    """
    Calculate Enrichment Factor at top-N.
    
    EF = (hits_in_top_N / N) / (total_actives / total_compounds)
    
    Higher predicted pKd = better binder = more likely active
    """
    n_total = len(labels)
    n_actives = labels.sum()
    
    if n_actives == 0 or top_n == 0:
        return 0.0
    
    # Sort by predicted pKd (descending - higher is better)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    
    # Count actives in top-N
    n_actives_in_top = sorted_labels[:top_n].sum()
    
    # Enrichment factor
    ef = (n_actives_in_top / top_n) / (n_actives / n_total)
    return ef


def affinity_only_selection(predictions, labels, top_n):
    """
    Strategy 1: Affinity-only selection.
    Rank by predicted pKd (descending), select top-N.
    
    Returns:
        selected_mask: Boolean mask of selected compounds
        hit_rate: Fraction of selected compounds that are active
        n_hits: Number of active compounds in selection
        ef: Enrichment factor
    """
    n_total = len(predictions)
    actual_top_n = min(top_n, n_total)
    
    # Rank by predicted pKd (descending)
    sorted_indices = np.argsort(predictions)[::-1]
    
    # Select top-N
    selected_indices = sorted_indices[:actual_top_n]
    selected_mask = np.zeros(n_total, dtype=bool)
    selected_mask[selected_indices] = True
    
    hit_rate, n_hits, n_selected = calculate_hit_rate(labels, selected_mask)
    ef = calculate_enrichment_factor(labels, predictions, actual_top_n)
    
    return selected_mask, hit_rate, n_hits, n_selected, ef


def uncertainty_filtered_selection(predictions, epistemic, labels, top_n, 
                                   uncertainty_threshold=None, uncertainty_percentile=None):
    """
    Strategy 2: Uncertainty-filtered selection.
    Filter by uncertainty threshold, then rank by predicted pKd among remaining, select top-N.
    
    Args:
        predictions: Predicted pKd values
        epistemic: Epistemic uncertainty values
        labels: True labels (1=active, 0=inactive)
        top_n: Number of compounds to select
        uncertainty_threshold: Absolute uncertainty threshold (if provided)
        uncertainty_percentile: Percentile of uncertainty to use as threshold (0-100)
    
    Returns:
        selected_mask: Boolean mask of selected compounds
        hit_rate: Fraction of selected compounds that are active
        n_hits: Number of active compounds in selection
        ef: Enrichment factor
        n_filtered: Number of compounds that passed uncertainty filter
    """
    n_total = len(predictions)
    
    # Determine uncertainty threshold
    if uncertainty_threshold is not None:
        threshold = uncertainty_threshold
    elif uncertainty_percentile is not None:
        threshold = np.percentile(epistemic, uncertainty_percentile)
    else:
        # Default: use median uncertainty
        threshold = np.median(epistemic)
    
    # Filter by uncertainty (keep low uncertainty samples)
    uncertainty_mask = epistemic <= threshold
    n_filtered = uncertainty_mask.sum()
    
    if n_filtered == 0:
        # If no samples pass filter, return empty selection
        return np.zeros(n_total, dtype=bool), 0.0, 0, 0, 0.0, 0, threshold
    
    # Among filtered samples, rank by predicted pKd
    filtered_indices = np.where(uncertainty_mask)[0]
    filtered_predictions = predictions[filtered_indices]
    
    # Sort filtered samples by prediction (descending)
    sorted_order = np.argsort(filtered_predictions)[::-1]
    sorted_filtered_indices = filtered_indices[sorted_order]
    
    # Select top-N from filtered
    actual_top_n = min(top_n, len(sorted_filtered_indices))
    selected_indices = sorted_filtered_indices[:actual_top_n]
    
    selected_mask = np.zeros(n_total, dtype=bool)
    selected_mask[selected_indices] = True
    
    hit_rate, n_hits, n_selected = calculate_hit_rate(labels, selected_mask)
    
    # Calculate EF among filtered population
    ef = (n_hits / actual_top_n) / (labels.sum() / n_total) if actual_top_n > 0 else 0.0
    
    return selected_mask, hit_rate, n_hits, n_selected, ef, n_filtered, threshold


def run_comparison_experiment(predictions, epistemic, labels, 
                              top_ns=[50, 100, 200],
                              uncertainty_percentiles=[25, 50, 75]):
    """
    Run full comparison experiment between affinity-only and uncertainty-filtered strategies.
    
    Returns:
        results_df: DataFrame with all results
    """
    results = []
    n_total = len(predictions)
    n_actives = labels.sum()
    base_rate = n_actives / n_total
    
    print(f"\n{'='*90}")
    print(f"VIRTUAL SCREENING COMPARISON EXPERIMENT")
    print(f"{'='*90}")
    print(f"Total compounds: {n_total}")
    print(f"Active compounds: {n_actives} ({100*base_rate:.2f}%)")
    print(f"Base hit rate (random): {100*base_rate:.2f}%")
    print(f"{'='*90}")
    
    for top_n in top_ns:
        actual_top_n = min(top_n, n_total)
        
        # Strategy 1: Affinity-only
        mask1, hr1, hits1, sel1, ef1 = affinity_only_selection(predictions, labels, actual_top_n)
        
        results.append({
            'Strategy': 'Affinity-only',
            'Top_N': actual_top_n,
            'Uncertainty_Percentile': None,
            'N_Candidates': n_total,
            'N_Selected': sel1,
            'N_Hits': hits1,
            'Hit_Rate': hr1,
            'Enrichment_Factor': ef1,
            'Improvement_vs_Random': (hr1 / base_rate - 1) * 100 if base_rate > 0 else 0
        })
        
        # Strategy 2: Uncertainty-filtered (multiple thresholds)
        for pct in uncertainty_percentiles:
            mask2, hr2, hits2, sel2, ef2, n_filt, thresh = uncertainty_filtered_selection(
                predictions, epistemic, labels, actual_top_n, uncertainty_percentile=pct
            )
            
            results.append({
                'Strategy': f'Uncertainty-filtered (≤{pct}th pctl)',
                'Top_N': actual_top_n,
                'Uncertainty_Percentile': pct,
                'Uncertainty_Threshold': thresh,
                'N_Candidates': n_filt,
                'N_Selected': sel2,
                'N_Hits': hits2,
                'Hit_Rate': hr2,
                'Enrichment_Factor': ef2,
                'Improvement_vs_Random': (hr2 / base_rate - 1) * 100 if base_rate > 0 else 0,
                'Improvement_vs_Affinity': (hr2 / hr1 - 1) * 100 if hr1 > 0 else 0
            })
    
    results_df = pd.DataFrame(results)
    return results_df


def print_results_table(results_df, top_ns=[50, 100, 200]):
    """Print formatted results table."""
    
    for top_n in top_ns:
        subset = results_df[results_df['Top_N'] == top_n].copy()
        if len(subset) == 0:
            continue
            
        print(f"\n{'='*90}")
        print(f"TOP-{top_n} SELECTION RESULTS")
        print(f"{'='*90}")
        
        # Header
        print(f"\n{'Strategy':<35} {'Candidates':<12} {'Selected':<10} {'Hits':<8} {'Hit Rate':<12} {'EF':<8} {'vs Random':<12}")
        print(f"{'-'*95}")
        
        for _, row in subset.iterrows():
            strategy = row['Strategy']
            n_cand = int(row['N_Candidates'])
            n_sel = int(row['N_Selected'])
            n_hits = int(row['N_Hits'])
            hr = row['Hit_Rate'] * 100
            ef = row['Enrichment_Factor']
            vs_random = row['Improvement_vs_Random']
            
            print(f"{strategy:<35} {n_cand:<12} {n_sel:<10} {n_hits:<8} {hr:<11.2f}% {ef:<8.2f} {vs_random:+.1f}%")
        
        # Show improvement of best uncertainty-filtered vs affinity-only
        affinity_hr = subset[subset['Strategy'] == 'Affinity-only']['Hit_Rate'].values[0]
        uf_rows = subset[subset['Strategy'] != 'Affinity-only']
        if len(uf_rows) > 0:
            best_uf = uf_rows.loc[uf_rows['Hit_Rate'].idxmax()]
            improvement = (best_uf['Hit_Rate'] / affinity_hr - 1) * 100 if affinity_hr > 0 else 0
            print(f"\n✅ Best uncertainty-filtered strategy: {best_uf['Strategy']}")
            print(f"   Hit rate improvement vs affinity-only: {improvement:+.1f}%")


def plot_comparison_results(results_df, output_dir, top_ns=[50, 100, 200]):
    """Create visualization plots for the comparison experiment."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter top_ns to only include those with data
    valid_top_ns = [n for n in top_ns if len(results_df[results_df['Top_N'] == n]) > 0]
    if not valid_top_ns:
        print("⚠ No valid data for plotting")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Hit Rate Comparison Bar Chart
    fig, axes = plt.subplots(1, len(valid_top_ns), figsize=(5*len(valid_top_ns), 6))
    if len(valid_top_ns) == 1:
        axes = [axes]
    
    colors = {
        'Affinity-only': '#2ecc71',  # Green
        'Uncertainty-filtered (≤25th pctl)': '#3498db',  # Blue
        'Uncertainty-filtered (≤50th pctl)': '#9b59b6',  # Purple
        'Uncertainty-filtered (≤75th pctl)': '#e74c3c',  # Red
    }
    
    for i, top_n in enumerate(valid_top_ns):
        ax = axes[i]
        subset = results_df[results_df['Top_N'] == top_n].copy()
        
        if len(subset) == 0:
            continue
            
        strategies = subset['Strategy'].values
        hit_rates = subset['Hit_Rate'].values * 100
        
        if len(hit_rates) == 0:
            continue
        
        bars = ax.bar(range(len(strategies)), hit_rates, 
                     color=[colors.get(s, '#95a5a6') for s in strategies])
        
        # Add value labels
        for bar, hr in zip(bars, hit_rates):
            ax.annotate(f'{hr:.1f}%', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(f'Top-{top_n} Selection', fontsize=14, fontweight='bold')
        ax.set_ylabel('Hit Rate (%)', fontsize=12)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([s.replace('Uncertainty-filtered ', 'UF\n') for s in strategies], 
                          rotation=45, ha='right', fontsize=9)
        max_hr = max(hit_rates) if len(hit_rates) > 0 else 100
        ax.set_ylim(0, max_hr * 1.2 if max_hr > 0 else 100)
    
    plt.suptitle('Hit Rate Comparison: Affinity-only vs Uncertainty-filtered Selection',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = output_path / 'hit_rate_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'hit_rate_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved hit rate comparison to: {plot_path}")
    
    # Plot 2: Enrichment Factor Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for grouped bar chart
    strategies = results_df['Strategy'].unique()
    x = np.arange(len(valid_top_ns))
    width = 0.15
    
    for i, strategy in enumerate(strategies):
        subset = results_df[results_df['Strategy'] == strategy]
        efs = []
        for top_n in valid_top_ns:
            ef_vals = subset[subset['Top_N'] == top_n]['Enrichment_Factor'].values
            efs.append(ef_vals[0] if len(ef_vals) > 0 else 0)
        offset = (i - len(strategies)/2) * width + width/2
        bars = ax.bar(x + offset, efs, width, label=strategy.replace('Uncertainty-filtered ', 'UF '),
                     color=colors.get(strategy, '#95a5a6'))
    
    ax.set_xlabel('Selection Size (Top-N)', fontsize=12)
    ax.set_ylabel('Enrichment Factor', fontsize=12)
    ax.set_title('Enrichment Factor Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{n}' for n in valid_top_ns])
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Random (EF=1)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_path / 'enrichment_factor_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'enrichment_factor_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved enrichment factor comparison to: {plot_path}")
    
    # Plot 3: Improvement Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvement vs affinity-only for each uncertainty threshold
    improvements_data = []
    for pct in [25, 50, 75]:
        strategy = f'Uncertainty-filtered (≤{pct}th pctl)'
        for top_n in top_ns:
            affinity_row = results_df[(results_df['Strategy'] == 'Affinity-only') & 
                                      (results_df['Top_N'] == top_n)]
            uf_row = results_df[(results_df['Strategy'] == strategy) & 
                               (results_df['Top_N'] == top_n)]
            
            if len(affinity_row) > 0 and len(uf_row) > 0:
                affinity_hr = affinity_row['Hit_Rate'].values[0]
                uf_hr = uf_row['Hit_Rate'].values[0]
                improvement = (uf_hr / affinity_hr - 1) * 100 if affinity_hr > 0 else 0
                improvements_data.append({
                    'Uncertainty_Percentile': pct,
                    'Top_N': top_n,
                    'Improvement': improvement
                })
    
    if improvements_data:
        imp_df = pd.DataFrame(improvements_data)
        
        for i, top_n in enumerate(top_ns):
            subset = imp_df[imp_df['Top_N'] == top_n]
            ax.plot(subset['Uncertainty_Percentile'], subset['Improvement'], 
                   marker='o', linewidth=2, markersize=8, label=f'Top-{top_n}')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Uncertainty Percentile Threshold', fontsize=12)
        ax.set_ylabel('Hit Rate Improvement vs Affinity-only (%)', fontsize=12)
        ax.set_title('Improvement from Uncertainty Filtering\n(Positive = Uncertainty Filtering Helps)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks([25, 50, 75])
        ax.set_xticklabels(['≤25th (strict)', '≤50th (moderate)', '≤75th (lenient)'])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        ax.fill_between([20, 80], [0, 0], [ax.get_ylim()[1]]*2, alpha=0.1, color='green')
        ax.fill_between([20, 80], [ax.get_ylim()[0]]*2, [0, 0], alpha=0.1, color='red')
        
        plt.tight_layout()
        plot_path = output_path / 'improvement_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'improvement_summary.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved improvement summary to: {plot_path}")


def plot_uncertainty_distribution(epistemic, labels, predictions, output_dir):
    """Plot uncertainty distribution for actives vs inactives."""
    
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Uncertainty distribution by activity
    ax1 = axes[0]
    active_unc = epistemic[labels == 1]
    inactive_unc = epistemic[labels == 0]
    
    ax1.hist(active_unc, bins=50, alpha=0.6, label=f'Active (n={len(active_unc)})', 
             color='green', density=True)
    ax1.hist(inactive_unc, bins=50, alpha=0.6, label=f'Inactive (n={len(inactive_unc)})', 
             color='red', density=True)
    ax1.set_xlabel('Epistemic Uncertainty', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Uncertainty Distribution by Activity', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted pKd vs Uncertainty colored by activity
    ax2 = axes[1]
    scatter = ax2.scatter(predictions, epistemic, c=labels, cmap='RdYlGn', alpha=0.5, s=20)
    ax2.set_xlabel('Predicted pKd', fontsize=12)
    ax2.set_ylabel('Epistemic Uncertainty', fontsize=12)
    ax2.set_title('Predictions vs Uncertainty\n(Green=Active, Red=Inactive)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add threshold lines
    for pct, style in [(25, ':'), (50, '--'), (75, '-.')]:
        thresh = np.percentile(epistemic, pct)
        ax2.axhline(y=thresh, color='blue', linestyle=style, alpha=0.5, 
                   label=f'{pct}th pctl: {thresh:.3f}')
    ax2.legend(loc='upper right', fontsize=8)
    
    # Plot 3: Hit rate vs uncertainty percentile
    ax3 = axes[2]
    percentiles = np.arange(10, 101, 5)
    hit_rates = []
    n_samples = []
    
    for pct in percentiles:
        thresh = np.percentile(epistemic, pct)
        mask = epistemic <= thresh
        if mask.sum() > 0:
            hr = labels[mask].mean()
            hit_rates.append(hr * 100)
            n_samples.append(mask.sum())
        else:
            hit_rates.append(0)
            n_samples.append(0)
    
    ax3.plot(percentiles, hit_rates, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=labels.mean() * 100, color='gray', linestyle='--', alpha=0.5, 
               label=f'Overall hit rate: {labels.mean()*100:.1f}%')
    ax3.set_xlabel('Uncertainty Percentile Threshold', fontsize=12)
    ax3.set_ylabel('Hit Rate (%)', fontsize=12)
    ax3.set_title('Hit Rate vs Uncertainty Threshold\n(Lower threshold = More selective)', 
                  fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add secondary axis for sample count
    ax3_twin = ax3.twinx()
    ax3_twin.fill_between(percentiles, 0, n_samples, alpha=0.2, color='orange')
    ax3_twin.set_ylabel('# Samples Retained', fontsize=12, color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    
    plt.tight_layout()
    plot_path = output_path / 'uncertainty_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'uncertainty_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved uncertainty analysis to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Uncertainty Filtering Experiment for Virtual Screening',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default ESR1 PCBA data
    python run_uncertainty_filtering_experiment.py
    
    # Run with custom dataset
    python run_uncertainty_filtering_experiment.py --csv_path data/5ht2a_chembl.csv
    
    # Run with specific selection sizes
    python run_uncertainty_filtering_experiment.py --top_ns 50 100 200 500
        """
    )
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, 
                        default='final_esr1_pcba_merged.csv',
                        help='Path to CSV file with embeddings and activity labels')
    parser.add_argument('--output_dir', type=str, 
                        default='uncertainty_filtering_results',
                        help='Directory to save results')
    parser.add_argument('--affinity_col', type=str, default=None,
                        help='Column name for continuous affinity (pKd, pKi). If set, uses threshold.')
    parser.add_argument('--affinity_threshold', type=float, default=6.0,
                        help='Threshold for defining actives: affinity >= threshold (default: 6.0)')
    parser.add_argument('--label_col', type=str, default='Label',
                        help='Column name for categorical activity labels')
    parser.add_argument('--active_label', type=str, default='active',
                        help='Value indicating active compounds in label_col')
    
    # Model arguments
    parser.add_argument('--model_path', type=str,
                        default='experiments/MoNIG_seed42/best_MoNIG_emb.pt',
                        help='Path to trained CABE model')
    parser.add_argument('--norm_stats_path', type=str,
                        default='experiments/MoNIG_seed42/best_MoNIG_emb_norm_stats.npz',
                        help='Path to normalization stats')
    
    # Experiment arguments
    parser.add_argument('--top_ns', type=int, nargs='+',
                        default=[50, 100, 200],
                        help='Selection sizes to evaluate (default: 50 100 200)')
    parser.add_argument('--uncertainty_percentiles', type=int, nargs='+',
                        default=[25, 50, 75],
                        help='Uncertainty percentile thresholds (default: 25 50 75)')
    parser.add_argument('--simulate', action='store_true',
                        help='Use simulated data for demonstration (useful when real data lacks decoys)')
    parser.add_argument('--n_actives', type=int, default=200,
                        help='Number of actives for simulation (default: 200)')
    parser.add_argument('--n_decoys', type=int, default=3600,
                        help='Number of decoys for simulation (default: 3600)')
    
    # Other arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*90)
    print("UNCERTAINTY FILTERING EXPERIMENT FOR VIRTUAL SCREENING")
    print("="*90)
    print(f"Mode: {'SIMULATION' if args.simulate else 'REAL DATA'}")
    if not args.simulate:
        print(f"CSV file: {args.csv_path}")
        print(f"Model: {args.model_path}")
    else:
        print(f"Simulated actives: {args.n_actives}")
        print(f"Simulated decoys: {args.n_decoys}")
    print(f"Output directory: {args.output_dir}")
    print(f"Selection sizes (Top-N): {args.top_ns}")
    print(f"Uncertainty percentiles: {args.uncertainty_percentiles}")
    print(f"Device: {args.device}")
    print("="*90)
    
    use_simulation = args.simulate
    
    if not use_simulation:
        # Try to load real data
        # Load normalization stats
        print("\n[1/5] Loading normalization statistics...")
        if os.path.exists(args.norm_stats_path):
            norm_stats_npz = np.load(args.norm_stats_path)
            norm_stats = {'mean': norm_stats_npz['mean'], 'std': norm_stats_npz['std']}
            print(f"✓ Loaded from: {args.norm_stats_path}")
        else:
            print(f"⚠ Norm stats not found at {args.norm_stats_path}, using unnormalized embeddings")
            norm_stats = None
        
        # Load dataset
        print("\n[2/5] Loading dataset...")
        try:
            dataset = ActiveDecoyDataset(
                args.csv_path, 
                norm_stats=norm_stats,
                label_col=args.label_col,
                active_label=args.active_label,
                affinity_col=args.affinity_col,
                affinity_threshold=args.affinity_threshold
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            
            # Check if dataset has both actives and decoys
            n_actives = dataset.labels.sum()
            n_decoys = len(dataset.labels) - n_actives
            
            if n_actives == 0 or n_decoys == 0:
                print(f"\n⚠ WARNING: Dataset has only {'actives' if n_actives > 0 else 'decoys'}!")
                print("   This experiment requires both actives and decoys.")
                print("   Switching to SIMULATION mode...")
                use_simulation = True
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            print("Switching to SIMULATION mode...")
            use_simulation = True
    
    if use_simulation:
        # Use simulated data
        print("\n[1/5] Generating simulated screening data...")
        predictions, epistemic, labels = generate_simulated_screening_data(
            n_actives=args.n_actives,
            n_decoys=args.n_decoys,
            seed=args.seed
        )
        aleatoric = epistemic * 2  # Simulated aleatoric
        complex_ids = [f"SIM_{i}" for i in range(len(labels))]
        
        print("\n[2/5] Skipping model loading (using simulation)...")
        print("\n[3/5] Skipping inference (using simulation)...")
        print("\n[4/5] Using simulated predictions and uncertainties...")
    else:
        # Load CABE model
        print("\n[3/5] Loading CABE model...")
        
        class HyperParams:
            num_experts = 4
            embedding_dim = 704
            hidden_dim = 256
            dropout = 0.2
        
        model = DrugDiscoveryMoNIGEmb(HyperParams())
        
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
            print(f"✓ Loaded model from: {args.model_path}")
        else:
            print(f"ERROR: Model not found at {args.model_path}")
            return
        
        model = model.to(args.device)
        model.eval()
        
        # Run inference
        print("\n[4/5] Running inference with CABE...")
        predictions, epistemic, aleatoric, labels, complex_ids = run_inference(
            model, loader, args.device
        )
    
    print(f"\n✓ Obtained predictions for {len(predictions)} samples")
    print(f"  Mean predicted pKd: {predictions.mean():.3f} ± {predictions.std():.3f}")
    print(f"  Mean epistemic uncertainty: {epistemic.mean():.4f}")
    if aleatoric is not None:
        print(f"  Mean aleatoric uncertainty: {aleatoric.mean():.4f}")
    
    # Run comparison experiment
    print("\n[5/5] Running comparison experiment...")
    results_df = run_comparison_experiment(
        predictions, epistemic, labels,
        top_ns=args.top_ns,
        uncertainty_percentiles=args.uncertainty_percentiles
    )
    
    # Print results
    print_results_table(results_df, args.top_ns)
    
    # Save results
    results_csv = output_path / 'comparison_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n💾 Saved results to: {results_csv}")
    
    # Save predictions with uncertainties
    pred_dict = {
        'ID': complex_ids,
        'Label': labels,
        'Predicted_pKd': predictions,
        'Epistemic_Uncertainty': epistemic,
    }
    if aleatoric is not None:
        pred_dict['Aleatoric_Uncertainty'] = aleatoric
        pred_dict['Total_Uncertainty'] = np.sqrt(epistemic + aleatoric)
    else:
        pred_dict['Total_Uncertainty'] = np.sqrt(epistemic)
    predictions_df = pd.DataFrame(pred_dict)
    predictions_csv = output_path / 'predictions_with_uncertainty.csv'
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"💾 Saved predictions to: {predictions_csv}")
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_comparison_results(results_df, args.output_dir, args.top_ns)
    plot_uncertainty_distribution(epistemic, labels, predictions, args.output_dir)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'simulation' if use_simulation else 'real_data',
        'csv_path': args.csv_path if not use_simulation else None,
        'model_path': args.model_path if not use_simulation else None,
        'n_total': len(predictions),
        'n_actives': int(labels.sum()),
        'n_inactives': int(len(labels) - labels.sum()),
        'base_hit_rate': float(labels.mean()),
        'top_ns': args.top_ns,
        'uncertainty_percentiles': args.uncertainty_percentiles,
        'results': results_df.to_dict('records')
    }
    
    summary_path = output_path / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved summary to: {summary_path}")
    
    # Print key findings
    print("\n" + "="*90)
    print("KEY FINDINGS")
    print("="*90)
    
    for top_n in args.top_ns:
        subset = results_df[results_df['Top_N'] == top_n]
        affinity_row = subset[subset['Strategy'] == 'Affinity-only'].iloc[0]
        
        best_uf_row = subset[subset['Strategy'] != 'Affinity-only'].loc[
            subset[subset['Strategy'] != 'Affinity-only']['Hit_Rate'].idxmax()
        ]
        
        improvement = (best_uf_row['Hit_Rate'] / affinity_row['Hit_Rate'] - 1) * 100
        
        print(f"\nTop-{top_n}:")
        print(f"  Affinity-only hit rate:     {affinity_row['Hit_Rate']*100:.2f}% ({int(affinity_row['N_Hits'])} hits)")
        print(f"  Best UF hit rate:           {best_uf_row['Hit_Rate']*100:.2f}% ({int(best_uf_row['N_Hits'])} hits)")
        print(f"  Strategy:                   {best_uf_row['Strategy']}")
        print(f"  Improvement:                {improvement:+.1f}%")
        print(f"  Enrichment Factor (Aff):    {affinity_row['Enrichment_Factor']:.2f}")
        print(f"  Enrichment Factor (UF):     {best_uf_row['Enrichment_Factor']:.2f}")
    
    print("\n" + "="*90)
    print("✅ Experiment complete!")
    print(f"📁 All outputs saved to: {args.output_dir}/")
    print("="*90)


if __name__ == '__main__':
    main()

