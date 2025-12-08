"""
Multi-Seed Uncertainty Analysis for MoNIG
Analyzes prediction accuracy across multiple seeds for experts and MoNIG
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob


def analyze_multiseed_prediction_accuracy(experiment_dir, model_name='MoNIG', seeds=None):
    """
    Analyze prediction accuracy across multiple seeds
    
    Args:
        experiment_dir: Path to experiments directory
        model_name: Name of the model (default: 'MoNIG')
        seeds: List of seeds to analyze (default: auto-detect)
    
    Returns:
        DataFrame with MAE statistics across seeds
    """
    experiment_dir = Path(experiment_dir)
    
    # Auto-detect seeds if not provided
    if seeds is None:
        seed_dirs = sorted(experiment_dir.glob(f'{model_name}_seed*'))
        seeds = [int(d.name.split('seed')[1]) for d in seed_dirs]
        print(f"Auto-detected seeds: {seeds}")
    
    # Collect MAE data for each seed
    all_seed_data = []
    
    for seed in seeds:
        csv_path = experiment_dir / f'{model_name}_seed{seed}' / 'test_inference_results.csv'
        
        if not csv_path.exists():
            print(f"⚠️  Warning: {csv_path} not found, skipping seed {seed}")
            continue
        
        print(f"Loading seed {seed}: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Detect number of experts
        expert_cols = [col for col in df.columns if col.startswith('Expert') and '_Prediction' in col]
        num_experts = len(expert_cols)
        
        # Calculate MAE for each expert
        seed_maes = {'seed': seed}
        
        for j in range(num_experts):
            expert_num = j + 1
            expert_mae = np.abs(df[f'Expert{expert_num}_Prediction'] - df['True_Affinity']).mean()
            seed_maes[f'Expert{expert_num}'] = expert_mae
        
        # Calculate MAE for MoNIG
        monig_mae = np.abs(df['MoNIG_Prediction'] - df['True_Affinity']).mean()
        seed_maes['MoNIG'] = monig_mae
        
        all_seed_data.append(seed_maes)
        
        expert_maes_str = [f"{seed_maes[f'Expert{j+1}']:.4f}" for j in range(num_experts)]
        print(f"  Seed {seed}: Expert MAEs = {expert_maes_str}, MoNIG MAE = {monig_mae:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_seed_data)
    
    # Calculate statistics
    stats_df = pd.DataFrame({
        'Model': [],
        'Mean_MAE': [],
        'Std_MAE': [],
        'Min_MAE': [],
        'Max_MAE': []
    })
    
    num_experts = len([col for col in results_df.columns if col.startswith('Expert')])
    
    for j in range(num_experts):
        expert_name = f'Expert{j+1}'
        stats_df = pd.concat([stats_df, pd.DataFrame({
            'Model': [expert_name],
            'Mean_MAE': [results_df[expert_name].mean()],
            'Std_MAE': [results_df[expert_name].std()],
            'Min_MAE': [results_df[expert_name].min()],
            'Max_MAE': [results_df[expert_name].max()]
        })], ignore_index=True)
    
    # Add MoNIG stats
    stats_df = pd.concat([stats_df, pd.DataFrame({
        'Model': ['MoNIG'],
        'Mean_MAE': [results_df['MoNIG'].mean()],
        'Std_MAE': [results_df['MoNIG'].std()],
        'Min_MAE': [results_df['MoNIG'].min()],
        'Max_MAE': [results_df['MoNIG'].max()]
    })], ignore_index=True)
    
    return results_df, stats_df, num_experts


def plot_multiseed_accuracy_comparison(stats_df, num_experts, output_path=None, seeds=None):
    """
    Create bar plot comparing prediction accuracy with error bars
    
    Args:
        stats_df: DataFrame with MAE statistics
        num_experts: Number of experts
        output_path: Path to save the figure
        seeds: List of seeds used (for title)
    """
    # Expert names mapping
    expert_names_map = {
        'Expert1': 'GNINA',
        'Expert2': 'BIND',
        'Expert3': 'flowdock',
        'Expert4': 'DynamicBind'
    }
    
    # Prepare data
    models = stats_df['Model'].tolist()
    mean_maes = stats_df['Mean_MAE'].tolist()
    std_maes = stats_df['Std_MAE'].tolist()
    
    # Create labels with expert names
    labels = []
    for model in models:
        if model in expert_names_map:
            labels.append(f'{model}\n({expert_names_map[model]})')
        else:
            labels.append(model)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors
    colors = ['coral', 'skyblue', 'lightgreen', 'orange', '#81c784']
    
    # Create bar plot with error bars
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, mean_maes, yerr=std_maes, 
                   color=colors[:len(models)],
                   alpha=0.8,
                   capsize=10,
                   error_kw={'linewidth': 2, 'elinewidth': 2},
                   edgecolor='black',
                   linewidth=1.5)
    
    # Customize plot
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    
    # Title
    if seeds:
        title = f'Prediction Accuracy Comparison\n(Seeds: {min(seeds)}-{max(seeds)}, n={len(seeds)})'
    else:
        title = 'Prediction Accuracy Comparison (Multi-Seed)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, mean_mae, std_mae) in enumerate(zip(bars, mean_maes, std_maes)):
        height = bar.get_height()
        # Add mean value on top
        ax.text(bar.get_x() + bar.get_width()/2., height + std_mae + 0.01,
                f'{mean_mae:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add std value below the bar
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'±{std_mae:.4f}',
                ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Highlight the best model (lowest MAE)
    best_idx = np.argmin(mean_maes)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)
    
    # Add legend
    ax.text(0.02, 0.98, 
            f'✓ Best: {models[best_idx]} (MAE = {mean_maes[best_idx]:.4f} ± {std_maes[best_idx]:.4f})',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontsize=11,
            fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved plot: {output_path}")
    
    return fig, ax


def print_summary_table(stats_df, num_experts):
    """Print formatted summary table"""
    print("\n" + "="*80)
    print("MULTI-SEED PREDICTION ACCURACY SUMMARY")
    print("="*80)
    
    # Expert names mapping
    expert_names_map = {
        'Expert1': 'GNINA',
        'Expert2': 'BIND',
        'Expert3': 'flowdock',
        'Expert4': 'DynamicBind'
    }
    
    print(f"\n{'Model':<20} {'Expert Name':<15} {'Mean MAE':<12} {'Std MAE':<12} {'Min MAE':<12} {'Max MAE':<12}")
    print("-" * 80)
    
    for _, row in stats_df.iterrows():
        model = row['Model']
        expert_name = expert_names_map.get(model, '-')
        print(f"{model:<20} {expert_name:<15} {row['Mean_MAE']:<12.4f} {row['Std_MAE']:<12.4f} {row['Min_MAE']:<12.4f} {row['Max_MAE']:<12.4f}")
    
    print("="*80)
    
    # Find best model
    best_idx = stats_df['Mean_MAE'].idxmin()
    best_model = stats_df.loc[best_idx, 'Model']
    best_mae = stats_df.loc[best_idx, 'Mean_MAE']
    best_std = stats_df.loc[best_idx, 'Std_MAE']
    
    print(f"\n✅ Best Model: {best_model} with MAE = {best_mae:.4f} ± {best_std:.4f}")
    
    # Calculate improvement over each expert
    print(f"\n📊 MoNIG Performance vs Experts:")
    monig_mae = stats_df[stats_df['Model'] == 'MoNIG']['Mean_MAE'].values[0]
    monig_std = stats_df[stats_df['Model'] == 'MoNIG']['Std_MAE'].values[0]
    
    for j in range(num_experts):
        expert_name = f'Expert{j+1}'
        expert_row = stats_df[stats_df['Model'] == expert_name]
        if len(expert_row) > 0:
            expert_mae = expert_row['Mean_MAE'].values[0]
            improvement = ((expert_mae - monig_mae) / expert_mae) * 100
            print(f"  vs {expert_name} ({expert_names_map.get(expert_name, '')}): {improvement:+.2f}%")
    
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Seed Uncertainty Analysis for Prediction Accuracy')
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Path to experiments directory (default: experiments)')
    parser.add_argument('--model', type=str, default='MoNIG',
                       help='Model name to analyze (default: MoNIG)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Seeds to analyze (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: auto-generate)')
    
    args = parser.parse_args()
    
    # Run analysis
    print(f"\n🔍 Analyzing multi-seed prediction accuracy for {args.model}...")
    print(f"Experiment directory: {args.experiment_dir}")
    
    results_df, stats_df, num_experts = analyze_multiseed_prediction_accuracy(
        args.experiment_dir, 
        model_name=args.model,
        seeds=args.seeds
    )
    
    # Print summary
    print_summary_table(stats_df, num_experts)
    
    # Save detailed results
    results_csv_path = Path(args.experiment_dir) / f'{args.model.lower()}_multiseed_accuracy_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n💾 Saved detailed results: {results_csv_path}")
    
    stats_csv_path = Path(args.experiment_dir) / f'{args.model.lower()}_multiseed_accuracy_stats.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"💾 Saved statistics: {stats_csv_path}")
    
    # Generate plot
    if args.output is None:
        output_path = Path(args.experiment_dir) / f'{args.model.lower()}_multiseed_accuracy_comparison.png'
    else:
        output_path = Path(args.output)
    
    seeds_used = results_df['seed'].tolist()
    plot_multiseed_accuracy_comparison(stats_df, num_experts, output_path, seeds=seeds_used)
    
    print(f"\n✅ Multi-seed analysis complete!")

