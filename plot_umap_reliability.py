#!/usr/bin/env python
"""
UMAP/t-SNE visualization of embeddings colored by CABE reliability scores.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed. Using t-SNE only.")

from src.drug_models_emb import DrugDiscoveryMoNIGEmb


def get_reliability_scores(model, embeddings, device='cpu'):
    """Extract reliability scores from the CABE model."""
    model.eval()
    with torch.no_grad():
        embeddings_tensor = torch.FloatTensor(embeddings).to(device)
        reliability_scores = model.reliability_net(embeddings_tensor)
    return reliability_scores.cpu().numpy()


def load_data_and_model(model_dir, csv_path):
    """Load model and data directly from CSV."""
    model_path = Path(model_dir) / 'best_MoNIG_emb.pt'
    norm_stats_path = Path(model_dir) / 'best_MoNIG_emb_norm_stats.npz'
    
    # Load norm stats
    norm_stats = np.load(norm_stats_path)
    emb_mean = norm_stats['mean']
    emb_std = norm_stats['std']
    
    # Load data directly from CSV
    df = pd.read_csv(csv_path)
    
    # Extract embedding columns
    emb_cols = [col for col in df.columns if col.startswith('Emb_')]
    embeddings_raw = df[emb_cols].values.astype(np.float32)
    
    # Normalize embeddings
    embeddings = (embeddings_raw - emb_mean) / (emb_std + 1e-8)
    
    # Load model - detect number of experts from state dict
    state_dict = torch.load(model_path, map_location='cpu')
    # Check reliability_net output size to determine num_experts
    num_experts = state_dict['reliability_net.3.weight'].shape[0]
    
    class HyperParams:
        pass
    hp = HyperParams()
    hp.num_experts = num_experts
    hp.embedding_dim = 704
    hp.hidden_dim = 256
    hp.dropout = 0.2
    
    model = DrugDiscoveryMoNIGEmb(hp)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, embeddings, df, num_experts


def plot_umap_tsne_reliability(output_dir='risk_coverage_comparison', method='both'):
    """Create UMAP/t-SNE visualization colored by reliability."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and data...")
    model, embeddings, df, num_experts = load_data_and_model(
        model_dir='experiments/MoNIG_seed42',
        csv_path='pdbbind_descriptors_with_experts_and_binding.csv'
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of experts: {num_experts}")
    
    # Get reliability scores
    print("Computing reliability scores...")
    reliability = get_reliability_scores(model, embeddings)
    print(f"Reliability shape: {reliability.shape}")
    
    # Average reliability across experts for coloring
    mean_reliability = reliability.mean(axis=1)
    
    # Subsample for faster computation if too many points
    n_samples = len(embeddings)
    max_samples = 5000
    if n_samples > max_samples:
        print(f"Subsampling from {n_samples} to {max_samples} points...")
        idx = np.random.choice(n_samples, max_samples, replace=False)
        embeddings_sub = embeddings[idx]
        mean_reliability_sub = mean_reliability[idx]
        reliability_sub = reliability[idx]
    else:
        embeddings_sub = embeddings
        mean_reliability_sub = mean_reliability
        reliability_sub = reliability
    
    # Color palette
    cmap = 'viridis'
    
    plots_to_make = []
    if method == 'both':
        if HAS_UMAP:
            plots_to_make = ['umap', 'tsne']
        else:
            plots_to_make = ['tsne']
    else:
        plots_to_make = [method]
    
    for plot_method in plots_to_make:
        print(f"\nComputing {plot_method.upper()}...")
        
        if plot_method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_sub)
        else:  # t-SNE
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_sub)
        
        # Create figure with subplots for each expert + mean
        n_experts = reliability_sub.shape[1]
        fig, axes = plt.subplots(1, n_experts + 1, figsize=(5 * (n_experts + 1), 5))
        
        # Expert names based on training data order
        if n_experts == 4:
            expert_names = ['GNINA', 'BIND', 'FlowDock', 'DynamicBind']
        else:
            expert_names = ['GNINA', 'FlowDock', 'DynamicBind']
        
        # Plot mean reliability
        sc = axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                            c=mean_reliability_sub, cmap=cmap, s=10, alpha=0.7)
        axes[0].set_title('Mean Reliability', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'{plot_method.upper()} 1', fontsize=12)
        axes[0].set_ylabel(f'{plot_method.upper()} 2', fontsize=12)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.colorbar(sc, ax=axes[0], label='Reliability')
        
        # Plot per-expert reliability
        for i in range(n_experts):
            sc = axes[i+1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                  c=reliability_sub[:, i], cmap=cmap, s=10, alpha=0.7)
            axes[i+1].set_title(f'{expert_names[i]} Reliability', fontsize=14, fontweight='bold')
            axes[i+1].set_xlabel(f'{plot_method.upper()} 1', fontsize=12)
            axes[i+1].set_ylabel(f'{plot_method.upper()} 2', fontsize=12)
            axes[i+1].set_xticks([])
            axes[i+1].set_yticks([])
            plt.colorbar(sc, ax=axes[i+1], label='Reliability')
        
        plt.tight_layout()
        
        # Save
        png_path = output_path / f'{plot_method}_reliability.png'
        pdf_path = output_path / f'{plot_method}_reliability.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved {plot_method.upper()} visualization to:")
        print(f"   {png_path}")
        print(f"   {pdf_path}")
    
    # Also create a single clean plot with just mean reliability
    print("\nCreating single mean reliability plot...")
    
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_sub)
        method_name = 'UMAP'
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_sub)
        method_name = 't-SNE'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                   c=mean_reliability_sub, cmap='viridis', s=15, alpha=0.7)
    ax.set_xlabel(f'{method_name} 1', fontsize=16)
    ax.set_ylabel(f'{method_name} 2', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Mean Reliability', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    png_path = output_path / 'embedding_reliability.png'
    pdf_path = output_path / 'embedding_reliability.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved single reliability plot to:")
    print(f"   {png_path}")
    print(f"   {pdf_path}")
    
    # Print statistics
    print("\n📊 Reliability Statistics:")
    print(f"   Mean: {mean_reliability.mean():.4f}")
    print(f"   Std:  {mean_reliability.std():.4f}")
    print(f"   Min:  {mean_reliability.min():.4f}")
    print(f"   Max:  {mean_reliability.max():.4f}")
    
    # Expert names based on number of experts
    if num_experts == 4:
        expert_names_stats = ['GNINA', 'BIND', 'FlowDock', 'DynamicBind']
    else:
        expert_names_stats = ['GNINA', 'FlowDock', 'DynamicBind']
    
    for i, name in enumerate(expert_names_stats):
        print(f"   {name}: {reliability[:, i].mean():.4f} ± {reliability[:, i].std():.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UMAP/t-SNE colored by reliability')
    parser.add_argument('--output_dir', type=str, default='risk_coverage_comparison',
                        help='Output directory for plots')
    parser.add_argument('--method', type=str, default='both', choices=['umap', 'tsne', 'both'],
                        help='Dimensionality reduction method')
    args = parser.parse_args()
    
    plot_umap_tsne_reliability(args.output_dir, args.method)

