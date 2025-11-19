#!/usr/bin/env python
"""
Main entry point for Drug Discovery MoNIG
Supports training, inference, and analysis modes
"""
import argparse
import sys
from pathlib import Path
import torch

from src.train_drug_discovery_emb import main as train_main
from src.inference_drug_discovery import main as inference_main
from src.analyze_uncertainty import analyze_uncertainty, visualize_uncertainty


def main():
    parser = argparse.ArgumentParser(
        description='Drug Discovery with MoNIG - Training, Inference, and Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --csv_path pdbbind_descriptors_with_experts_and_binding.csv --model_type MoNIG
  
  # Run inference
  python main.py infer --model_path saved_models/best_MoNIG_emb.pt --split test
  
  # Analyze uncertainty
  python main.py analyze --csv test_inference_results.csv --output_prefix test_uncertainty
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train, infer, or analyze')
    subparsers.required = True
    
    # ===== Training subparser =====
    train_parser = subparsers.add_parser('train', help='Train a drug discovery model')
    
    # Data
    train_parser.add_argument('--csv_path', type=str, 
                             default='pdbbind_descriptors_with_experts_and_binding.csv',
                             help='Path to CSV file')
    train_parser.add_argument('--batch_size', type=int, default=64,
                             help='Batch size')
    
    # Model
    train_parser.add_argument('--model_type', type=str, default='MoNIG',
                             choices=['MoNIG', 'NIG', 'Gaussian', 'Baseline'],
                             help='Model type')
    train_parser.add_argument('--hidden_dim', type=int, default=256,
                             help='Hidden dimension')
    train_parser.add_argument('--dropout', type=float, default=0.2,
                             help='Dropout rate')
    
    # Expert selection
    train_parser.add_argument('--expert1_only', action='store_true',
                             help='Use only Expert 1 (GNINA)')
    train_parser.add_argument('--expert2_only', action='store_true',
                             help='Use only Expert 2 (BIND)')
    train_parser.add_argument('--expert3_only', action='store_true',
                             help='Use only Expert 3 (flowdock)')
    
    # Training
    train_parser.add_argument('--epochs', type=int, default=150,
                             help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=5e-4,
                             help='Learning rate')
    train_parser.add_argument('--risk_weight', type=float, default=0.005,
                             help='Risk regularization weight')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # Device
    train_parser.add_argument('--device', type=str, 
                             default='cuda' if torch.cuda.is_available() else 'cpu',
                             help='Device to use (cuda/cpu)')
    
    # ===== Inference subparser =====
    infer_parser = subparsers.add_parser('infer', help='Run inference with trained model')
    
    # Model
    infer_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to trained model checkpoint')
    infer_parser.add_argument('--csv_path', type=str, 
                             default='pdbbind_descriptors_with_experts_and_binding.csv',
                             help='Path to input CSV file')
    infer_parser.add_argument('--output_path', type=str, default='test_inference_results.csv',
                             help='Path to save output CSV')
    infer_parser.add_argument('--norm_stats_path', type=str, default=None,
                             help='Path to normalization stats (.npz). Auto-detected if not provided')
    infer_parser.add_argument('--calibrator_path', type=str, default=None,
                             help='Path to isotonic calibrator (.pkl). Auto-detected if not provided')
    
    # Data split
    infer_parser.add_argument('--split', type=str, default='test',
                             choices=['train', 'valid', 'test', 'casf2016'],
                             help='Which split to run inference on (test=internal test set, casf2016=CASF 2016 benchmark)')
    infer_parser.add_argument('--batch_size', type=int, default=64,
                             help='Batch size')
    
    # Model config (must match training)
    infer_parser.add_argument('--hidden_dim', type=int, default=256,
                             help='Hidden dimension (must match training)')
    infer_parser.add_argument('--dropout', type=float, default=0.2,
                             help='Dropout rate (must match training)')
    
    # Device
    infer_parser.add_argument('--device', type=str, 
                             default='cuda' if torch.cuda.is_available() else 'cpu',
                             help='Device to use (cuda/cpu)')
    
    # ===== Analysis subparser =====
    analyze_parser = subparsers.add_parser('analyze', help='Analyze uncertainty from inference results')
    
    analyze_parser.add_argument('--csv', type=str, default='test_inference_results.csv',
                               help='Path to inference results CSV (default: test_inference_results.csv)')
    analyze_parser.add_argument('--output_prefix', type=str, default='test_uncertainty',
                               help='Prefix for output files (default: test_uncertainty)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Build argument list for training
        train_args = [
            'train_drug_discovery_emb.py',
            '--csv_path', args.csv_path,
            '--batch_size', str(args.batch_size),
            '--model_type', args.model_type,
            '--hidden_dim', str(args.hidden_dim),
            '--dropout', str(args.dropout),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--risk_weight', str(args.risk_weight),
            '--seed', str(args.seed),
            '--device', args.device,
        ]
        if args.expert1_only:
            train_args.append('--expert1_only')
        if args.expert2_only:
            train_args.append('--expert2_only')
        if args.expert3_only:
            train_args.append('--expert3_only')
        
        # Temporarily replace sys.argv and call train_main
        old_argv = sys.argv
        sys.argv = train_args
        try:
            train_main()
        finally:
            sys.argv = old_argv
            
    elif args.mode == 'infer':
        # Build argument list for inference
        infer_args = [
            'inference_drug_discovery.py',
            '--model_path', args.model_path,
            '--csv_path', args.csv_path,
            '--output_path', args.output_path,
            '--split', args.split,
            '--batch_size', str(args.batch_size),
            '--hidden_dim', str(args.hidden_dim),
            '--dropout', str(args.dropout),
            '--device', args.device,
        ]
        if args.norm_stats_path:
            infer_args.extend(['--norm_stats_path', args.norm_stats_path])
        if args.calibrator_path:
            infer_args.extend(['--calibrator_path', args.calibrator_path])
        
        # Temporarily replace sys.argv and call inference_main
        old_argv = sys.argv
        sys.argv = infer_args
        try:
            inference_main()
        finally:
            sys.argv = old_argv
            
    elif args.mode == 'analyze':
        # Run uncertainty analysis directly
        df, _ = analyze_uncertainty(args.csv)
        
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


if __name__ == '__main__':
    main()
