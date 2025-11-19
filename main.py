#!/usr/bin/env python
"""
Main orchestrator for CABE MoNIG
Handles:
  • Training
  • Inference
  • Uncertainty Analysis
"""

import argparse
import sys
from pathlib import Path
import torch

from src.train_drug_discovery_emb import run_training
from src.inference_drug_discovery import run_inference
from src.analyze_uncertainty import analyze_uncertainty, visualize_uncertainty


def main():
    parser = argparse.ArgumentParser(
        description="CABE MoNIG - Train, Infer, Analyze",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ===============================
    # TRAIN
    # ===============================
    p_train = subparsers.add_parser("train", help="Train a model")

    # Data
    p_train.add_argument("--csv_path", type=str,
                         default="pdbbind_descriptors_with_experts_and_binding.csv")
    p_train.add_argument("--batch_size", type=int, default=64)

    # Model
    p_train.add_argument("--model_type", type=str, default="MoNIG",
                         choices=["MoNIG", "NIG", "Gaussian", "Baseline"])
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--dropout", type=float, default=0.2)

    # Expert masking
    p_train.add_argument("--expert1_only", action="store_true")
    p_train.add_argument("--expert2_only", action="store_true")
    p_train.add_argument("--expert3_only", action="store_true")

    # Training config
    p_train.add_argument("--epochs", type=int, default=150)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--risk_weight", type=float, default=0.005)
    p_train.add_argument("--evidence_reg_weight", type=float, default=0.01)
    p_train.add_argument("--precision_temperature", type=float, default=1.0)
    p_train.add_argument("--balance_factor", type=float, default=0.3,
                         help="Expert balance factor (0.0-1.0, higher = more balanced, prevents dominance)")
    p_train.add_argument("--seed", type=int, default=42)

    p_train.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")

    # ===============================
    # INFERENCE
    # ===============================
    p_infer = subparsers.add_parser("infer", help="Run inference")

    p_infer.add_argument("--model_path", type=str, required=True)
    p_infer.add_argument("--csv_path", type=str,
                         default="pdbbind_descriptors_with_experts_and_binding.csv")
    p_infer.add_argument("--output_path", type=str,
                         default="test_inference_results.csv")
    p_infer.add_argument("--norm_stats_path", type=str, default=None)
    p_infer.add_argument("--calibrator_path", type=str, default=None)

    p_infer.add_argument("--split", type=str, default="test",
                         choices=["train", "valid", "test", "casf2016", "casf2013"])
    p_infer.add_argument("--batch_size", type=int, default=64)

    p_infer.add_argument("--hidden_dim", type=int, default=256)
    p_infer.add_argument("--dropout", type=float, default=0.2)
    p_infer.add_argument("--precision_temperature", type=float, default=1.0)
    p_infer.add_argument("--balance_factor", type=float, default=0.3,
                         help="Expert balance factor (0.0-1.0, must match training)")

    p_infer.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")

    # ===============================
    # ANALYZE
    # ===============================
    p_analyze = subparsers.add_parser("analyze", help="Analyze uncertainty outputs")
    p_analyze.add_argument("--csv", type=str, default="test_inference_results.csv")
    p_analyze.add_argument("--output_prefix", type=str, default="test_uncertainty")

    args = parser.parse_args()

    # ===============================
    # EXECUTION
    # ===============================
    if args.mode == "train":
        run_training(args)

    elif args.mode == "infer":
        run_inference(args)

    elif args.mode == "analyze":
        df, _ = analyze_uncertainty(args.csv)
        visualize_uncertainty(df, args.output_prefix)
        print("\nAnalysis complete.")

    else:
        raise RuntimeError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()