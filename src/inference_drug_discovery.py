import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
import uncertainty_toolbox as uct  # only used for calibration helpers

from src.drug_dataset_emb import DrugDiscoveryDatasetEmb
from src.drug_models_emb import (
    DrugDiscoveryMoNIGEmb,
    DrugDiscoveryNIGEmb,
    DrugDiscoveryGaussianEmb,
    DrugDiscoveryBaselineEmb,
)
from src.utils import moe_nig, moe_nig_balanced


def aggregate_nigs(nigs, precision_temperature: float = 1.0, balance_factor: float = 0.3):
    """
    Same aggregation as in training: fuse multiple NIGs with optional
    precision tempering (divide v and alpha by T >= 1) and balance factor.
    """
    if len(nigs) == 0:
        raise ValueError("Cannot aggregate empty NIG list")
    if len(nigs) == 1:
        mu, v, alpha, beta = nigs[0]
        v = v / precision_temperature
        alpha = alpha / precision_temperature
        return mu, v, alpha, beta

    # Apply precision temperature first
    nigs_temp = []
    for mu, v, alpha, beta in nigs:
        v_t = v / precision_temperature
        alpha_t = alpha / precision_temperature
        nigs_temp.append((mu, v_t, alpha_t, beta))
    
    # Use balanced aggregation if balance_factor > 0
    if balance_factor > 0.0 and len(nigs_temp) > 1:
        return moe_nig_balanced(nigs_temp, balance_factor=balance_factor)
    else:
        # Standard sequential aggregation
        mu_final, v_final, alpha_final, beta_final = nigs_temp[0]
        for mu, v, alpha, beta in nigs_temp[1:]:
            mu_final, v_final, alpha_final, beta_final = moe_nig(
                mu_final, v_final, alpha_final, beta_final, mu, v, alpha, beta
            )
        return mu_final, v_final, alpha_final, beta_final


def nig_uncertainty(v, alpha, beta):
    aleatoric = beta / (alpha - 1)
    epistemic = beta / (v * (alpha - 1))
    return epistemic, aleatoric


def apply_interval_calibration(y_pred, y_std, iso_model, coverage=0.682689):
    """Recalibrate total std using isotonic interval recalibration."""
    if iso_model is None:
        return y_std
    if y_pred.size == 0:
        return y_std
    z = stats.norm.ppf(0.5 + coverage / 2.0)
    bounds = uct.metrics_calibration.get_prediction_interval(
        y_pred.flatten(),
        y_std.flatten(),
        np.array([coverage]),
        recal_model=iso_model
    )
    width = bounds.upper - bounds.lower
    recalibrated_std = (width / (2.0 * z)).reshape(y_std.shape)
    return recalibrated_std


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Inference script for Drug Discovery MoNIG (Embeddings)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="pdbbind_descriptors_with_experts_and_binding.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="test_inference_results.csv",
        help="Path to save output CSV",
    )
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default=None,
        help="Path to normalization stats (.npz). Auto-detected if not provided",
    )
    parser.add_argument(
        "--calibrator_path",
        type=str,
        default=None,
        help="Path to isotonic calibrator (.pkl). Auto-detected if not provided",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "casf2016", "casf2013"],
        help="Dataset split to run inference on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension (must match training)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (must match training)",
    )
    parser.add_argument(
        "--precision_temperature",
        type=float,
        default=1.0,
        help="Precision temperature (>=1.0, must match training)",
    )
    parser.add_argument(
        "--balance_factor",
        type=float,
        default=0.3,
        help="Expert balance factor (0.0-1.0, must match training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    return parser


def run_inference(args):
    device = args.device

    # infer model type from filename prefix
    base = os.path.basename(args.model_path)
    if "MoNIG" in base:
        model_type = "MoNIG"
    elif "NIG" in base:
        model_type = "NIG"
    elif "Gaussian" in base:
        model_type = "Gaussian"
    else:
        model_type = "Baseline"

    # normalization / score stats
    if args.norm_stats_path is None:
        # try to auto-detect from same folder
        folder = os.path.dirname(args.model_path)
        guess = None
        for fname in os.listdir(folder):
            if fname.endswith("_emb_norm_stats.npz"):
                guess = os.path.join(folder, fname)
                break
        if guess is None:
            raise FileNotFoundError(
                "Could not auto-detect norm_stats .npz; please pass --norm_stats_path"
            )
        args.norm_stats_path = guess

    stats_npz = np.load(args.norm_stats_path)
    emb_mean = stats_npz["mean"]
    emb_std = stats_npz["std"]
    # New: expert score stats for z-scoring inside the model
    if "score_mean" in stats_npz and "score_std" in stats_npz:
        score_mean_full = stats_npz["score_mean"]
        score_std_full = stats_npz["score_std"]
    else:
        # backward-compat: fall back to zeros/ones (no effect)
        score_mean_full = np.zeros(3, dtype=np.float32)
        score_std_full = np.ones(3, dtype=np.float32)
    norm_stats = {"mean": emb_mean, "std": emb_std}

    # Check if split.json exists - if so, use it instead of loading CASF from CSV
    split_json_path = "split.json"
    use_split_json = os.path.isfile(split_json_path)
    
    casf2016_pdb_ids = None
    if not use_split_json:
        # Fallback to old method: load CASF from CSV (only needed for casf2016 split)
        if args.split == "casf2016":
            casf_path = "data/coreset_dirs.csv"
            if os.path.isfile(casf_path):
                with open(casf_path, "r") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if lines and ("directory" in lines[0].lower() or "pdb" in lines[0].lower()):
                    lines = lines[1:]
                casf2016_pdb_ids = lines
                print(f"Loaded {len(casf2016_pdb_ids)} CASF 2016 PDB IDs from {casf_path}")
            else:
                raise FileNotFoundError(
                    f"CASF 2016 split requested but {casf_path} not found and {split_json_path} not available"
                )

    print("=" * 50)
    print("Drug Discovery Inference (Embeddings)")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"CSV path: {args.csv_path}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Model type detected: {model_type}")
    print(f"Norm stats: {args.norm_stats_path}")
    if use_split_json:
        print(f"Using splits from: {split_json_path}")
    print("=" * 50)

    dataset = DrugDiscoveryDatasetEmb(
        args.csv_path,
        split=args.split,
        seed=42,
        normalization_stats=norm_stats,
        casf2016_pdb_ids=casf2016_pdb_ids,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    embedding_dim, num_experts_available = dataset.get_dim()

    print(f"\nDataset: {len(dataset)} samples")
    print(f"Embedding dim: {embedding_dim}, num_experts: {num_experts_available}")

    # expert indices: here we assume all experts were used at training time
    expert_indices = list(range(num_experts_available))
    if len(expert_indices) == len(score_mean_full):
        score_mean = score_mean_full
        score_std = score_std_full
    else:
        score_mean = score_mean_full[expert_indices]
        score_std = score_std_full[expert_indices]

    # build hyp params
    class Hyp:
        pass

    hyp = Hyp()
    hyp.num_experts = len(expert_indices)
    hyp.embedding_dim = embedding_dim
    hyp.hidden_dim = args.hidden_dim
    hyp.dropout = args.dropout
    hyp.score_mean = score_mean.tolist()
    hyp.score_std = score_std.tolist()
    # Expert normalization to prevent dominance (must match training)
    hyp.use_expert_normalization = True
    hyp.expert_normalization_strength = 0.8  # Very strong normalization: compress differences by 80%

    if model_type == "MoNIG":
        model = DrugDiscoveryMoNIGEmb(hyp)
    elif model_type == "NIG":
        model = DrugDiscoveryNIGEmb(hyp)
    elif model_type == "Gaussian":
        model = DrugDiscoveryGaussianEmb(hyp)
    else:
        model = DrugDiscoveryBaselineEmb(hyp)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("\nModel loaded, running inference...")

    calibrator = None
    if args.calibrator_path is None and model_type in ["MoNIG", "NIG"]:
        folder = os.path.dirname(args.model_path)
        guess = None
        for fname in os.listdir(folder):
            if fname.endswith("_emb_calibrator.pkl"):
                guess = os.path.join(folder, fname)
                break
        args.calibrator_path = guess

    if args.calibrator_path and os.path.isfile(args.calibrator_path):
        with open(args.calibrator_path, "rb") as f:
            data = pickle.load(f)
        calibrator = data.get("iso_model", None)
        print(f"Loaded isotonic calibrator from {args.calibrator_path}")
    else:
        if model_type in ["MoNIG", "NIG"]:
            print("Warning: no calibrator found; using raw NIG intervals")

    all_rows = []
    with torch.no_grad():
        for (expert_scores, embeddings), labels, complex_ids in loader:
            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)

            if model_type == "MoNIG":
                nigs = model(expert_scores, embeddings)
                
                # Extract per-expert predictions and uncertainties
                num_experts = len(nigs)
                expert_mus = []
                expert_epistemics = []
                expert_aleatorics = []
                expert_confidences = []
                expert_weights = []
                
                # Collect all v (precision) values for weight computation
                all_vs = []
                for mu_k, v_k, alpha_k, beta_k in nigs:
                    # Apply precision temperature
                    v_k_temp = v_k / args.precision_temperature
                    alpha_k_temp = alpha_k / args.precision_temperature
                    
                    epi_k, ale_k = nig_uncertainty(v_k_temp, alpha_k_temp, beta_k)
                    expert_mus.append(mu_k)
                    expert_epistemics.append(epi_k)
                    expert_aleatorics.append(ale_k)
                    expert_confidences.append(v_k_temp)
                    all_vs.append(v_k_temp)
                
                # Compute weights: weight_k = v_k / sum(all v's)
                total_v = sum(all_vs)
                for v_k in all_vs:
                    weight_k = v_k / (total_v + 1e-8)  # Add small epsilon to avoid division by zero
                    expert_weights.append(weight_k)
                
                # Aggregate for MoNIG prediction
                mu, v, alpha, beta = aggregate_nigs(
                    nigs, args.precision_temperature, args.balance_factor
                )
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                var = epistemic + aleatoric
                std = torch.sqrt(torch.clamp(var, min=1e-8))
            elif model_type == "NIG":
                mu, v, alpha, beta = model(expert_scores, embeddings)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                var = epistemic + aleatoric
                std = torch.sqrt(torch.clamp(var, min=1e-8))
            elif model_type == "Gaussian":
                mu, sigma = model(expert_scores, embeddings)
                std = sigma
                epistemic = torch.zeros_like(std)
                aleatoric = sigma ** 2
            else:
                mu = model(expert_scores, embeddings)
                std = torch.zeros_like(mu)
                epistemic = torch.zeros_like(mu)
                aleatoric = torch.zeros_like(mu)

            mu_np = mu.detach().cpu().numpy().flatten()
            std_np = std.detach().cpu().numpy().flatten()
            labels_np = labels.detach().cpu().numpy().flatten()
            epistemic_np = epistemic.detach().cpu().numpy().flatten()
            aleatoric_np = aleatoric.detach().cpu().numpy().flatten()
            complex_ids_np = np.array(list(complex_ids))
            
            # Convert expert data to numpy (only for MoNIG)
            expert_mus_np = None
            expert_epistemics_np = None
            expert_aleatorics_np = None
            expert_confidences_np = None
            expert_weights_np = None
            if model_type == "MoNIG":
                expert_mus_np = [m.detach().cpu().numpy().flatten() for m in expert_mus]
                expert_epistemics_np = [e.detach().cpu().numpy().flatten() for e in expert_epistemics]
                expert_aleatorics_np = [a.detach().cpu().numpy().flatten() for a in expert_aleatorics]
                expert_confidences_np = [c.detach().cpu().numpy().flatten() for c in expert_confidences]
                expert_weights_np = [w.detach().cpu().numpy().flatten() for w in expert_weights]

            for idx, (cid, y_true, y_pred, s, e_u, a_u) in enumerate(zip(
                complex_ids_np,
                labels_np,
                mu_np,
                std_np,
                epistemic_np,
                aleatoric_np,
            )):
                row = {
                    "ComplexID": cid,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "std": float(s),
                    "epistemic": float(e_u),
                    "aleatoric": float(a_u),
                }
                
                # Add MoNIG aggregated columns for compatibility
                if model_type == "MoNIG":
                    row["MoNIG_Prediction"] = float(y_pred)
                    row["MoNIG_Epistemic"] = float(e_u)
                    row["MoNIG_Aleatoric"] = float(a_u)
                
                # Add per-expert data for MoNIG
                if model_type == "MoNIG":
                    for j in range(num_experts):
                        expert_num = j + 1
                        row[f"Expert{expert_num}_Prediction"] = float(expert_mus_np[j][idx])
                        row[f"Expert{expert_num}_Epistemic"] = float(expert_epistemics_np[j][idx])
                        row[f"Expert{expert_num}_Aleatoric"] = float(expert_aleatorics_np[j][idx])
                        row[f"Expert{expert_num}_Confidence_nu"] = float(expert_confidences_np[j][idx])
                        row[f"Expert{expert_num}_Weight"] = float(expert_weights_np[j][idx])
                    
                    # Expert analysis: most confident expert and confidence ratio
                    confidences = [expert_confidences_np[j][idx] for j in range(num_experts)]
                    most_confident_idx = np.argmax(confidences)
                    row['More_Confident_Expert'] = int(most_confident_idx + 1)
                    if len(confidences) > 1:
                        row['Confidence_Ratio'] = float(max(confidences) / (min(confidences) + 1e-8))
                    else:
                        row['Confidence_Ratio'] = 1.0
                    
                    # Expert disagreement: max pairwise difference in predictions
                    predictions = [expert_mus_np[j][idx] for j in range(num_experts)]
                    if num_experts >= 2:
                        max_disagreement = max(abs(predictions[idx1] - predictions[idx2]) 
                                              for idx1 in range(num_experts) 
                                              for idx2 in range(idx1+1, num_experts))
                        row['Expert_Disagreement'] = float(max_disagreement)
                    else:
                        row['Expert_Disagreement'] = 0.0
                
                all_rows.append(row)

    if len(all_rows) == 0:
        raise RuntimeError("No predictions produced; check your dataset and split")

    import pandas as pd

    df = pd.DataFrame(all_rows)

    if calibrator is not None and model_type in ["MoNIG", "NIG"]:
        # Apply interval calibration to recalibrate uncertainties
        y_pred_array = df["y_pred"].values
        y_std_array = df["std"].values
        
        # Recalibrate std using isotonic calibration
        calibrated_std = apply_interval_calibration(
            y_pred_array, y_std_array, calibrator
        )
        
        # Scale uncertainties proportionally
        total_var = y_std_array ** 2
        calibrated_var = calibrated_std ** 2
        scale = np.ones_like(calibrated_var)
        valid_mask = total_var > 0
        scale[valid_mask] = calibrated_var[valid_mask] / total_var[valid_mask]
        scale = np.clip(scale, 1e-4, None)
        
        # Update uncertainties with calibrated values
        df["epistemic"] = df["epistemic"].values * scale
        df["aleatoric"] = df["aleatoric"].values * scale
        df["std"] = calibrated_std
        
        # Also update MoNIG uncertainties if present
        if "MoNIG_Epistemic" in df.columns:
            df["MoNIG_Epistemic"] = df["MoNIG_Epistemic"].values * scale
            df["MoNIG_Aleatoric"] = df["MoNIG_Aleatoric"].values * scale
            # Update MoNIG_Prediction to match y_pred (should already match, but ensure consistency)
            if "MoNIG_Prediction" in df.columns:
                df["MoNIG_Prediction"] = df["y_pred"].values

    # Compute summary statistics
    y_pred = df["y_pred"].values
    y_true = df["y_true"].values
    errors = np.abs(y_pred - y_true)
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    mean_epistemic = np.mean(df["epistemic"].values)
    mean_aleatoric = np.mean(df["aleatoric"].values)
    mean_std = np.mean(df["std"].values)
    
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total samples: {len(df)}")
    print(f"\nPrediction Accuracy:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Per-expert statistics for MoNIG
    if model_type == "MoNIG" and "Expert1_Prediction" in df.columns:
        num_experts = len([c for c in df.columns if c.startswith("Expert") and "_Prediction" in c])
        print(f"\nPer-Expert Accuracy:")
        for j in range(num_experts):
            expert_num = j + 1
            expert_mae = np.mean(np.abs(df[f'Expert{expert_num}_Prediction'] - df['y_true']))
            print(f"  Expert{expert_num} MAE: {expert_mae:.4f}")
        
        print(f"\nExpert Confidence (ν):")
        for j in range(num_experts):
            expert_num = j + 1
            avg_nu = df[f'Expert{expert_num}_Confidence_nu'].mean()
            print(f"  Expert{expert_num} avg ν: {avg_nu:.4f}")
        
        print(f"\nExpert Weights:")
        for j in range(num_experts):
            expert_num = j + 1
            avg_weight = df[f'Expert{expert_num}_Weight'].mean()
            print(f"  Expert{expert_num} avg weight: {avg_weight:.3f}")
        
        if "More_Confident_Expert" in df.columns:
            print(f"\nExpert Trust Distribution:")
            for j in range(num_experts):
                expert_num = j + 1
                count = (df['More_Confident_Expert'] == expert_num).sum()
                percentage = count / len(df) * 100
                print(f"  Expert{expert_num} more confident: {count} samples ({percentage:.1f}%)")
    
    print(f"\nUncertainty:")
    print(f"  Mean Epistemic: {mean_epistemic:.4f}")
    print(f"  Mean Aleatoric: {mean_aleatoric:.4f}")
    print(f"  Mean Total Std: {mean_std:.4f}")
    
    # Interesting cases for MoNIG
    if model_type == "MoNIG" and "Expert_Disagreement" in df.columns:
        print(f"\nInteresting Cases:")
        high_disagreement = df.nlargest(5, 'Expert_Disagreement')
        print(f"\nTop 5 samples with highest expert disagreement:")
        cols_to_show = ['ComplexID', 'Expert1_Prediction', 'Expert2_Prediction']
        if 'Expert3_Prediction' in df.columns:
            cols_to_show.append('Expert3_Prediction')
        cols_to_show.extend(['Expert_Disagreement', 'y_pred', 'y_true'])
        print(high_disagreement[cols_to_show].to_string(index=False))
        
        if "Confidence_Ratio" in df.columns:
            high_conf_diff = df.nlargest(5, 'Confidence_Ratio')
            print(f"\nTop 5 samples with highest confidence difference:")
            conf_cols = ['ComplexID', 'More_Confident_Expert', 'Confidence_Ratio']
            expert_count = len([c for c in df.columns if c.startswith("Expert") and "_Weight" in c])
            for j in range(expert_count):
                expert_num = j + 1
                conf_cols.append(f'Expert{expert_num}_Weight')
            conf_cols.extend(['y_pred', 'y_true'])
            print(high_conf_diff[conf_cols].to_string(index=False))
    
    print("=" * 70)
    
    df.to_csv(args.output_path, index=False)
    print(f"\nSaved inference results to {args.output_path}")
    print(f"Total rows: {len(df)}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()