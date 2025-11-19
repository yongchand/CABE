import os
import pickle
import argparse
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import uncertainty_toolbox as uct  # only used for calibration helpers

from src.drug_dataset_emb import DrugDiscoveryDatasetEmb
from src.drug_models_emb import (
    DrugDiscoveryMoNIGEmb,
    DrugDiscoveryNIGEmb,
    DrugDiscoveryGaussianEmb,
    DrugDiscoveryBaselineEmb,
)
from src.utils import moe_nig, moe_nig_balanced, criterion_nig as criterion_nig_original


# --------------------------- NIG utilities ---------------------------


def aggregate_nigs(nigs, precision_temperature: float = 1.0, balance_factor: float = 0.3):
    """
    Aggregate multiple NIGs using balanced MoNIG aggregation, with optional
    precision tempering (divide v and alpha by T >= 1 to reduce overconfidence).
    
    Args:
        nigs: List of (mu, v, alpha, beta) tuples
        precision_temperature: Temperature for precision tempering (>= 1.0)
        balance_factor: Balance factor for preventing expert dominance (0.0-1.0)
                       0.0 = standard MoNIG, higher = more balanced
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


def criterion_nig(u, la, alpha, beta, y, risk_weight: float = 0.01):
    """
    Thin wrapper for criterion_nig from utils.py to match signature.
    """
    class Hyp:
        pass

    h = Hyp()
    h.risk = risk_weight
    return criterion_nig_original(u, la, alpha, beta, y, h)


def nig_uncertainty(v, alpha, beta):
    """
    Compute epistemic and aleatoric uncertainties from NIG parameters.
    """
    aleatoric = beta / (alpha - 1)
    epistemic = beta / (v * (alpha - 1))
    return epistemic, aleatoric


def criterion_gaussian(mu, sigma, y):
    """
    Gaussian negative log-likelihood.
    """
    loss = torch.mean(
        0.5 * (torch.log(2 * np.pi * sigma) + ((y - mu) ** 2) / sigma)
    )
    return loss


def evaluate_predictions(mu, y_true):
    mu = mu.detach().cpu().numpy().flatten()
    y_true = y_true.detach().cpu().numpy().flatten()

    mae = np.mean(np.abs(mu - y_true))
    rmse = np.sqrt(np.mean((mu - y_true) ** 2))
    corr = np.corrcoef(mu, y_true)[0, 1]

    ss_res = np.sum((y_true - mu) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"mae": mae, "rmse": rmse, "corr": corr, "r2": r2}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------- Training / eval ---------------------------


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    model_type: str,
    risk_weight: float,
    expert_indices=None,
    evidence_reg_weight: float = 0.01,
    precision_temperature: float = 1.0,
    balance_factor: float = 0.3,
):
    model.train()
    total_loss = 0.0

    for (expert_scores, embeddings), labels, _ in loader:
        # optional expert subset
        if expert_indices is not None and len(expert_indices) < expert_scores.shape[1]:
            expert_scores = expert_scores[:, expert_indices]

        expert_scores = expert_scores.to(device)
        embeddings = embeddings.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        if model_type == "MoNIG":
            nigs = model(expert_scores, embeddings)
            mu_agg, v_agg, alpha_agg, beta_agg = aggregate_nigs(
                nigs, precision_temperature, balance_factor
            )

            # Sum per-expert + aggregated loss (stronger MoNIG signal)
            loss = 0.0
            for mu, v, alpha, beta in nigs:
                loss = loss + criterion_nig(mu, v, alpha, beta, labels, risk_weight)
            loss = loss + criterion_nig(
                mu_agg, v_agg, alpha_agg, beta_agg, labels, risk_weight
            )

            # Evidence regularization on per-expert NIGs only
            evidence_reg = 0.0
            beta_reg = 0.0
            for mu, v, alpha, beta in nigs:
                v_reg = torch.where(
                    torch.abs(v) < 1.0, 0.5 * v ** 2, torch.abs(v) - 0.5
                )
                alpha_reg = torch.where(
                    torch.abs(alpha - 1.0) < 1.0,
                    0.5 * (alpha - 1.0) ** 2,
                    torch.abs(alpha - 1.0) - 0.5,
                )
                evidence_reg = evidence_reg + torch.mean(v_reg) + torch.mean(alpha_reg)

                log_beta = torch.log(beta + 1e-8)
                log_beta_reg = torch.where(
                    torch.abs(log_beta) < 1.0,
                    0.5 * log_beta ** 2,
                    torch.abs(log_beta) - 0.5,
                )
                beta_reg = beta_reg + torch.mean(log_beta_reg)

            loss = loss + evidence_reg_weight * evidence_reg
            loss = loss + evidence_reg_weight * 0.5 * beta_reg
            
            # Expert balance regularization: penalize extreme confidence differences
            # This encourages balanced expert contributions
            if len(nigs) > 1:
                vs = torch.stack([v for _, v, _, _ in nigs], dim=0)  # [num_experts, B, 1]
                vs_mean = vs.mean(dim=0, keepdim=True)  # [1, B, 1]
                vs_std = vs.std(dim=0, keepdim=True)  # [1, B, 1]
                # Penalize when one expert's v is much higher than others
                # Use coefficient of variation squared as penalty
                cv_squared = (vs_std / (vs_mean + 1e-6)) ** 2
                balance_reg = torch.mean(cv_squared)
                # Make this penalty stronger - multiply by 2x
                loss = loss + evidence_reg_weight * balance_reg
                
                # Additional penalty: penalize max/min ratio
                vs_max = vs.max(dim=0, keepdim=True)[0]
                vs_min = vs.min(dim=0, keepdim=True)[0]
                max_min_ratio = (vs_max / (vs_min + 1e-6))
                ratio_penalty = torch.mean(torch.clamp(max_min_ratio - 1.5, min=0.0) ** 2)
                loss = loss + evidence_reg_weight * 0.5 * ratio_penalty

        elif model_type == "NIG":
            mu, v, alpha, beta = model(expert_scores, embeddings)
            loss = criterion_nig(mu, v, alpha, beta, labels, risk_weight)

            v_reg = torch.where(
                torch.abs(v) < 1.0, 0.5 * v ** 2, torch.abs(v) - 0.5
            )
            alpha_reg = torch.where(
                torch.abs(alpha - 1.0) < 1.0,
                0.5 * (alpha - 1.0) ** 2,
                torch.abs(alpha - 1.0) - 0.5,
            )
            evidence_reg = torch.mean(v_reg) + torch.mean(alpha_reg)
            loss = loss + evidence_reg_weight * evidence_reg

        elif model_type == "Gaussian":
            mu, sigma = model(expert_scores, embeddings)
            loss = criterion_gaussian(mu, sigma, labels)
        else:  # Baseline
            preds = model(expert_scores, embeddings)
            loss = torch.nn.functional.mse_loss(preds, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(
    model,
    loader,
    device,
    model_type: str,
    expert_indices=None,
    precision_temperature: float = 1.0,
    balance_factor: float = 0.3,
):
    model.eval()
    all_preds = []
    all_labels = []
    all_epistemic = []
    all_aleatoric = []
    all_variance = []

    with torch.no_grad():
        for (expert_scores, embeddings), labels, _ in loader:
            if expert_indices is not None and len(expert_indices) < expert_scores.shape[1]:
                expert_scores = expert_scores[:, expert_indices]

            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)

            if model_type == "MoNIG":
                nigs = model(expert_scores, embeddings)
                mu, v, alpha, beta = aggregate_nigs(nigs, precision_temperature, balance_factor)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                all_epistemic.extend(epistemic.cpu().numpy().flatten())
                all_aleatoric.extend(aleatoric.cpu().numpy().flatten())
            elif model_type == "NIG":
                mu, v, alpha, beta = model(expert_scores, embeddings)
                epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
                all_epistemic.extend(epistemic.cpu().numpy().flatten())
                all_aleatoric.extend(aleatoric.cpu().numpy().flatten())
            elif model_type == "Gaussian":
                mu, sigma = model(expert_scores, embeddings)
                all_variance.extend(sigma.cpu().numpy().flatten())
            else:
                mu = model(expert_scores, embeddings)

            all_preds.append(mu)
            all_labels.append(labels)

    if len(all_preds) == 0:
        metrics = {"mae": np.nan, "rmse": np.nan, "corr": np.nan, "r2": np.nan}
        if model_type in ["MoNIG", "NIG"]:
            metrics["mean_epistemic"] = np.nan
            metrics["mean_aleatoric"] = np.nan
        elif model_type == "Gaussian":
            metrics["mean_variance"] = np.nan
            metrics["mean_std"] = np.nan
        return metrics

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = evaluate_predictions(all_preds, all_labels)

    if model_type in ["MoNIG", "NIG"]:
        metrics["mean_epistemic"] = float(np.mean(all_epistemic))
        metrics["mean_aleatoric"] = float(np.mean(all_aleatoric))
    elif model_type == "Gaussian":
        metrics["mean_variance"] = float(np.mean(all_variance))
        metrics["mean_std"] = float(np.sqrt(metrics["mean_variance"]))

    return metrics


def collect_predictions_with_uncertainty(
    model,
    loader,
    device,
    model_type: str,
    expert_indices=None,
    precision_temperature: float = 1.0,
    balance_factor: float = 0.3,
):
    if model_type not in ["MoNIG", "NIG"]:
        return None, None, None

    model.eval()
    preds = []
    trues = []
    stds = []
    eps = 1e-8

    with torch.no_grad():
        for (expert_scores, embeddings), labels, _ in loader:
            if expert_indices is not None and len(expert_indices) < expert_scores.shape[1]:
                expert_scores = expert_scores[:, expert_indices]

            expert_scores = expert_scores.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)

            if model_type == "MoNIG":
                nigs = model(expert_scores, embeddings)
                mu, v, alpha, beta = aggregate_nigs(nigs, precision_temperature, balance_factor)
            else:
                mu, v, alpha, beta = model(expert_scores, embeddings)

            epistemic, aleatoric = nig_uncertainty(v, alpha, beta)
            total_std = torch.sqrt(torch.clamp(epistemic + aleatoric, min=eps))

            preds.append(mu.cpu().numpy())
            stds.append(total_std.cpu().numpy())
            trues.append(labels.cpu().numpy())

    if len(preds) == 0:
        return None, None, None

    y_pred = np.concatenate(preds, axis=0).flatten()
    y_std = np.concatenate(stds, axis=0).flatten()
    y_true = np.concatenate(trues, axis=0).flatten()
    return y_pred, y_std, y_true


def fit_isotonic_interval_calibrator(y_pred, y_std, y_true):
    if y_pred is None or y_std is None or y_true is None:
        return None
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
        y_pred, y_std, y_true, prop_type="interval"
    )
    iso_model = uct.recalibration.iso_recal(exp_props, obs_props)
    return iso_model


# --------------------------- Top-level training ---------------------------


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Drug Discovery with MoNIG (Embeddings)"
    )

    # Data
    parser.add_argument(
        "--csv_path",
        type=str,
        default="pdbbind_descriptors_with_experts_and_binding.csv",
        help="Path to CSV file",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="MoNIG",
        choices=["MoNIG", "NIG", "Gaussian", "Baseline"],
        help="Model type",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # Expert selection
    parser.add_argument(
        "--expert1_only", action="store_true", help="Use only Expert 1 (GNINA)"
    )
    parser.add_argument(
        "--expert2_only", action="store_true", help="Use only Expert 2 (BIND)"
    )
    parser.add_argument(
        "--expert3_only", action="store_true", help="Use only Expert 3 (flowdock)"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--risk_weight",
        type=float,
        default=0.005,
        help="Risk regularization weight",
    )
    parser.add_argument(
        "--evidence_reg_weight",
        type=float,
        default=0.01,
        help="Evidence regularization weight",
    )
    parser.add_argument(
        "--precision_temperature",
        type=float,
        default=1.0,
        help="Precision temperature (>=1.0)",
    )
    parser.add_argument(
        "--balance_factor",
        type=float,
        default=0.3,
        help="Expert balance factor (0.0-1.0, higher = more balanced, prevents dominance)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    return parser


def run_training(args):
    # expert flags
    expert_flags = [args.expert1_only, args.expert2_only, getattr(args, "expert3_only", False)]
    expert_names = ["GNINA", "BIND", "flowdock"]
    expert_mode = sum(expert_flags)

    if expert_mode == 0:
        expert_indices = list(range(3))  # assume all 3 experts
        actual_num_experts = len(expert_indices)
        expert_config = "All Experts (GNINA, BIND, flowdock)"
    elif expert_mode == 1:
        expert_indices = [i for i, f in enumerate(expert_flags) if f]
        actual_num_experts = 1
        expert_config = f"Expert {expert_indices[0] + 1} ({expert_names[expert_indices[0]]}) only"
    else:
        expert_indices = [i for i, f in enumerate(expert_flags) if f]
        actual_num_experts = len(expert_indices)
        selected_names = [expert_names[i] for i in expert_indices]
        expert_config = f"Experts: {', '.join(selected_names)}"

    # seed + splits
    set_seed(args.seed)

    # Check if split.json exists - if so, use it instead of loading CASF from CSV
    split_json_path = "split.json"
    use_split_json = os.path.isfile(split_json_path)
    
    casf2016_pdb_ids = None
    if not use_split_json:
        # Fallback to old method: load CASF from CSV
        casf_path = "data/coreset_dirs.csv"
        if os.path.isfile(casf_path):
            with open(casf_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if lines and ("directory" in lines[0].lower() or "pdb" in lines[0].lower()):
                lines = lines[1:]
            casf2016_pdb_ids = lines
            print(f"Loaded {len(casf2016_pdb_ids)} CASF 2016 PDB IDs from {casf_path}")
        else:
            print(f"Warning: CASF 2016 file not found at {casf_path}, proceeding without exclusion")

    print("=" * 50)
    print("Drug Discovery with MoNIG (Embeddings)")
    print("=" * 50)
    print(f"Model: {args.model_type}")
    print(f"Expert Config: {expert_config}")
    print(f"Device: {args.device}")
    print(f"CSV: {args.csv_path}")
    if use_split_json:
        print(f"Using splits from: {split_json_path}")
    elif casf2016_pdb_ids:
        print(f"CASF 2016 excluded: {len(casf2016_pdb_ids)} complexes")
    print("=" * 50)

    # Datasets
    print("\nLoading datasets...")
    train_dataset = DrugDiscoveryDatasetEmb(
        args.csv_path, split="train", seed=args.seed, casf2016_pdb_ids=casf2016_pdb_ids
    )
    norm_stats = {"mean": train_dataset.emb_mean, "std": train_dataset.emb_std}

    # --- Dataset-level expert score statistics for z-scoring ---
    expert_scores_train = train_dataset.expert_scores.numpy()
    score_mean_full = expert_scores_train.mean(axis=0).astype(np.float32)
    score_std_full = (expert_scores_train.std(axis=0) + 1e-6).astype(np.float32)
    print("Expert score stats (mean, std) per expert:")
    for i, (m, s) in enumerate(zip(score_mean_full, score_std_full)):
        print(f"  Expert {i+1}: mean={m:.4f}, std={s:.4f}")

    valid_dataset = DrugDiscoveryDatasetEmb(
        args.csv_path,
        split="valid",
        seed=args.seed,
        normalization_stats=norm_stats,
        casf2016_pdb_ids=casf2016_pdb_ids,
    )
    test_dataset = DrugDiscoveryDatasetEmb(
        args.csv_path,
        split="test",
        seed=args.seed,
        normalization_stats=norm_stats,
        casf2016_pdb_ids=casf2016_pdb_ids,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    embedding_dim, num_experts_available = train_dataset.get_dim()

    # map expert_indices to actual stats
    if len(expert_indices) == num_experts_available:
        score_mean = score_mean_full
        score_std = score_std_full
    else:
        score_mean = score_mean_full[expert_indices]
        score_std = score_std_full[expert_indices]

    print("\nData dimensions:")
    print(f"  Embeddings: {embedding_dim}")
    print(f"  Experts available: {num_experts_available}")
    print(f"  Experts using: {len(expert_indices)}")

    # Hyperparams holder
    class Hyp:
        pass

    hyp = Hyp()
    hyp.num_experts = len(expert_indices)
    hyp.embedding_dim = embedding_dim
    hyp.hidden_dim = args.hidden_dim
    hyp.dropout = args.dropout
    hyp.score_mean = score_mean.tolist()
    hyp.score_std = score_std.tolist()
    # Expert normalization to prevent dominance
    hyp.use_expert_normalization = True
    hyp.expert_normalization_strength = 0.8  # Very strong normalization: compress differences by 80%

    # Model
    if args.model_type == "MoNIG":
        model = DrugDiscoveryMoNIGEmb(hyp)
    elif args.model_type == "NIG":
        model = DrugDiscoveryNIGEmb(hyp)
    elif args.model_type == "Gaussian":
        model = DrugDiscoveryGaussianEmb(hyp)
    else:
        model = DrugDiscoveryBaselineEmb(hyp)

    model = model.to(args.device)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=15, factor=0.5)

    # Training loop
    print("\nStarting training...")
    best_val_mae = float("inf")
    patience_counter = 0
    max_patience = 30

    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/best_{args.model_type}_emb.pt"
    norm_stats_path = f"saved_models/best_{args.model_type}_emb_norm_stats.npz"
    calibrator_path = f"saved_models/best_{args.model_type}_emb_calibrator.pkl"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.model_type,
            args.risk_weight,
            expert_indices,
            args.evidence_reg_weight,
            args.precision_temperature,
            args.balance_factor,
        )
        val_metrics = evaluate(
            model,
            valid_loader,
            args.device,
            args.model_type,
            expert_indices,
            args.precision_temperature,
            args.balance_factor,
        )

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["mae"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"  → LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

        if epoch == 1 or epoch % 5 == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                f"  Train Loss: {train_loss:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"RMSE: {val_metrics['rmse']:.4f}, "
                f"Corr: {val_metrics['corr']:.4f}, "
                f"R²: {val_metrics['r2']:.4f}"
            )
            if args.model_type in ["MoNIG", "NIG"]:
                print(
                    f"  Epistemic: {val_metrics['mean_epistemic']:.4f}, "
                    f"Aleatoric: {val_metrics['mean_aleatoric']:.4f}"
                )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(model.state_dict(), model_path)
            print(f"  → New best model saved to {model_path} (MAE: {best_val_mae:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    # Final eval + calibration
    print("\n" + "=" * 50)
    print("Final evaluation on test set...")
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model = model.to(args.device)

    iso_model = None
    if args.model_type in ["MoNIG", "NIG"]:
        y_pred_val, y_std_val, y_true_val = collect_predictions_with_uncertainty(
            model,
            valid_loader,
            args.device,
            args.model_type,
            expert_indices,
            args.precision_temperature,
            args.balance_factor,
        )
        try:
            iso_model = fit_isotonic_interval_calibrator(
                y_pred_val, y_std_val, y_true_val
            )
        except RuntimeError as e:
            print(f"Warning: isotonic calibration failed: {e}")
            iso_model = None

        if iso_model is not None:
            with open(calibrator_path, "wb") as f:
                pickle.dump({"iso_model": iso_model}, f)
            print(f"Saved isotonic recalibrator to {calibrator_path}")
        else:
            calibrator_path = None
    else:
        calibrator_path = None

    # Save normalization + score stats
    np.savez(
        norm_stats_path,
        mean=norm_stats["mean"],
        std=norm_stats["std"],
        score_mean=score_mean_full,
        score_std=score_std_full,
    )
    print(f"Saved normalization + score stats to {norm_stats_path}")
    if calibrator_path:
        print(f"Calibration artifact: {calibrator_path}")

    # Test
    if len(test_dataset) > 0:
        test_metrics = evaluate(
            model,
            test_loader,
            args.device,
            args.model_type,
            expert_indices,
            args.precision_temperature,
            args.balance_factor,
        )
        print("\nTest Results:")
        print(
            f"  MAE: {test_metrics['mae']:.4f}, "
            f"RMSE: {test_metrics['rmse']:.4f}, "
            f"Corr: {test_metrics['corr']:.4f}, "
            f"R²: {test_metrics['r2']:.4f}"
        )
        if args.model_type in ["MoNIG", "NIG"]:
            print(
                f"  Mean Epistemic: {test_metrics['mean_epistemic']:.4f}, "
                f"Mean Aleatoric: {test_metrics['mean_aleatoric']:.4f}"
            )
        elif args.model_type == "Gaussian":
            print(
                f"  Mean Predicted Variance: {test_metrics['mean_variance']:.4f}, "
                f"Mean Predicted Std Dev: {test_metrics['mean_std']:.4f}"
            )
    else:
        print("Test Results: [Skipped - empty test set]")
    print("=" * 50)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()