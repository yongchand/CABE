# Drug Discovery with MoNIG

Probabilistic binding-affinity prediction pipeline that combines multiple docking experts through a Mixture of Normal–Inverse-Gamma (MoNIG) head. The system produces point predictions plus calibrated epistemic and aleatoric uncertainties for every protein–ligand complex.

## Overview

This repository implements a drug discovery pipeline using MoNIG (Mixture of Normal-Inverse Gamma distributions) for trustworthy uncertainty quantification in binding affinity prediction. The system:

- Combines multiple expert predictions (e.g., GNINA, BIND) with molecular embeddings
- Produces calibrated epistemic and aleatoric uncertainty estimates
- Uses isotonic recalibration for improved uncertainty calibration
- Supports ablation studies with single-expert configurations

## Setup

### Requirements

- Python ≥3.10
- PyTorch ≥2.0
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Create a conda environment (recommended)
conda create -n monig python=3.10
conda activate monig

# Install dependencies
pip install -r requirements.txt
```

The main dependencies include:
- `torch` - Deep learning framework
- `numpy`, `pandas` - Data processing
- `matplotlib` - Plotting
- `uncertainty_toolbox` - Uncertainty calibration utilities
- `scipy` - Scientific computing

## Quick Start

> **📖 For detailed step-by-step instructions, see [QUICKSTART.md](QUICKSTART.md)**

### Immediate Commands

**Run inference with existing model:**
```bash
python main.py infer \
  --model_path saved_models/best_MoNIG_emb.pt \
  --csv_path pdbbind_descriptors_with_experts_and_binding.csv \
  --split test
```

**Train a new model:**
```bash
python main.py train \
  --csv_path pdbbind_descriptors_with_experts_and_binding.csv \
  --model_type MoNIG
```

### 1. Training

Train a MoNIG model on your binding affinity data:

```bash
python main.py train \
  --csv_path pdbbind_descriptors_with_experts_and_binding.csv \
  --model_type MoNIG \
  --batch_size 64 \
  --hidden_dim 256 \
  --dropout 0.2 \
  --epochs 150 \
  --lr 5e-4 \
  --risk_weight 0.005 \
  --seed 42
```

**Key Arguments:**

| Argument | Description | Default |
| --- | --- | --- |
| `--model_type` | Model type: `MoNIG`, `NIG`, `Gaussian`, `Baseline` | `MoNIG` |
| `--csv_path` | Path to input CSV file | `pdbbind_descriptors_with_experts_and_binding.csv` |
| `--batch_size` | Batch size for training | `64` |
| `--hidden_dim` | Hidden dimension for neural networks | `256` |
| `--dropout` | Dropout rate | `0.2` |
| `--epochs` | Number of training epochs | `150` |
| `--lr` | Learning rate | `5e-4` |
| `--risk_weight` | Evidential risk regularization weight | `0.005` |
| `--expert1_only` | Use only Expert 1 (GNINA) | `False` |
| `--expert2_only` | Use only Expert 2 (BIND) | `False` |
| `--seed` | Random seed for reproducibility | `42` |

**Outputs:**
- `saved_models/best_<MODEL>_emb.pt` - Best model checkpoint (lowest validation MAE)
- `saved_models/best_<MODEL>_emb_norm_stats.npz` - Normalization statistics (mean/std)
- `saved_models/best_<MODEL>_emb_calibrator.pkl` - Isotonic recalibration model

### 2. Inference

Run inference on test data:

```bash
python main.py infer \
  --model_path saved_models/best_MoNIG_emb.pt \
  --csv_path pdbbind_descriptors_with_experts_and_binding.csv \
  --split test \
  --output_path test_inference_results.csv
```

**Key Arguments:**

| Argument | Description | Default |
| --- | --- | --- |
| `--model_path` | Path to trained model checkpoint | **Required** |
| `--csv_path` | Path to input CSV file | `pdbbind_descriptors_with_experts_and_binding.csv` |
| `--split` | Data split: `train`, `valid`, `test` | `test` |
| `--output_path` | Path to save results CSV | `test_inference_results.csv` |
| `--norm_stats_path` | Path to normalization stats (auto-detected if not provided) | `None` |
| `--calibrator_path` | Path to isotonic calibrator (auto-detected if not provided) | `None` |

The inference script automatically:
- Loads normalization statistics from `*_norm_stats.npz`
- Applies isotonic recalibration if `*_calibrator.pkl` exists
- Generates detailed CSV with per-expert and aggregated predictions/uncertainties

### 3. Uncertainty Analysis

Analyze uncertainty calibration and generate plots:

```bash
python main.py analyze \
  --csv test_inference_results.csv \
  --output_prefix test_uncertainty
```

**Outputs:**
- `*_uct_calibration.png` - Reliability/calibration curve
- `*_uct_intervals.png` - Predictive intervals visualization
- `*_uct_intervals_ordered.png` - Ordered predictive intervals
- `*_uct_confidence_band.png` - Confidence bands
- `*_custom_analysis.png` - Custom diagnostic plots
- `*_expert_stats.png` - Expert-level statistics

## Data Format

The input CSV file must contain the following columns:

- **`ComplexID`**: Unique identifier for each protein-ligand complex
- **`Binding_Affinity`**: Binding affinity strings (e.g., `Kd=6.67uM`, `Ki=19nM`, `IC50=5.2mM`)
  - Supported units: `nM`, `uM`/`μM`, `mM`, `M`, `pM`
  - Automatically converted to pKd/pKi values (negative log of molar concentration)
- **`GNINA_Affinity`**: Expert 1 prediction (GNINA docking score)
- **`BIND_pIC50`**: Expert 2 prediction (BIND pIC50 score)
- **`Emb_0` to `Emb_703`**: Molecular/protein embeddings (704-dimensional float vectors)

**Example CSV structure:**
```csv
ComplexID,Binding_Affinity,GNINA_Affinity,BIND_pIC50,Emb_0,Emb_1,...,Emb_703
1A2B,Kd=6.67uM,7.2,6.8,0.123,0.456,...,0.789
1C3D,Ki=19nM,8.1,7.9,0.234,0.567,...,0.890
...
```

## Model Architecture

The MoNIG architecture combines expert predictions with molecular embeddings:

```
Molecular embeddings (704-dim)
┌─────────────────────────────────────────────┐
│  Shared Embedding Tower (all experts)      │
│  Linear(hidden_dim) → ReLU → Dropout       │
│  Linear(hidden_dim) → ReLU → Dropout       │
└──────────────────┬──────────────────────────┘
                   │ emb_feat [hidden_dim]
                   │
Expert i           │
────────┐          │
        │          │
Raw score (Eᵢ) ────┼─── Normalize (per-batch, training)
        │          │    Dropout (p=0.2, training)
        │          │
        │          │    ┌──────────────────────────┐
        └─ Score MLP_i ─┤  Linear(score_hidden)    │
          (small)       │  → ReLU → Dropout        │
                        │  Linear(score_hidden)    │
                        └───┬──────────────────────┘
                            │ score_feat [score_hidden]
                            │
                            │ Late Fusion (Concat)
                            ├──────────────────────┐
                            │                      │
                            ▼                      ▼
                    ┌─────────────────────────────────────┐
                    │  Unified NIG Head_i                 │
                    │  Linear(hidden_dim) → ReLU → Dropout│
                    │  Linear(4) → [μ, log_ν, log_α, log_β]│
                    └─────────────────────────────────────┘
                            │
                            ▼
                    (μᵢ, νᵢ, αᵢ, βᵢ)
                    with constraints:
                    ν > v_min, α > 1 + alpha_min, β > beta_min

MoNIG Aggregator (Equation 9) with Precision Tempering
NIG₁ ⊕ NIG₂ ⊕ … ⊕ NIG_k → Combined NIG(μ, ν, α, β)
```

**Key Components:**

1. **Shared Embedding Tower**: Single MLP processes embeddings for all experts (parameter-efficient)
2. **Per-Expert Score Towers**: Small MLPs (hidden_dim//4) process each expert's scalar score independently
3. **Late Fusion**: Concatenates embedding features and score features before NIG head
4. **Unified NIG Heads**: Each expert has a single head that outputs all 4 NIG parameters (μ, ν, α, β) simultaneously
5. **Expert Dropout**: Randomly zeros out expert scores during training (p=0.2) for regularization
6. **Per-Batch Normalization**: Normalizes expert scores per batch during training
7. **MoNIG Aggregator**: Combines multiple NIG distributions using closed-form formulas with precision tempering

**Uncertainty Decomposition:**
- **Epistemic uncertainty** = β / (ν × (α - 1)) - model uncertainty
- **Aleatoric uncertainty** = β / (α - 1) - data uncertainty

## Workflow

1. **Dataset Preparation** (`src/drug_dataset_emb.py`)
   - Parses binding affinity strings to pKd/pKi values
   - Filters invalid samples
   - Splits data into train/valid/test (70/15/15 by default)
   - **Fits normalization statistics from training data only** (prevents data leakage)
   - Reuses normalization stats for validation, test, and inference

2. **Training** (`main.py train` or `src/train_drug_discovery_emb.py`)
   - Supports multiple model types: MoNIG, NIG, Gaussian, Baseline
   - Early stopping based on validation MAE
   - Fits isotonic recalibrator on validation set
   - Saves best checkpoint, normalization stats, and calibrator

3. **Inference** (`main.py infer` or `src/inference_drug_discovery.py`)
   - Loads model, normalization stats, and calibrator
   - Generates per-expert and aggregated predictions
   - Applies isotonic recalibration to uncertainties
   - Exports detailed CSV with all predictions and uncertainties

4. **Uncertainty Analysis** (`main.py analyze` or `src/analyze_uncertainty.py`)
   - Computes calibration metrics
   - Generates diagnostic plots
   - Analyzes expert behavior and disagreement

## Isotonic Recalibration

The pipeline uses isotonic regression to improve uncertainty calibration:

1. **Validation Pass**: Collect (μ, σ, y) triples where σ = √(epistemic + aleatoric)
2. **Fitting**: Compute empirical interval coverages and fit isotonic regression model
3. **Inference**: Rescale uncertainties using the fitted calibrator

This approach reduces miscalibration without assuming uniform correction across all quantile levels.

## Example Results

Typical performance on PDBbind subset (test set, before isotonic recalibration):

| Model | MAE ↓ | RMSE ↓ | Corr ↑ | R² ↑ | Mean Epistemic | Mean Aleatoric |
| --- | --- | --- | --- | --- | --- | --- |
| GNINA expert only | 0.976 | 1.318 | 0.737 | 0.537 | 0.996 | 0.077 |
| BIND expert only | 0.993 | 1.325 | 0.734 | 0.532 | 1.423 | 0.104 |
| **MoNIG (both experts)** | **0.964** | **1.301** | **0.747** | **0.549** | 0.146 | 0.241 |

MoNIG achieves better accuracy while providing well-calibrated uncertainty estimates. Calibration plots after isotonic recalibration are saved as PNG files.

## Repository Structure

```
├── main.py                          # Main entry point (train/infer/analyze modes)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── train_drug_discovery_emb.py  # Training implementation
│   ├── inference_drug_discovery.py  # Inference implementation
│   ├── analyze_uncertainty.py       # Uncertainty analysis implementation
│   ├── drug_dataset_emb.py          # Dataset + normalization
│   ├── drug_models_emb.py           # Model architectures
│   └── utils.py                     # MoNIG aggregation + losses
├── saved_models/                    # Model checkpoints and artifacts
│   ├── best_<MODEL>_emb.pt          # Model weights
│   ├── best_<MODEL>_emb_norm_stats.npz  # Normalization stats
│   └── best_<MODEL>_emb_calibrator.pkl   # Isotonic calibrator
└── test_uncertainty_*.png           # Example calibration plots
```

## Running from Source

You can also run the scripts directly from the `src/` directory:

```bash
# Training
python -m src.train_drug_discovery_emb [args]

# Inference
python -m src.inference_drug_discovery [args]
```

Or use the unified main.py entry point:

```bash
# Training
python main.py train [args]

# Inference
python main.py infer [args]

# Uncertainty Analysis
python main.py analyze [args]
```

## Extending the Pipeline

To add more experts:

1. Add expert score columns to your CSV (e.g., `Expert3_Score`)
2. Update `src/drug_dataset_emb.py` to include the new expert columns in `self.expert_cols`
3. The model will automatically create calibrators and evidential heads for all experts

To use different embeddings:

1. Update the embedding column names/indices in `src/drug_dataset_emb.py`
2. Adjust `embedding_dim` hyperparameter accordingly

## Notes

- **Data Leakage Prevention**: Normalization statistics are computed from training data only and reused for all splits
- **Reproducibility**: Use `--seed` flag for consistent train/valid/test splits
- **Model Checkpoints**: Always keep the `.pt`, `.npz`, and `.pkl` files together for consistent inference
- **Calibration**: Isotonic recalibration is optional but recommended for better uncertainty estimates

## Citation

If you use this code, please cite the original MoNIG paper:

```bibtex
@article{ma2021trustworthy,
  title={Trustworthy multimodal regression with mixture of normal-inverse gamma distributions},
  author={Ma, Huan and Han, Zongbo and Zhang, Changqing and Fu, Huazhu and Zhou, Joey Tianyi and Hu, Qinghua},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={6881--6893},
  year={2021}
}
```

## License

See LICENSE file for details.
