# CABE: Context-Aware Binding Affinity Estimation with MoNIG

**CABE** is a probabilistic binding-affinity prediction pipeline that combines multiple docking engines through a **Mixture of Normal-Inverse-Gamma (MoNIG)** framework. The system produces point predictions plus calibrated epistemic and aleatoric uncertainties for every protein-ligand complex, with **per-engine confidence and uncertainty quantification**.

## Key Features

- **Multi-Engine Fusion**: Combines multiple docking engines (GNINA, BIND, flowdock, DynamicBind) into a single calibrated prediction
- **Per-Engine Analysis**: Provides confidence, uncertainty, and reliability scores for each individual engine
- **Context-Aware Reliability**: Adapts engine reliability based on molecular context (protein/ligand embeddings)
- **Uncertainty Decomposition**: Separates epistemic (model) and aleatoric (data) uncertainty
- **Conformal Prediction**: Provides coverage-guaranteed prediction intervals (95% target)
- **Comprehensive Analysis**: Detailed uncertainty metrics and visualizations

## Architecture & Logic

### High-Level Architecture

```
Input: Expert Scores + Molecular Embeddings
    ↓
┌─────────────────────────────────────────────────────────┐
│  MoNIG Architecture                                      │
│                                                          │
│  ┌──────────────────┐      ┌──────────────────────┐   │
│  │ Expert Scores    │      │ 703D Embeddings      │   │
│  │ [GNINA, BIND,    │      │ (protein/ligand)     │   │
│  │  flowdock,       │      │                      │   │
│  │  DynamicBind]    │      │                      │   │
│  └────────┬─────────┘      └──────────┬───────────┘   │
│           │                           │                │
│           ↓                           ↓                │
│  ┌─────────────────┐      ┌──────────────────────┐   │
│  │ Engine Score     │      │ Reliability Network │   │
│  │ MLP              │      │ (MLP)                │   │
│  │                  │      │                      │   │
│  │ Per-expert:      │      │ Outputs: r_j ∈ (0,1) │   │
│  │ • Calibrator     │      │ per engine           │   │
│  │ • Evidential     │      │                      │   │
│  │   Head           │      │                      │   │
│  └────────┬────────┘      └──────────┬───────────┘   │
│           │                           │                │
│           ↓                           ↓                │
│  ┌─────────────────────────────────────┐              │
│  │ Per-Expert NIG Parameters           │              │
│  │ μ_j, ν_j, α_j, β_j                  │              │
│  │                                      │              │
│  │ RELIABILITY SCALING (BEFORE AGG)    │              │
│  │ • ν_j ← ν_j × r_j                   │              │
│  │ • α_j ← 1 + (α_j-1) × r_j          │              │
│  │ • β_j ← β_j × r_j                   │              │
│  │ • μ_j ← μ_j (NOT scaled)            │              │
│  └──────────────┬──────────────────────┘              │
│                 │                                       │
│                 ↓                                       │
│  ┌──────────────────────────────────────┐              │
│  │ MoNIG Aggregation (Equation 9)        │              │
│  │ Uses SCALED parameters                │              │
│  └──────────────┬───────────────────────┘              │
│                 │                                       │
│                 ↓                                       │
│  ┌──────────────────────────────────────┐              │
│  │ Final Aggregated NIG                 │              │
│  │ μ_final, ν_final, α_final, β_final   │              │
│  └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
    ↓
Output: Prediction + Epistemic + Aleatoric Uncertainty
```

### Detailed Component Logic

#### 1. **Engine Score MLP** (Per-Expert Processing)
- **Purpose**: Convert raw expert scores to NIG parameters
- **Components**:
  - **Calibrator**: MLP (1→64→32→1) adjusts each expert's score
  - **Evidential Head**: Produces μ, ν, α, β from calibrated score
- **Output**: Per-expert NIG parameters (μ_j, ν_j, α_j, β_j)

#### 2. **Reliability Network**
- **Purpose**: Compute context-dependent reliability for each expert
- **Architecture**: MLP (703→512→256→128→num_experts) with Sigmoid
- **Output**: r_j ∈ (0,1) per expert
- **Interpretation**: Higher r_j = more reliable expert for that specific molecular context

#### 3. **Reliability Scaling** (Mathematical Formulation)

**Critical**: Reliability scaling happens **BEFORE** MoNIG aggregation.

For each expert j with NIG parameters (μ_j, ν_j, α_j, β_j) and reliability r_j:

```
μ̃_j = μ_j                                    (mean unchanged)
ν̃_j = ν_j × r_j                              (precision scaled)
ᾶ_j = 1 + (α_j - 1) × r_j                    (shape scaled, ensures α > 1)
β̃_j = β_j × r_j                              (scale scaled)
```

**Effect on Uncertainty:**
- **Epistemic uncertainty** = β̃_j / (ν̃_j × (ᾶ_j - 1)) = β_j / (ν_j × (α_j - 1) × r_j)
  - Low r_j → Higher epistemic uncertainty
- **Aleatoric uncertainty** = β̃_j / (ᾶ_j - 1) = β_j / (α_j - 1)
  - Unchanged by reliability scaling

**Why μ_j is NOT scaled:**
- Preserves calibrated predictions
- Only uncertainty reflects reliability
- Ensures predictions remain accurate while uncertainty adapts

#### 4. **MoNIG Aggregation** (After Scaling)

**Critical**: Aggregation uses the **scaled** parameters (ν̃_j, ᾶ_j, β̃_j), not original ones.

For two scaled NIGs (μ₁, ν̃₁, ᾶ₁, β̃₁) and (μ₂, ν̃₂, ᾶ₂, β̃₂):

```
μ_final = (ν̃₁ × μ₁ + ν̃₂ × μ₂) / (ν̃₁ + ν̃₂)
        = (r₁ν₁μ₁ + r₂ν₂μ₂) / (r₁ν₁ + r₂ν₂)
ν_final = ν̃₁ + ν̃₂ = r₁ν₁ + r₂ν₂
α_final = ᾶ₁ + ᾶ₂ + 0.5
β_final = β̃₁ + β̃₂ + 0.5 × [ν̃₁(μ₁ - μ_final)² + ν̃₂(μ₂ - μ_final)²]
```

**Key Insight**: Reliability scores r_j directly influence the final aggregated mean through the weighted average formula. Lower reliability → lower weight in aggregation.

#### 5. **Uncertainty Decomposition**

From final aggregated NIG parameters:
- **Epistemic uncertainty** = β_final / (ν_final × (α_final - 1)) - model uncertainty (reducible)
- **Aleatoric uncertainty** = β_final / (α_final - 1) - data uncertainty (inherent)
- **Total uncertainty** = Epistemic + Aleatoric

#### 6. **Conformal Prediction** (Post-Training)

- **Calibration**: Computes quantile on validation set using normalized residuals `|y - y_pred| / uncertainty`
- **Inference**: Creates intervals as `[μ - quantile×σ, μ + quantile×σ]`
- **Guarantee**: Provides coverage guarantees (e.g., 95%)
- **Note**: Currently applied to aggregated prediction only; per-engine CP is possible but not implemented

## What CABE Can Do

### Core Capabilities

#### 1. **Multi-Engine Binding Affinity Prediction**
- Combines multiple docking engines into a single calibrated prediction
- Produces aggregated prediction with uncertainty quantification

#### 2. **Per-Engine Confidence & Uncertainty**

**Yes, CABE provides detailed per-engine analysis:**

For each engine j:
- **Prediction (μ_j)**: Calibrated binding affinity prediction
- **Confidence (ν_j)**: Precision parameter (higher = more confident)
- **Epistemic Uncertainty**: β_j / (ν_j × (α_j - 1)) - model uncertainty
- **Aleatoric Uncertainty**: β_j / (α_j - 1) - data uncertainty
- **Reliability Score (r_j)**: Context-dependent reliability (0-1) - implicit via scaling
- **Weight**: Contribution to final prediction (normalized ν_j)

**Example Output Columns:**
```
Expert1_Prediction: 7.23
Expert1_Confidence_nu: 2.45
Expert1_Epistemic: 0.12
Expert1_Aleatoric: 0.08
Expert1_Weight: 0.35
```

#### 3. **Context-Aware Reliability**
- Reliability network adapts to molecular context
- Same engine can have different reliability for different complexes
- Enables dynamic expert weighting based on molecular features

#### 4. **Uncertainty Decomposition**
- Separates epistemic (model) and aleatoric (data) uncertainty
- Helps identify when more data vs. better models are needed

#### 5. **Calibrated Prediction Intervals**
- **Standard intervals**: Based on NIG uncertainty (z-score based)
- **Conformal intervals**: Coverage-guaranteed intervals (95% target)
- Currently: Conformal prediction applied to aggregated prediction only
- **Future**: Per-engine conformal prediction can be implemented

#### 6. **Expert Analysis**
- Expert disagreement: Max pairwise difference
- Most confident expert: Highest ν_j
- Confidence ratio: Max/min confidence
- Weight distribution: How much each expert contributes

### Use Cases

1. **Drug Discovery**: Predict binding affinity with uncertainty
2. **Active Learning**: Prioritize high-uncertainty samples for annotation
3. **Quality Control**: Flag low-confidence predictions
4. **Engine Comparison**: Compare engine performance and reliability
5. **Risk Assessment**: Use uncertainty for decision-making

### Output Information

For each protein-ligand complex, CABE provides:

**Aggregated (MoNIG):**
- Final prediction: Aggregated binding affinity
- Total uncertainty: Epistemic + aleatoric
- Conformal intervals: Coverage-guaranteed prediction ranges

**Per-Engine:**
- Individual predictions: Each engine's calibrated output
- Individual uncertainties: Epistemic and aleatoric per engine
- Confidence scores: ν_j (precision parameter)
- Reliability scores: Context-dependent r_j (implicit via scaling)
- Contribution weights: How much each engine contributes

**Analysis:**
- Expert disagreement: Measure of consensus
- Most confident expert: Which engine is most certain
- Confidence ratio: Spread of confidence levels

## Setup

### Requirements

- Python ≥3.10
- PyTorch ≥2.0
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Create a conda environment (recommended)
conda create -n cabe python=3.10
conda activate cabe

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

### Training

```bash
python main.py train \
  --csv_path pdbbind_descriptors_with_experts_and_binding.csv \
  --model_type MoNIG \
  --batch_size 64 \
  --epochs 150 \
  --lr 5e-4 \
  --risk_weight 0.005 \
  --conformal_coverage 0.95 \
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
| `--conformal_coverage` | Target coverage for conformal prediction | `0.95` |
| `--expert1_only` | Use only Expert 1 (GNINA) | `False` |
| `--expert2_only` | Use only Expert 2 (BIND) | `False` |
| `--seed` | Random seed for reproducibility | `42` |

**Outputs:**
- `saved_models/best_<MODEL>_emb.pt` - Best model checkpoint (lowest validation MAE)
- `saved_models/best_<MODEL>_emb_norm_stats.npz` - Normalization statistics (mean/std)
- `saved_models/best_<MODEL>_emb_calibrator.pkl` - Isotonic recalibration model
- `saved_models/best_<MODEL>_emb_conformal.npz` - Conformal prediction quantile

### Inference

```bash
python main.py infer \
  --model_path saved_models/best_MoNIG_emb.pt \
  --csv_path pdbbind_descriptors_with_experts_and_binding.csv \
  --split test \
  --output_path test_inference_results.csv
```

The inference script automatically:
- Loads normalization statistics from `*_norm_stats.npz`
- Loads conformal quantile from `*_conformal.npz` (if available)
- Applies isotonic recalibration if `*_calibrator.pkl` exists
- Generates detailed CSV with per-expert and aggregated predictions/uncertainties
- Includes conformal prediction intervals if quantile is available

### Uncertainty Analysis

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
- `*_conformal_analysis.png` - Conformal prediction analysis (if conformal intervals present)
- `*_untrustworthy_engines.png` - Analysis of when engines are unreliable

## Data Format

The input CSV file must contain the following columns:

- **`ComplexID`**: Unique identifier for each protein-ligand complex
- **`Binding_Affinity`**: Binding affinity strings (e.g., `Kd=6.67uM`, `Ki=19nM`, `IC50=5.2mM`)
  - Supported units: `nM`, `uM`/`μM`, `mM`, `M`, `pM`
  - Automatically converted to pKd/pKi values (negative log of molar concentration)
- **`GNINA_Affinity`**: Expert 1 prediction (GNINA docking score)
- **`BIND_pIC50`**: Expert 2 prediction (BIND pIC50 score)
- **`flowdock_score`**: Expert 3 prediction (flowdock score)
- **`DynamicBind_score`**: Expert 4 prediction (DynamicBind score)
- **`Emb_0` to `Emb_703`**: Molecular/protein embeddings (704-dimensional float vectors)

**Example CSV structure:**
```csv
ComplexID,Binding_Affinity,GNINA_Affinity,BIND_pIC50,flowdock_score,DynamicBind_score,Emb_0,Emb_1,...,Emb_703
1A2B,Kd=6.67uM,7.2,6.8,7.1,7.0,0.123,0.456,...,0.789
1C3D,Ki=19nM,8.1,7.9,8.0,8.2,0.234,0.567,...,0.890
...
```

## Model Architecture Details

### Training Flow

1. **Forward Pass**: Expert scores + embeddings → per-expert NIGs → reliability scaling → aggregated NIG
2. **Loss Computation**: NIG loss (proper scoring rule) + risk regularization
   - Loss computed for each expert NIG + aggregated NIG
   - Average loss used for backpropagation
3. **Backpropagation**: Updates calibrators, evidential heads, and reliability network
4. **Post-Training**: 
   - Computes conformal quantile on validation set
   - Fits isotonic recalibrator for uncertainty calibration

### Loss Function

Uses the original NIG loss from the MoNIG paper:
- Proper scoring rule for Student-t predictive distribution
- Risk regularization term to prevent overconfidence
- Mathematically consistent with NIG distribution properties

### Conformal Prediction

**Current Implementation:**
- Conformal quantile computed on validation set using normalized residuals
- Applied to aggregated MoNIG prediction only
- Provides coverage-guaranteed intervals (95% target)

**Per-Engine Conformal Prediction:**
- **Currently**: Not implemented
- **Possible**: Yes, can be added by:
  1. Computing separate quantiles for each engine on validation set
  2. Storing per-engine quantiles
  3. Computing per-engine conformal intervals during inference
- **Use Case**: Individual engine calibration and comparison

## Example Results

Typical performance on PDBbind subset (test set):

| Model | MAE ↓ | RMSE ↓ | Corr ↑ | R² ↑ | Mean Epistemic | Mean Aleatoric |
| --- | --- | --- | --- | --- | --- | --- |
| GNINA expert only | 0.976 | 1.318 | 0.737 | 0.537 | 0.996 | 0.077 |
| BIND expert only | 0.993 | 1.325 | 0.734 | 0.532 | 1.423 | 0.104 |
| **MoNIG (all experts)** | **0.965** | **1.301** | **0.747** | **0.549** | 0.146 | 0.241 |

**Conformal Prediction:**
- PICP (Coverage): ~0.98 (target: 0.95)
- Average interval width: ~5.47
- Standard 95% interval coverage: ~0.90

MoNIG achieves better accuracy while providing well-calibrated uncertainty estimates.

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
│   ├── best_<MODEL>_emb_calibrator.pkl   # Isotonic calibrator
│   └── best_<MODEL>_emb_conformal.npz   # Conformal quantile
└── test_uncertainty_*.png           # Example calibration plots
```

## Extending the Pipeline

### To add more experts:

1. Add expert score columns to your CSV (e.g., `Expert5_Score`)
2. Update `src/drug_dataset_emb.py` to include the new expert columns in `self.expert_cols`
3. The model will automatically create calibrators and evidential heads for all experts

### To use different embeddings:

1. Update the embedding column names/indices in `src/drug_dataset_emb.py`
2. Adjust `embedding_dim` hyperparameter accordingly

### To implement per-engine conformal prediction:

1. Modify `compute_conformal_quantile()` to compute per-engine quantiles
2. Store per-engine quantiles in conformal.npz file
3. Update inference to compute per-engine conformal intervals
4. Add per-engine conformal columns to output CSV

## Notes

- **Data Leakage Prevention**: Normalization statistics are computed from training data only and reused for all splits
- **Reproducibility**: Use `--seed` flag for consistent train/valid/test splits
- **Model Checkpoints**: Always keep the `.pt`, `.npz`, and `.pkl` files together for consistent inference
- **Calibration**: Isotonic recalibration is optional but recommended for better uncertainty estimates
- **Conformal Prediction**: Currently applied to aggregated prediction only; per-engine CP can be added
- **Reliability Scaling**: Happens BEFORE aggregation, ensuring unreliable experts contribute less to final prediction

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
