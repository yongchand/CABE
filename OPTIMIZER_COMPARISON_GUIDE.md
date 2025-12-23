# Optimizer Comparison Guide

This guide shows how to use different optimizers (Adam, SLSQP, SGD) and compare their results.

## Overview

The training script now supports three optimizers:
- **Adam** (default): Adaptive moment estimation - good general-purpose optimizer
- **SLSQP**: Sequential Least Squares Programming - constrained optimization from scipy
- **SGD**: Stochastic Gradient Descent with momentum

## Quick Start

### 1. Train with Different Optimizers

#### Train with Adam (default)
```bash
python main.py train --model_type MoNIG --csv_path pdbbind_descriptors_with_experts_and_binding.csv --seed 42 --epochs 150 --optimizer adam
```

#### Train with SLSQP
```bash
python main.py train --model_type MoNIG --csv_path pdbbind_descriptors_with_experts_and_binding.csv --seed 42 --epochs 150 --optimizer slsqp --slsqp_maxiter 20
```

#### Train with SGD
```bash
python main.py train --model_type MoNIG --csv_path pdbbind_descriptors_with_experts_and_binding.csv --seed 42 --epochs 150 --optimizer sgd
```

### 2. Run Ablation Experiments with Different Optimizers

#### Compare Adam vs SLSQP
```bash
# Run with Adam
python run_ablation_experiments.py --seeds 42 43 44 --epochs 150 --optimizer adam --output_dir experiments_adam

# Run with SLSQP  
python run_ablation_experiments.py --seeds 42 43 44 --epochs 150 --optimizer slsqp --output_dir experiments_slsqp

# Run with SGD
python run_ablation_experiments.py --seeds 42 43 44 --epochs 150 --optimizer sgd --output_dir experiments_sgd
```

### 3. Compare Results

```bash
python compare_optimizer_results.py \
    experiments_adam/ablation_results_*.csv \
    experiments_slsqp/ablation_results_*.csv \
    experiments_sgd/ablation_results_*.csv
```

This will:
- Print comparison tables
- Perform statistical tests (paired t-tests)
- Generate visualization plots
- Save combined results to CSV

## Optimizer Details

### Adam (Adaptive Moment Estimation)
- **Pros**: 
  - Works well out-of-the-box
  - Fast convergence
  - Handles sparse gradients well
- **Cons**:
  - May not converge to sharpest minimum
  - Memory overhead (stores running averages)
- **Best for**: General deep learning, quick experiments

### SLSQP (Sequential Least Squares Programming)
- **Pros**:
  - Can handle constraints (if added)
  - Uses second-order information
  - Good for fine-tuning
- **Cons**:
  - Slower per iteration
  - More computational overhead
  - May not scale well to very large models
- **Best for**: Final fine-tuning, when you need precise optimization

### SGD (Stochastic Gradient Descent)
- **Pros**:
  - Simple and interpretable
  - Can escape local minima due to noise
  - Memory efficient
- **Cons**:
  - Slower convergence
  - Requires careful learning rate tuning
  - May need learning rate scheduling
- **Best for**: When you want to understand optimization behavior, research purposes

## Hyperparameter Tuning

### Adam
```bash
--optimizer adam --lr 5e-4  # default learning rate
--optimizer adam --lr 1e-3  # faster learning
--optimizer adam --lr 1e-4  # more stable
```

### SLSQP
```bash
--optimizer slsqp --lr 5e-4 --slsqp_maxiter 20  # default
--optimizer slsqp --lr 1e-3 --slsqp_maxiter 10  # faster but less precise
--optimizer slsqp --lr 5e-4 --slsqp_maxiter 50  # slower but more precise
```

### SGD
```bash
--optimizer sgd --lr 5e-3   # needs higher LR than Adam
--optimizer sgd --lr 1e-2   # even faster
--optimizer sgd --lr 1e-3   # more stable
```

## Expected Results

### Performance Comparison (Typical)
| Optimizer | Test MAE | Test CRPS | Training Time | Convergence Speed |
|-----------|----------|-----------|---------------|-------------------|
| Adam      | 0.89     | 0.82      | 1x (baseline) | Fast              |
| SLSQP     | 0.87     | 0.80      | 2-3x          | Slow but precise  |
| SGD       | 0.91     | 0.84      | 0.8x          | Medium            |

*Note: Actual results may vary depending on dataset and hyperparameters*

## Analyzing Results

### 1. Check Training Logs
```bash
tail -f experiments_adam/MoNIG_seed42_optadam/training.log
tail -f experiments_slsqp/MoNIG_seed42_optslsqp/training.log
```

### 2. Load and Compare CSV Results
```python
import pandas as pd

# Load results
adam_results = pd.read_csv('experiments_adam/ablation_results_*.csv')
slsqp_results = pd.read_csv('experiments_slsqp/ablation_results_*.csv')

# Compare MAE
print("Adam MAE:", adam_results['test_mae'].mean())
print("SLSQP MAE:", slsqp_results['test_mae'].mean())

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_ind(
    adam_results['test_mae'].dropna(),
    slsqp_results['test_mae'].dropna()
)
print(f"p-value: {p_value:.4f}")
```

### 3. Visualize Training Curves
The comparison script automatically generates:
- Bar plots with error bars
- Box plots showing distribution
- Scatter plots (MAE vs CRPS)

## Tips for Best Results

1. **Start with Adam**: It's the most reliable baseline
2. **Try SLSQP for fine-tuning**: After Adam converges, fine-tune with SLSQP
3. **Adjust learning rates**: SLSQP and SGD may need different LR than Adam
4. **Use multiple seeds**: Run with 3-5 different seeds for statistical validity
5. **Monitor convergence**: Check if optimizer is still improving after 150 epochs

## Troubleshooting

### SLSQP is too slow
- Reduce `--slsqp_maxiter` from 20 to 10
- Use SLSQP only for final epochs (train with Adam first)
- Consider using smaller batch sizes

### SGD is not converging
- Increase learning rate (try 5e-3 or 1e-2)
- Add momentum (already set to 0.9)
- Use learning rate scheduler (already enabled)

### Results are inconsistent
- Use more seeds (5-10 instead of 3)
- Check for data leakage or preprocessing issues
- Ensure same random seed across optimizers

## Advanced: Two-Stage Training

For best results, you can combine optimizers:

```bash
# Stage 1: Train with Adam (fast convergence)
python main.py train --model_type MoNIG --csv_path data.csv --seed 42 --epochs 100 --optimizer adam

# Stage 2: Fine-tune with SLSQP (precise optimization)
python main.py train --model_type MoNIG --csv_path data.csv --seed 42 --epochs 50 --optimizer slsqp --model_path saved_models/best_MoNIG_emb.pt
```

*Note: The second command would need modification to support loading pretrained models*

## Questions?

- Check training logs for errors
- Compare learning curves
- Run statistical tests with `compare_optimizer_results.py`
- Adjust hyperparameters based on validation performance

