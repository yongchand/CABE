# Summary of Changes: L-BFGS-B Optimizer Integration

## What Was Added

### 1. **L-BFGS-B Optimizer Support** (`src/train_drug_discovery_emb.py`)
- Added `LBFGSBOptimizer` class that wraps scipy's L-BFGS-B for PyTorch models
- L-BFGS-B is more memory efficient than SLSQP and better suited for large neural networks
- Refactored training loop to support multiple optimizers
- Added `compute_loss()` helper function to avoid code duplication
- Added command-line arguments:
  - `--optimizer {adam,lbfgs,sgd}` - Choose optimizer (default: adam)
  - `--lbfgs_maxiter` - Max iterations per L-BFGS-B step (default: 20)

### 2. **Ablation Script Updates** (`run_ablation_experiments.py`)
- Added `--optimizer` argument to test different optimizers
- Updated experiment directory naming to include optimizer
- Updated CSV output to include optimizer column
- All results now track which optimizer was used

### 3. **Comparison Tools** (`compare_optimizer_results.py`)
- New script to compare optimizer performance
- Features:
  - Statistical tests (paired t-tests)
  - Visualization plots (bar plots, box plots, scatter plots)
  - Summary tables
  - Combined CSV output

### 4. **Documentation** (`OPTIMIZER_COMPARISON_GUIDE.md`)
- Complete guide on using different optimizers
- Examples and best practices
- Expected performance comparisons
- Troubleshooting tips

### 5. **Bug Fixes**
- Fixed test dataset handling (no longer fails when test samples missing)
- Fixed requirements.txt (torch instead of pytorch, removed Windows-only packages)

## How to Use

### Quick Test (Single Model)
```bash
# Test with Adam (baseline)
python main.py train --model_type MoNIG --csv_path pdbbind_descriptors_with_experts_and_binding.csv --seed 42 --epochs 10 --optimizer adam --device cpu

# Test with L-BFGS-B
python main.py train --model_type MoNIG --csv_path pdbbind_descriptors_with_experts_and_binding.csv --seed 42 --epochs 10 --optimizer lbfgs --device cpu
```

### Full Comparison (Recommended)
```bash
# Run experiments with Adam
python run_ablation_experiments.py --seeds 42 43 44 --epochs 150 --optimizer adam --output_dir experiments_adam --device cpu

# Run experiments with L-BFGS-B
python run_ablation_experiments.py --seeds 42 43 44 --epochs 150 --optimizer lbfgs --output_dir experiments_lbfgs --device cpu

# Compare results
python compare_optimizer_results.py experiments_adam/ablation_results_*.csv experiments_lbfgs/ablation_results_*.csv
```

## Files Modified

1. `src/train_drug_discovery_emb.py` - Added SLSQP support
2. `run_ablation_experiments.py` - Added optimizer parameter
3. `requirements.txt` - Fixed package names

## Files Created

1. `compare_optimizer_results.py` - Comparison analysis script
2. `OPTIMIZER_COMPARISON_GUIDE.md` - User guide
3. `CHANGES_SUMMARY.md` - This file

## Expected Outcomes

### Performance Metrics to Compare
- **Test MAE** (Mean Absolute Error) - Lower is better
- **Test RMSE** (Root Mean Square Error) - Lower is better
- **Test Correlation** - Higher is better
- **Test R²** - Higher is better
- **Test CRPS** (Continuous Ranked Probability Score) - Lower is better
- **Test NLL** (Negative Log-Likelihood) - Lower is better
- **Test PICP@95%** (Prediction Interval Coverage) - Closer to 0.95 is better
- **Test ECE** (Expected Calibration Error) - Lower is better

### What to Expect
- **Adam**: Fast convergence, good general performance
- **L-BFGS-B**: Slower but potentially more precise, better for fine-tuning
- **SGD**: Baseline comparison, may need higher learning rate

## Next Steps

1. **Test Basic Functionality**
   ```bash
   python main.py train --model_type MoNIG --csv_path pdbbind_descriptors_with_experts_and_binding.csv --seed 42 --epochs 5 --optimizer adam --device cpu
   python main.py train --model_type MoNIG --csv_path pdbbind_descriptors_with_experts_and_binding.csv --seed 42 --epochs 5 --optimizer slsqp --device cpu
   ```

2. **Run Full Experiments** (if test works)
   ```bash
   # This will take several hours
   python run_ablation_experiments.py --seeds 42 43 44 45 46 --epochs 150 --optimizer adam --output_dir experiments_adam
   python run_ablation_experiments.py --seeds 42 43 44 45 46 --epochs 150 --optimizer slsqp --output_dir experiments_slsqp
   ```

3. **Analyze Results**
   ```bash
   python compare_optimizer_results.py experiments_adam/ablation_results_*.csv experiments_slsqp/ablation_results_*.csv
   ```

## Notes

- SLSQP is 2-3x slower per epoch than Adam
- SLSQP may give better final performance but takes longer
- Use GPU (`--device cuda`) for faster training if available
- Results are saved in separate directories per optimizer
- The comparison script requires matplotlib and seaborn (already in scipy dependencies)

## Troubleshooting

If you encounter issues:

1. **Check logs**: `cat experiments_*/*/training.log`
2. **Verify optimizer is working**: Look for "optimizer=" in output
3. **SLSQP too slow**: Reduce `--slsqp_maxiter` or `--epochs`
4. **Out of memory**: Reduce `--batch_size`

## Questions?

Refer to `OPTIMIZER_COMPARISON_GUIDE.md` for detailed documentation.

