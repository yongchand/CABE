# MoNIG Improved Models - Performance Enhancement Guide

## 📊 **Current Performance Gap**

**Problem Identified:**
- **MoNIG_Improved** has the best **MAE (0.8191)** but **RMSE (1.0370)** is slightly worse than **UniformReliability (1.0326)**
- **RMSE > MAE** indicates some predictions have large errors
- **Correlation** is also slightly lower (0.8457 vs 0.8460)

**Root Cause:**
- Learned reliability might be **too aggressive** on certain samples
- When reliability is low (e.g., r_j=0.1), current scaling (0.5+0.5*r_j) still reduces scale to 0.55
- This can cause issues if the reliability estimation is incorrect on outliers

---

## 🎯 **Three New Model Variants**

### **1. MoNIG_Improved_v2 - More Conservative Scaling** ⭐ **Recommended First**

**Key Change:**
```python
# Original MoNIG_Improved:
scale_factor = 0.5 + 0.5 * r_j  # Range: [0.5, 1.0]

# New MoNIG_Improved_v2:
scale_factor = 0.7 + 0.3 * r_j  # Range: [0.7, 1.0]
```

**Scaling Comparison:**

| Reliability (r_j) | Improved v1 | Improved v2 |
|-------------------|-------------|-------------|
| 0.0 (very uncertain) | 0.50 | **0.70** ✅ |
| 0.25 | 0.62 | **0.77** ✅ |
| 0.50 | 0.75 | **0.85** ✅ |
| 0.75 | 0.88 | **0.92** ✅ |
| 1.0 (very certain) | 1.00 | 1.00 |

**Expected Benefits:**
- ✅ **Better RMSE** - less aggressive scaling reduces large errors
- ✅ **Better Correlation** - maintains more original signal
- ✅ **Similar MAE** - should remain competitive with v1

---

### **2. MoNIG_Hybrid - Blend Learned + Uniform**

**Key Change:**
```python
# Compute hybrid reliability
uniform_r = 1.0 / num_experts  # 0.25 for 4 experts
hybrid_r = 0.5 * learned_r + 0.5 * uniform_r

# Then apply standard soft scaling
scale_factor = 0.5 + 0.5 * hybrid_r
```

**Reliability Blending:**

| Learned r_j | Uniform r | Hybrid r | Scale Factor |
|-------------|-----------|----------|--------------|
| 0.0 | 0.25 | **0.125** | 0.5625 |
| 0.5 | 0.25 | **0.375** | 0.6875 |
| 1.0 | 0.25 | **0.625** | 0.8125 |

**Expected Benefits:**
- ✅ **Balanced** - combines adaptiveness (learned) + stability (uniform)
- ✅ **Robust** - less sensitive to reliability estimation errors
- ✅ **Consistent** - should have low variance across seeds

---

### **3. Comparison Table**

| Model | Scaling Formula | Scale Range | Strategy |
|-------|----------------|-------------|----------|
| **MoNIG_Improved** | 0.5 + 0.5*r | [0.5, 1.0] | Moderate |
| **MoNIG_Improved_v2** | 0.7 + 0.3*r | [0.7, 1.0] | Conservative ✅ |
| **MoNIG_Hybrid** | 0.5 + 0.5*(0.5*r + 0.125) | [0.5625, 0.8125] | Balanced |
| **UniformReliability** | N/A (r=0.25) | Fixed | Reference |

---

## 🚀 **Quick Testing (10 epochs)**

Test all three models quickly to see which direction is promising:

```bash
# Test v2 (conservative scaling)
python main.py train \
  --model_type MoNIG_Improved_v2 \
  --csv_path pdbbind_descriptors_with_experts_and_binding.csv \
  --epochs 10 \
  --optimizer adam \
  --device cuda:0 \
  --output_path test_v2_quick

# Test Hybrid (blended reliability)
python main.py train \
  --model_type MoNIG_Hybrid \
  --csv_path pdbbind_descriptors_with_experts_and_binding.csv \
  --epochs 10 \
  --optimizer adam \
  --device cuda:0 \
  --output_path test_hybrid_quick
```

**What to Look For:**
- Val MAE trend (should be decreasing steadily)
- Final val MAE (ideally < 0.90 after 10 epochs)

---

## 🔬 **Full Comparison (150 epochs, 3 seeds)**

Once you've identified promising candidates:

```bash
# Run full ablation for all three improved models
python run_ablation_experiments.py \
  --ablation_types MoNIG_Improved MoNIG_Improved_v2 MoNIG_Hybrid \
  --seeds 42 43 44 \
  --epochs 150 \
  --optimizer adam \
  --device cuda:0 \
  --output_dir experiments_improved_comparison
```

**Expected Results:**

| Model | Expected MAE | Expected RMSE | Expected Corr |
|-------|--------------|---------------|---------------|
| **MoNIG_Improved** | **0.8191** ✅ | 1.0370 | 0.8457 |
| **MoNIG_Improved_v2** | ~0.82 | **< 1.033** ✅ | **> 0.846** ✅ |
| **MoNIG_Hybrid** | ~0.82 | ~1.035 | ~0.846 |
| **Target (Best)** | **< 0.82** | **< 1.033** | **> 0.846** |

---

## 📈 **Analysis Commands**

After experiments complete:

```bash
# Quick view results
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('experiments_improved_comparison/ablation_results_*.csv')

print("Results by Model:")
for model in df['ablation_type'].unique():
    subset = df[df['ablation_type'] == model]
    print(f"\n{model}:")
    print(f"  MAE:  {subset['test_mae'].mean():.4f} ± {subset['test_mae'].std():.4f}")
    print(f"  RMSE: {subset['test_rmse'].mean():.4f} ± {subset['test_rmse'].std():.4f}")
    print(f"  Corr: {subset['test_corr'].mean():.4f} ± {subset['test_corr'].std():.4f}")
EOF
```

---

## 🎯 **Decision Criteria**

**Choose MoNIG_Improved_v2 if:**
- ✅ RMSE < 1.033 (better than UniformReliability)
- ✅ Correlation > 0.846
- ✅ MAE remains < 0.825

**Choose MoNIG_Hybrid if:**
- ✅ Most stable (lowest std across seeds)
- ✅ Balanced performance across all metrics
- ✅ Lower variance than v2

**Stay with MoNIG_Improved if:**
- ❌ Both v2 and Hybrid fail to improve RMSE
- ❌ MAE degrades significantly (> 0.83)

---

## 💡 **Alternative Approaches (If Above Don't Work)**

If neither v2 nor Hybrid improves RMSE significantly:

### **Option A: Loss Function Modification**
Add Huber loss or weighted MSE to reduce sensitivity to outliers:

```python
# In train_drug_discovery_emb.py
def huber_nig_loss(mu, v, alpha, beta, target, delta=1.0):
    error = target - mu
    # Huber loss on mean prediction
    huber_term = torch.where(
        torch.abs(error) <= delta,
        0.5 * error ** 2,
        delta * (torch.abs(error) - 0.5 * delta)
    )
    # NIG uncertainty term
    nig_term = ... # existing NIG loss
    return huber_term + nig_term
```

### **Option B: Ensemble**
Combine predictions from multiple improved models:

```python
# Weighted ensemble
pred_final = 0.5 * pred_improved + 0.3 * pred_v2 + 0.2 * pred_hybrid
```

### **Option C: Post-hoc Calibration**
Train a small calibration network on validation set to correct predictions.

---

## 📊 **All Model Files Updated**

✅ `src/drug_models_emb.py` - Added `DrugDiscoveryMoNIG_Improved_v2` and `DrugDiscoveryMoNIG_Hybrid`
✅ `src/train_drug_discovery_emb.py` - Added training support
✅ `src/inference_drug_discovery.py` - Added inference support
✅ `main.py` - Added to command-line arguments
✅ `run_ablation_experiments.py` - Added to ablation experiment runner
✅ `test_new_models.py` - Validation script

---

## 🎯 **Recommended Next Steps**

1. **Quick Test (10 minutes):**
   ```bash
   # Test both new models with 10 epochs
   python main.py train --model_type MoNIG_Improved_v2 --epochs 10 --device cuda:0
   python main.py train --model_type MoNIG_Hybrid --epochs 10 --device cuda:0
   ```

2. **If promising, run full comparison (2-3 hours):**
   ```bash
   python run_ablation_experiments.py \
     --ablation_types MoNIG_Improved MoNIG_Improved_v2 MoNIG_Hybrid \
     --seeds 42 43 44 \
     --epochs 150 \
     --device cuda:0
   ```

3. **Analyze and choose the best:**
   - Compare MAE, RMSE, Correlation
   - Choose the model with best **overall balance**
   - Publish results!

---

## 🎉 **Expected Outcome**

**Goal:** Achieve **all three** metrics better than current best:
- MAE < 0.82 (currently 0.8191)
- RMSE < 1.033 (currently 1.0370)
- Correlation > 0.846 (currently 0.8457)

**Most Likely Winner:** **MoNIG_Improved_v2** 🥇
- More conservative scaling should reduce large errors (better RMSE)
- Maintains learned reliability's adaptiveness (good MAE)
- Same parameter count as MoNIG_Improved (213K)

Good luck! 🚀


