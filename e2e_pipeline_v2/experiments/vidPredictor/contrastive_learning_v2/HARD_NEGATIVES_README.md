# Hard Negative Training for Contrastive Learning

This document describes the enhanced contrastive learning implementation that incorporates hard negative mining from false positives to improve model performance.

## 🎯 Overview

The hard negative training system addresses a key limitation in contrastive learning: **false positives become hard negatives**. When your model incorrectly predicts a class with high confidence, those samples represent the most challenging negative examples for that class. By incorporating these into training, we can significantly improve the model's ability to distinguish between similar classes.

## 📁 New Files Added

### Core Implementation
- **`src/trainer.py`** - Enhanced with hard negative methods:
  - `set_hard_negatives()` - Load hard negatives for training
  - `train_step_with_hard_negatives()` - Training step with hard negative loss
  - `load_representatives_from_dataframe()` - Load representatives from DataFrame
  - `save_representatives_to_dataframe()` - Save representatives to DataFrame

### Scripts
- **`scripts/extract_hard_negatives.py`** - Extract hard negatives from false positives CSV
- **`scripts/validate_hard_negatives.py`** - Validate that hard negatives are being pushed away
- **`train_with_hard_negatives.py`** - Main training script with hard negatives
- **`example_workflow.py`** - Complete workflow demonstration

## 🔄 Workflow

### Step 1: Extract Hard Negatives
```bash
python scripts/extract_hard_negatives.py \
    --false_positives_csv analysis_results_false_positives_all.csv \
    --resnet_data path/to/resnet_embeddings.pkl \
    --output_path hard_negatives.pkl \
    --min_confidence 0.7 \
    --max_per_class 30
```

**What it does:**
- Loads false positives from your results analysis
- Filters by confidence threshold (higher = harder negatives)
- Extracts corresponding embeddings from ResNet data
- Groups by ground truth class
- Saves as structured hard negatives dictionary

### Step 2: Train with Hard Negatives
```bash
python train_with_hard_negatives.py \
    --representatives_data initial_representatives.pkl \
    --hard_negatives hard_negatives.pkl \
    --output_dir results/hard_negative_training \
    --epochs 100 \
    --lambda_hard 0.5
```

**What it does:**
- Loads initial representatives and hard negatives
- Trains using enhanced loss: `L_total = L_standard + λ_hard * L_hard`
- `L_hard` pushes hard negatives beyond the margin
- Saves trained representatives and training history

### Step 3: Validate Results
```bash
python scripts/validate_hard_negatives.py \
    --representatives_before initial_representatives.pkl \
    --representatives_after results/hard_negative_training/trained_representatives.pkl \
    --hard_negatives hard_negatives.pkl \
    --output_path validation_results.csv
```

**What it does:**
- Compares similarities before and after training
- Shows which hard negatives were successfully pushed away
- Provides per-class improvement statistics

### Complete Workflow
```bash
python example_workflow.py \
    --false_positives_csv analysis_results_false_positives_all.csv \
    --resnet_data path/to/resnet_embeddings.pkl \
    --representatives_data initial_representatives.pkl \
    --output_dir results/complete_workflow
```

## 🧮 Technical Details

### Enhanced Loss Function
The total loss combines standard contrastive loss with hard negative loss:

```
L_total = L_standard + λ_hard * L_hard

where:
L_standard = standard contrastive loss (pull + push)
L_hard = Σ max(0, sim(rep_i, hard_neg_j) - margin)
λ_hard = weight for hard negative loss (default: 0.5)
```

### Hard Negative Selection
Hard negatives are selected based on:
1. **High confidence** - These are the model's most confident mistakes
2. **False positive status** - Predicted class ≠ ground truth class
3. **Per-class limits** - Prevent class imbalance in hard negatives

### Data Structure
Hard negatives are stored as:
```python
{
    "ground_truth_class": [
        {
            "embedding": torch.Tensor,
            "video": str,
            "frame": int,
            "predicted_class": str,
            "confidence": float
        },
        ...
    ],
    ...
}
```

## 📊 Expected Results

### Good Signs
- **Validation improvement rate > 70%** - Most hard negatives are pushed away
- **Average similarity decrease > 0.05** - Meaningful separation improvement
- **Consistent per-class improvements** - All classes benefit

### Warning Signs
- **Low improvement rate < 50%** - May need different hyperparameters
- **Negative improvements** - Some similarities increased (investigate these cases)
- **High hard negative loss** - λ_hard might be too high

## 🎛️ Hyperparameter Tuning

### Key Parameters

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `min_confidence` | 0.7 | Minimum confidence for hard negatives | Higher = harder negatives, fewer samples |
| `max_per_class` | 30 | Maximum hard negatives per class | Balance between diversity and training time |
| `lambda_hard` | 0.5 | Weight for hard negative loss | Start low (0.1-0.5), increase if needed |
| `epochs` | 100 | Training epochs | Monitor validation F1 for early stopping |

### Tuning Strategy
1. **Start conservative**: `min_confidence=0.8`, `lambda_hard=0.3`
2. **Monitor validation**: Look for F1 improvement and loss convergence
3. **Adjust gradually**: Increase `lambda_hard` if hard negatives aren't improving
4. **Check balance**: Ensure hard negative loss doesn't dominate standard loss

## 🔍 Debugging

### Common Issues

**1. No hard negatives found**
- Check false positives CSV has correct columns
- Verify confidence threshold isn't too high
- Ensure ResNet data matches false positives

**2. Hard negatives not improving**
- Increase `lambda_hard`
- Check if margin is appropriate
- Verify hard negatives are correctly loaded

**3. Training instability**
- Reduce `lambda_hard`
- Check for NaN values in embeddings
- Monitor loss components separately

### Validation Checks
```python
# Check hard negatives structure
with open('hard_negatives.pkl', 'rb') as f:
    hn = pickle.load(f)
    print(f"Classes: {list(hn.keys())}")
    print(f"Total negatives: {sum(len(v) for v in hn.values())}")

# Check similarity improvements
df = pd.read_csv('validation_results.csv')
print(f"Improvement rate: {df['improved'].mean():.1%}")
print(f"Average improvement: {df['improvement'].mean():.4f}")
```

## 🚀 Integration with Existing Pipeline

### Before Training
1. Run your existing results analysis to get false positives CSV
2. Ensure you have the ResNet embeddings pickle file
3. Have initial representatives ready

### After Training
1. Use trained representatives for new predictions
2. Re-run results analysis to measure improvement
3. Compare false positive rates before/after

### Iterative Improvement
1. **First iteration**: Use initial false positives
2. **Subsequent iterations**: Use false positives from improved model
3. **Convergence**: When false positive rate stops decreasing

## 📈 Performance Expectations

Based on the implementation, you should expect:
- **10-30% reduction** in false positive rate
- **5-15% improvement** in overall F1 score
- **Better class separation** in embedding space
- **More confident correct predictions**

The exact improvements depend on:
- Quality of initial representatives
- Diversity of hard negatives
- Hyperparameter tuning
- Dataset characteristics 