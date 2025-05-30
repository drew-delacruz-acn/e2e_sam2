# Experimental Plan: Contrastive Representative Learning

*(Quick experiment to test if learned class representatives outperform simple means)*

---

## Goal & Success Metric

**Hypothesis:** Learning class representatives with contrastive loss will give better classification than using raw class means.

**Test:** Train representatives, compare macro-F1 on validation split. If F1-reps > F1-means, experiment succeeds.

---

## Core Parameters

| Parameter     | Default | Notes |
|---------------|---------|-------|
| `epochs`      | `50`    | Keep low for quick iteration |
| `lr`          | `1e-3`  | Standard Adam rate |
| `lambda_push` | `0.25`  | Weight for negative pushing |
| `margin`      | `0.15`  | Cosine margin between classes |
| `val_frac`    | `0.3`   | Validation split |

---

## Loss Function

For each class **c** with representative **r^c**:

- **P^c** = all embeddings belonging to class c (positives)
- **N^c** = all embeddings from other classes (negatives)

$$
\mathcal L^{c} = \underbrace{-\frac{1}{|P^{c}|}\sum_{p\in P^{c}}\cos(\mathbf r^{c},\mathbf p)}_{\text{pull positives closer}} + \underbrace{\lambda \frac{1}{|N^{c}|}\sum_{n\in N^{c}} \max(0, \cos(\mathbf r^{c},\mathbf n) - m)}_{\text{push negatives beyond margin}}
$$

**Total loss** = sum over all classes: $\mathcal L = \sum_{c} \mathcal L^{c}$

---

## Experiment Steps

1. **Load data** from PKL → validate basic format
2. **Split** train/val stratified by class  
3. **Initialize** representatives as class means
4. **Train** for N epochs with Adam
5. **Evaluate** final F1 scores: baseline vs learned reps
6. **Plot** loss curve + t-SNE visualization
7. **Save** results and learned representatives

## Quick Validation Checks

- ✅ Data loads without errors
- ✅ All classes present in both splits
- ✅ Loss decreases during training
- ✅ F1-reps ≥ F1-baseline (success condition)

## Output Files

- `results.json` - F1 scores, config, final loss
- `representatives.pkl` - Learned embeddings  
- `loss_curve.png` - Training progress
- `tsne_plot.png` - Visualization of learned reps vs means

---

**Focus:** Get quick results to validate the approach. Optimize later if promising.
