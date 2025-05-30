# 📋 Iterative Contrastive Learning Pipeline - Implementation Plan

## 🎯 **Project Overview**
Build an iterative pipeline that:
1. Uses contrastive learning to compress multiple embeddings per class into single representatives
2. Evaluates representatives against predictions using cosine similarity
3. Extracts false positives and adds them back to training data
4. Iterates until convergence for improved class prototypes

---

## 🔄 **Iterative Development Strategy**

### **Milestone 1: Basic Data Pipeline** ⭐ *START HERE*
**Goal**: Load data, validate, and create basic end-to-end flow
**Time**: 1-2 hours
**Output**: Working data loading with validation

#### Tasks:
- [ ] Create `load_and_validate_data()` function
- [ ] Validate column names match specifications:
  - `definitiveObjects`: `['class', 'finetuned_embedding']` from full columns
  - `resnetPredictions`: `['video', 'frame', 'owl_label', 'finetuned_embedding']` 
  - `trackingInfo`: `['video', 'tag', 'actual']`
- [ ] Print data statistics and sample distributions
- [ ] **Test**: Load your actual data files and verify structure

```python
# Expected output:
✅ Loaded definitiveObjects: (X, 6) → using ['class', 'finetuned_embedding']
✅ Loaded resnetPredictions: (Y, 6) → using ['video', 'frame', 'owl_label', 'finetuned_embedding']  
✅ Loaded trackingInfo: (Z, 7) → using ['video', 'tag', 'actual']
```

---

### **Milestone 2: Single Iteration Training** ⭐ *CORE FUNCTIONALITY*
**Goal**: Complete one full training → evaluation → FP extraction cycle
**Time**: 2-3 hours
**Output**: Working single iteration with all components

#### Phase 2A: Contrastive Training Integration
- [ ] Adapt existing `train_representatives.py` for new data format
- [ ] Train on `definitiveObjects[['class', 'finetuned_embedding']]`
- [ ] Output `representatives.pkl` with single embedding per class
- [ ] **Test**: Verify representatives DataFrame format

#### Phase 2B: Evaluation Logic (from results_analysis.py)
- [ ] Load representatives as "model"
- [ ] Generate predictions on `resnetPredictions` via cosine similarity
- [ ] Apply threshold filtering and deduplication
- [ ] **Test**: Verify predictions generate correctly

#### Phase 2C: Ground Truth Comparison
- [ ] Create ground truth lookup from `trackingInfo`
- [ ] Generate TP/FP/FN/TN matrix for all video-class combinations
- [ ] Calculate F1, precision, recall
- [ ] **Test**: Manual verification of a few TP/FP cases

#### Phase 2D: False Positive Extraction
- [ ] Extract FP cases (predicted present, actually absent)
- [ ] Map back to `resnetPredictions` embeddings
- [ ] **📍 Track scene/frame information for data exclusion**
- [ ] Create new training data = original + FP embeddings
- [ ] **Save exclusion list for future iterations**
- [ ] **Test**: Verify FP embeddings have correct class labels (owl_label)

```python
# Expected output for Milestone 2:
Iteration 1 Results:
📊 F1: 0.7234, Precision: 0.8123, Recall: 0.6456
🚨 False Positives: 45 cases extracted
🔄 Updated training data: 1250 → 1295 samples (+45)
📍 Exclusion list: 45 video-frame pairs to exclude from future evaluation
```

---

### **🚫 Data Integrity & Exclusion Strategy**

#### **Problem**: Train-Test Data Leakage
- When we add false positives to training data, those same video-frame pairs should be excluded from future evaluation
- **Rule**: If we train on it → don't test on it

#### **Solution**: Cumulative Exclusion Tracking
```python
# Track all video-frame pairs used for training
exclusion_tracker = {
    'iteration_1': [
        {'video': 'scene_001.mp4', 'frame': 1250, 'class': 'mirror', 'reason': 'false_positive'},
        {'video': 'scene_002.mp4', 'frame': 890, 'class': 'lamp', 'reason': 'false_positive'},
        # ...
    ],
    'iteration_2': [
        # Additional exclusions from iteration 2
    ]
}

# Configurable exclusion behavior
if args.exclude_training_from_eval:
    # OPTION 1: Clean evaluation (recommended for research)
    evaluation_data = resnetPredictions[
        ~resnetPredictions[['video', 'frame']].apply(tuple, axis=1).isin(exclusion_set)
    ]
    print(f"🚫 Excluded {len(exclusion_set)} training samples from evaluation")
else:
    # OPTION 2: Keep all data (useful for debugging/analysis)
    evaluation_data = resnetPredictions.copy()
    print(f"⚠️  Including training data in evaluation (contaminated metrics)")
```

#### **🔧 Configurable Exclusion Strategy**:

**Option 1: Clean Evaluation** (Default - Recommended)
```bash
--exclude-training-from-eval  # Default behavior
```
- ✅ **Research integrity**: Clean train/test separation
- ✅ **True performance**: F1 reflects generalization
- ✅ **Publication ready**: Results are defensible
- ❌ **Smaller eval set**: Evaluation data shrinks over iterations

**Option 2: Contaminated Evaluation** 
```bash
--include-training-in-eval    # Keep training data in evaluation
```
- ✅ **Consistent eval set**: Same data every iteration
- ✅ **Debugging friendly**: Can verify model learns training examples
- ✅ **Full data utilization**: No data "wasted"
- ❌ **Inflated metrics**: F1 scores will be artificially high
- ❌ **Not publication ready**: Results are contaminated

**Option 3: Hybrid Approach**
```bash
--track-training-separately   # Evaluate both clean + contaminated
```
- Report **two sets of metrics**: clean and contaminated
- Best of both worlds for analysis

---

### **Milestone 3: Multi-Iteration Loop** ⭐ *SCALING UP*
**Goal**: Full iterative pipeline with convergence checking
**Time**: 1-2 hours  
**Output**: Complete pipeline running multiple iterations

#### Tasks:
- [ ] Wrap single iteration in loop (max 5 iterations)
- [ ] Track metrics across iterations
- [ ] Implement convergence checking (F1 improvement < 0.001)
- [ ] Save iteration-specific results
- [ ] **🚫 Implement cumulative exclusion logic**
- [ ] **Filter evaluation data before each iteration**
- [ ] **Test**: Run 3-iteration pipeline and verify F1 progression

```python
# Expected output:
🔄 ITERATION 1: F1: 0.7234, Eval samples: 18328
📍 Added 45 exclusions, Total excluded: 45
🔄 ITERATION 2: F1: 0.7456 (+0.0222), Eval samples: 18283 (-45)
📍 Added 23 exclusions, Total excluded: 68  
🔄 ITERATION 3: F1: 0.7467 (+0.0011), Eval samples: 18260 (-68)
🎯 Converged after 3 iterations (improvement < 0.001)

# With --exclude-training-from-eval (Clean - Default):
🔄 ITERATION 1: F1: 0.7234, Eval samples: 18328 (clean)
🚫 Excluded 0 training samples from evaluation
🔄 ITERATION 2: F1: 0.7456, Eval samples: 18283 (clean, -45 excluded)
🚫 Excluded 45 training samples from evaluation
🔄 ITERATION 3: F1: 0.7467, Eval samples: 18260 (clean, -68 excluded)

# With --include-training-in-eval (Contaminated):
🔄 ITERATION 1: F1: 0.7234, Eval samples: 18328 (all data)
⚠️  Including training data in evaluation (contaminated metrics)
🔄 ITERATION 2: F1: 0.7856, Eval samples: 18328 (all data, contaminated)
⚠️  Including training data in evaluation (contaminated metrics)
🔄 ITERATION 3: F1: 0.8012, Eval samples: 18328 (all data, contaminated)

# With --track-training-separately (Both):
🔄 ITERATION 1: 
   Clean F1: 0.7234 (18328 samples)
   Contaminated F1: 0.7234 (18328 samples)
🔄 ITERATION 2:
   Clean F1: 0.7456 (18283 samples, -45 excluded)  
   Contaminated F1: 0.7856 (18328 samples, +45 training)
🔄 ITERATION 3:
   Clean F1: 0.7467 (18260 samples, -68 excluded)
   Contaminated F1: 0.8012 (18328 samples, +68 training)
```

---

### **Milestone 4: Enhanced Features** ⭐ *POLISH & ROBUSTNESS*
**Goal**: Add advanced features and comprehensive logging
**Time**: 2-3 hours
**Output**: Production-ready pipeline with analysis

#### Phase 4A: Advanced Evaluation
- [ ] Cross-class confusion analysis (predicted vs owl_label)
- [ ] Both threshold-filtered AND all-predictions evaluation
- [ ] Multiple FP selection strategies (high_confidence, all_predictions, mixed)
- [ ] **Test**: Compare different FP strategies

#### Phase 4B: Comprehensive Output
- [ ] Save detailed CSVs for each iteration
- [ ] Pipeline summary with metrics progression
- [ ] Visualization: F1 curves, t-SNE plots
- [ ] **Test**: Verify all output files are generated correctly

#### Phase 4C: Error Handling & Validation
- [ ] Handle edge cases (no FPs, convergence on iteration 1)
- [ ] Validate embedding dimensions consistency
- [ ] Memory optimization for large datasets
- [ ] **Test**: Run with edge case scenarios

---

## 📁 **File Structure**

```
iterative_contrastive_pipeline.py    # Main script
results/
├── iteration_1/
│   ├── representatives.pkl          # Learned prototypes
│   ├── evaluation_results.csv       # TP/FP/FN/TN matrix
│   ├── false_positives.csv         # FPs extracted
│   ├── training_results.json       # Contrastive learning metrics
│   ├── exclusions.json             # 🚫 Video-frame pairs added to training
│   └── visualizations/             # Plots
├── iteration_2/
│   └── ...
├── pipeline_summary.json           # Overall results
├── metrics_comparison.csv          # F1 progression
└── cumulative_exclusions.json      # 🚫 All excluded video-frame pairs
```

---

## 🧪 **Testing Strategy**

### **Unit Tests** (build as you go):
1. **Data Loading**: Verify column extraction and validation
2. **Representative Training**: Check output format and dimensions
3. **Prediction Generation**: Verify cosine similarity calculations
4. **Ground Truth Mapping**: Test TP/FP classification logic
5. **FP Extraction**: Ensure correct embeddings and labels

### **Integration Tests** (after each milestone):
1. **End-to-End Single Iteration**: Complete cycle works
2. **Multi-Iteration Convergence**: Metrics improve over iterations  
3. **Real Data Validation**: Works with your actual datasets

### **Edge Case Tests** (Milestone 4):
1. **No False Positives**: Pipeline handles gracefully
2. **Early Convergence**: Stops at iteration 1
3. **Large Dataset**: Memory and performance optimization

---

## 🚀 **Quick Start Commands**

```bash
# Milestone 1: Test data loading
python iterative_contrastive_pipeline.py \
    --definitive-objects definitiveObjects.pkl \
    --resnet-predictions resnetPredictions.pkl \
    --tracking-info trackingInfo.pkl \
    --iterations 1 \
    --test-mode

# Milestone 2: Single iteration
python iterative_contrastive_pipeline.py \
    --definitive-objects definitiveObjects.pkl \
    --resnet-predictions resnetPredictions.pkl \
    --tracking-info trackingInfo.pkl \
    --iterations 1 \
    --threshold 0.6

# Milestone 3: Full pipeline
python iterative_contrastive_pipeline.py \
    --definitive-objects definitiveObjects.pkl \
    --resnet-predictions resnetPredictions.pkl \
    --tracking-info trackingInfo.pkl \
    --iterations 5 \
    --threshold 0.6 \
    --convergence-threshold 0.001

# Milestone 4: Advanced features
python iterative_contrastive_pipeline.py \
    --definitive-objects definitiveObjects.pkl \
    --resnet-predictions resnetPredictions.pkl \
    --tracking-info trackingInfo.pkl \
    --iterations 5 \
    --threshold 0.6 \
    --fp-strategy mixed \
    --cross-class-analysis \
    --fp-limit 1000

# 🔧 EXCLUSION STRATEGY OPTIONS:

# Option 1: Clean evaluation (default - recommended for research)
python iterative_contrastive_pipeline.py \
    --definitive-objects definitiveObjects.pkl \
    --resnet-predictions resnetPredictions.pkl \
    --tracking-info trackingInfo.pkl \
    --iterations 5 \
    --threshold 0.6 \
    --exclude-training-from-eval

# Option 2: Keep training data in evaluation (useful for debugging)
python iterative_contrastive_pipeline.py \
    --definitive-objects definitiveObjects.pkl \
    --resnet-predictions resnetPredictions.pkl \
    --tracking-info trackingInfo.pkl \
    --iterations 5 \
    --threshold 0.6 \
    --include-training-in-eval

# Option 3: Track both clean and contaminated metrics
python iterative_contrastive_pipeline.py \
    --definitive-objects definitiveObjects.pkl \
    --resnet-predictions resnetPredictions.pkl \
    --tracking-info trackingInfo.pkl \
    --iterations 5 \
    --threshold 0.6 \
    --track-training-separately
```

---

## ✅ **Success Criteria**

### **Milestone 1 Success**:
- [ ] All 3 data files load without errors
- [ ] Column validation passes
- [ ] Data statistics print correctly

### **Milestone 2 Success**:
- [ ] Representatives.pkl generates with correct format
- [ ] F1 score calculated and reasonable (> 0.5)
- [ ] False positives extracted and mapped correctly
- [ ] Training data augmented successfully
- [ ] **🚫 Video-frame exclusion list generated and saved**

### **Milestone 3 Success**:
- [ ] Multiple iterations complete without errors
- [ ] F1 score improves or stays stable across iterations
- [ ] Convergence detection works
- [ ] All iteration results saved
- [ ] **🚫 Evaluation data properly filtered each iteration**
- [ ] **🚫 Cumulative exclusion tracking works correctly**

### **Milestone 4 Success**:
- [ ] Cross-class analysis provides insights
- [ ] Multiple FP strategies work
- [ ] Comprehensive outputs generated
- [ ] Pipeline handles edge cases gracefully

---

## 🔧 **Development Notes**

### **Key Integration Points**:
1. **Contrastive Learning**: Use existing `src/trainer.py` and `src/data_loader.py`
2. **Evaluation Logic**: Copy patterns from `scripts/results_analysis.py`
3. **Data Formats**: Ensure DataFrame compatibility throughout pipeline

### **Performance Considerations**:
- Cosine similarity computation can be vectorized
- Consider embedding dimension (2048) × number of classes
- Memory usage scales with number of FPs added

### **Common Pitfalls**:
- FP extraction: Use TRUE class (owl_label), not predicted class
- Ground truth scope: Evaluate ALL video-class combinations
- Convergence: Track F1 improvement, not absolute F1
- Data leakage: Don't use same embeddings for training and evaluation
- **🚫 Train-test contamination**: Always exclude video-frame pairs used in training from evaluation

---

## 📊 **Expected Timeline**

- **Week 1**: Milestones 1-2 (Basic functionality)
- **Week 2**: Milestone 3 (Full pipeline)  
- **Week 3**: Milestone 4 (Polish & analysis)
- **Week 4**: Testing, optimization, documentation

**Total Estimated Time**: 15-20 hours over 3-4 weeks

---

*Last Updated: [Current Date]*
*Next Review: After Milestone 2 completion*
