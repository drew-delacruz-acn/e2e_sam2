# Team Handoff Guide - Contrastive Learning Project


## 🎯 Project Summary

**What this does**: Learns better class representatives for object classification using contrastive learning. Instead of using simple class averages, this trains embeddings that are closer to same-class samples and farther from different-class samples.

## ✅ What's Currently Working

### **Core Training Pipeline** 
- ✅ Basic contrastive learning (`train_representatives.py`) - READY FOR PRODUCTION
- ✅ Multiple initialization methods (class_means, random, bounded_random, perturbed_means)
- ✅ Automatic parameter-based folder naming
- ✅ Comprehensive test suite (pytest) - all tests passing
- ✅ Data loading and validation for PKL format

### **Advanced Features**
- ✅ Parameter sweep automation (`scripts/parameter_sweep.py`) - finds best settings
- ✅ Iterative pipeline (`run_iterative_pipeline.py`) - adds false positives back to training
- ✅ Exclusion strategy analysis - frame-level vs video-level filtering
- ✅ Visualization tools (t-SNE plots, loss curves, HTML diagrams)

### **Analysis Tools**
- ✅ Exclusion impact analyzer - quantifies data waste
- ✅ Results analysis scripts - confusion matrices, F1 comparisons
- ✅ Performance tracking across iterations

## 🆕 Recent Major Improvements (2025)

### **New Features**
- ✅ **Per-Class FP Capping**: `--max-fps-per-class` parameter prevents excessive dataset shrinkage with video-level exclusions
- ✅ **Enhanced Debug Logging**: Complete pipeline decisions saved to `analysis_logs/debug_log_*.txt` files for VM analysis
- ✅ **Dynamic Ground Truth Filtering**: Evaluation properly excludes training data based on exclusion strategy
- ✅ **Robust Deduplication**: Sophisticated embedding-based deduplication works with any input data format


## 🔄 Iterative Pipeline
This pipeline runs the contrastive learning process, filters out the false positives and retrains the contrastive learner with these added to the dataset

# Example command (WORKS)
python run_iterative_pipeline.py \
--definitive-objects /home/ubuntu/code/drew/e2e_sam2/data/definitiveObjects_jeremiah.pkl \ ##DREW TO CHANGE THESE TO DATAFRAMES (COLS: CLASS/EMBEDDINGS)
--resnet-predictions /home/ubuntu/code/libby/pipeline/data/finetuned_may19.pkl \ ####DREW TO CHANGE THESE TO DATAFRAMES 
--tracking-info /home/ubuntu/code/libby/pipeline/data/sourceTruth_jeremiah.pkl \
--iterations 2 \
--epochs 50 \
--threshold 0.8 \
--margin 0.2 \
--output NO_FPS_CAP \
--exclude-training-from-eval \
--video-level-exclusions \
--max-fps-per-class 2



### **What It Does**
- Automatically learns from classification mistakes


### **Current Status: PRODUCTION READY**
- ✅ **Core Pipeline**: Fully functional, tested on multiple datasets
- ✅ **Exclusion Strategies**: Frame-level vs video-level comparison working
- ✅ **Adaptive Parameters**: Secondary thresholds/margins for later iterations
- ✅ **Comprehensive Logging**: Detailed analysis logs for debugging
- ✅ **Data Tracking**: Complete data flow monitoring system

**Requires 3 Data Files**: definitiveObjects.pkl (ground truth embeddings of objects), resnetPredictions.pkl (predictions), trackingInfo.pkl (ground truth tracking objects)

### **Production Deployment Ready**


```bash
# Standard production command
python run_iterative_pipeline.py \
    --definitive-objects ./definitiveObjects.pkl \
    --resnet-predictions ./resnetPredictions.pkl \
    --tracking-info ./trackingInfo.pkl \
    --iterations 5 \
    --threshold 0.6 \
    --secondary-threshold 0.7 \
    --margin 0.2 \
    --secondary-margin 0.25 \
    --exclusion-strategy frame-level \
    --exclude-training-from-eval

# NEW: For video-level exclusions (prevents dataset decimation)
python run_iterative_pipeline.py \
    --definitive-objects ./definitiveObjects.pkl \
    --resnet-predictions ./resnetPredictions.pkl \
    --tracking-info ./trackingInfo.pkl \
    --iterations 5 \
    --threshold 0.5 \
    --exclusion-strategy video-level \
    --max-fps-per-class 2 \
    --exclude-training-from-eval
```


```

### **Enhanced Debugging & Analysis**
```bash
# Debug logs automatically saved to analysis_logs/debug_log_TIMESTAMP.txt
# Copy from VM for detailed pipeline analysis:
ls analysis_logs/debug_log_*.txt

# Check training data growth between iterations:
grep "FINAL SUMMARY" analysis_logs/debug_log_*.txt

# Verify confusion matrix size changes:
grep "FINAL EVALUATION MATRIX" analysis_logs/debug_log_*.txt
```

### **Critical Files for Team**
- `iterative_pipeline/pipeline_manager.py` - Main orchestration logic
- `iterative_pipeline/config.py` - All configuration options
- `run_iterative_pipeline.py` - Entry point script
- **DO NOT MODIFY** core pipeline logic without extensive testing


## 📊 Current Experiment Status

### **Completed Experiments**
- ✅ Baseline contrastive learning with class means initialization
- ✅ Parameter sweep analysis (324 combinations tested)
- ✅ Initialization method comparison (4 methods)
- ✅ Exclusion strategy comparison (frame-level vs video-level)


## 📁 Important File Locations

### **Core Scripts** (use these regularly)
- `train_representatives.py` - Main training script
- `scripts/parameter_sweep.py` - Hyperparameter optimization
- `run_iterative_pipeline.py` - Advanced pipeline with false positive feedback

### **Results Storage**
- Training results: `results/` (auto-organized by parameters)
- Parameter sweeps: `sweep_results/`
- Analysis outputs: `analyze_exclusions/`, `tracking_exports/`

### **Documentation**
- Main guide: `README.md` (comprehensive, start here)
- Troubleshooting: `TROUBLESHOOTING.md`
- Interactive diagrams: `docs/html/*.html` (open in browser)
- Archived docs: `docs/` (detailed technical documentation)

## 🔧 Environment & Dependencies

### **Current Setup**
- Python environment: `../../venv/` (shared with parent project)
- Key dependencies: torch, pandas, numpy, scikit-learn, pytest
- GPU support: Automatic detection (works with CPU fallback)

### **Installation Check**
```bash
cd e2e_pipeline_v2/experiments/contrastive_learning/
source ../../venv/bin/activate
pytest  # Should pass all tests
```


## 🚨 Critical Information

### **Do NOT Change**
- Core loss function implementation (`src/loss_functions.py`)
- Data loading logic (`src/data_loader.py`) 
- Test suite configuration (`pytest.ini`, `tests/`)

### **Safe to Modify**
- Default parameters in `config.yaml`
- Analysis scripts in `scripts/`
- Visualization code (`src/visualizer.py`)
- Documentation files

### **Before Any Major Changes**
```bash
# Always run tests first
pytest -v

# Backup current results
cp -r results/ results_backup_$(date +%Y%m%d)/
```

## Team Contact & Handoff

### **External Dependencies**
- SAM2 pipeline integration points
- Embedding generation (upstream dependency)
- Classification deployment (downstream usage)

## 🛠️ Troubleshooting Guide


**Problem: Massive dataset shrinkage with video-level exclusions**
```bash
# Use FP capping to prevent decimation:
python run_iterative_pipeline.py [...] --max-fps-per-class 2
```

### **Data Format Issues**

**Mixed embedding types in definitiveObjects:**
```python
# Check your input data format:
import pickle
with open('definitiveObjects.pkl', 'rb') as f:
    data = pickle.load(f)
print("Embedding type:", type(data.iloc[0]['finetuned_embedding']))
# Should work with both list and numpy.ndarray formats now
```
