# Contrastive Learning for Representative Learning

Learn optimal class representatives using contrastive learning. This project takes embedding data and creates better class prototypes than simple averages.

## 🚀 Quick Start (5 Steps)

### 1. Setup Environment
```bash
# Make sure you're using the existing venv in the parent directory
source ../../venv/bin/activate  # or activate your python environment
pip install -r requirements.txt
```

### 2. Test Installation
```bash
# Run tests to make sure everything works
pytest
```

### 3. Basic Training
```bash
# Train representatives on your data
python train_representatives.py --data path/to/your/embeddings.pkl

# Results saved to: results/init_class_means_lr_0.01_margin_0.15_lambda_0.25_epochs_50/
```

### 4. Advanced Training (Better Results)
```bash
# For better class separation (recommended for armor/clothing data)
python train_representatives.py \
    --data path/to/your/embeddings.pkl \
    --init-method random \
    --margin 0.3 \
    --lambda-push 0.6 \
    --epochs 150
```

### 5. Parameter Sweep (Find Best Settings)
```bash
# Test multiple parameter combinations automatically
python scripts/parameter_sweep.py --data path/to/your/embeddings.pkl
```

## 📊 Data Format

Your PKL file needs a pandas DataFrame with:
- `class`: String class labels (e.g., "TVA Monitor", "Classic Loki Armor")
- `finetuned_embedding`: Numpy arrays (embeddings, typically 2048D)

Example:
```python
import pandas as pd
import numpy as np

data = [
    {'class': 'TVA Monitor', 'finetuned_embedding': np.array([1.0, 2.0, ...])},
    {'class': 'Classic Loki Armor', 'finetuned_embedding': np.array([3.0, 4.0, ...])},
]
df = pd.DataFrame(data)
df.to_pickle('my_data.pkl')
```

## ⚙️ Key Parameters

| Parameter | Default | What it does | When to change |
|-----------|---------|--------------|----------------|
| `--init-method` | `class_means` | How to start training | Use `random` for better separation |
| `--margin` | `0.15` | Minimum distance between classes | Increase to `0.3` for clearer boundaries |
| `--lambda-push` | `0.25` | How hard to push classes apart | Increase to `0.6` for armor/clothing data |
| `--epochs` | `50` | Training duration | Use `150+` with random init |
| `--lr` | `0.01` | Learning speed | Usually fine as-is |

## 📈 Understanding Results

### Output Files (in results folder):
- **`representatives.pkl`** - Your trained class representatives (use this for classification)
- **`results.json`** - F1 scores and metrics
- **`loss_curve.png`** - Training progress visualization
- **`tsne_plot.png`** - How well classes are separated

### Good Results Look Like:
- F1 score improves over baseline
- Loss decreases during training
- Max similarity between classes < 0.4 (check results.json)
- Classes are well-separated in t-SNE plot

## 🔄 Iterative Pipeline (Advanced Feature)

**What it does**: The iterative pipeline automatically improves your class representatives by learning from mistakes. It identifies false positives from your predictions and adds them back as negative training examples ("not_classname") to teach the model what NOT to predict.

### How the Iterative Process Works:
1. **Initial Training**: Trains representatives on your positive examples
2. **Prediction**: Uses representatives to predict on a large dataset 
3. **Evaluation**: Compares predictions against ground truth to find false positives
4. **Learning**: Adds false positives as "not_[class]" negative examples
5. **Repeat**: Trains new representatives that avoid previous mistakes

### Quick Start with Iterative Pipeline:
```bash
# Run 5 iterations with your three data files
python run_iterative_pipeline.py \
    --definitive-objects path/to/your/positive_examples.pkl \
    --resnet-predictions path/to/your/full_dataset.pkl \
    --tracking-info path/to/your/ground_truth.pkl \
    --iterations 5
```

### Required Data Files:

**1. Definitive Objects** (`--definitive-objects`)
- Format: DataFrame with `class` and `finetuned_embedding` columns
- Contains: Your initial positive training examples
- Example: Hand-labeled crops of TVA Monitor, Classic Loki Armor, etc.

**2. ResNet Predictions** (`--resnet-predictions`) 
- Format: DataFrame with `video`, `frame`, `owl_label`, `finetuned_embedding`
- Contains: Large dataset for making predictions (frame-level embeddings)
- Example: All video frames with their embeddings and true labels

**3. Tracking Info** (`--tracking-info`)
- Format: DataFrame with `video`, `tag`, `actual` (0/1)
- Contains: Video-level ground truth for evaluation
- Example: Which videos actually contain each object class

### Advanced Iterative Options:

**Exclusion Strategies** (How to handle training data in evaluation):
```bash
# Frame-level exclusions (default) - only exclude specific video+frame pairs
python run_iterative_pipeline.py --exclusion-strategy frame-level ...

# Video-level exclusions - exclude entire videos if any frame is problematic  
python run_iterative_pipeline.py --exclusion-strategy video-level ...

# Compare both strategies and see the difference
python run_iterative_pipeline.py --exclusion-strategy compare-both ...
```

**Adaptive Parameters** (Different settings for later iterations):
```bash
# Use stricter thresholds and margins after first iteration
python run_iterative_pipeline.py \
    --threshold 0.6 --secondary-threshold 0.7 \
    --margin 0.15 --secondary-margin 0.25 \
    --iterations 5 ...
```

**Evaluation Modes**:
```bash
# Clean evaluation (exclude training data, recommended)
python run_iterative_pipeline.py --exclude-training-from-eval ...

# Contaminated evaluation (include all data)  
python run_iterative_pipeline.py --include-training-in-eval ...

# Track both clean and contaminated for comparison
python run_iterative_pipeline.py --track-training-separately ...
```

### Understanding Iterative Results:

**Output Structure**:
```
results_negative/
├── pipeline_summary.json              # Overall metrics across iterations
├── final_training_data.pkl            # Combined positive + negative examples
├── cumulative_exclusions.json         # All excluded false positives
├── iteration_1/                       # First iteration results
│   ├── representatives.pkl            # Learned representatives
│   ├── evaluation_results.csv         # Prediction vs ground truth
│   ├── false_positives_for_training.csv  # FPs to add as negatives
│   └── training_results.json          # F1, precision, recall
├── iteration_2/                       # Second iteration (with negatives)
│   └── ...
├── analysis_logs/                     # Debugging information  
│   ├── chat_summary_[timestamp].txt   # Copy this for debugging help
│   ├── evaluation_analysis_[timestamp].txt
│   └── exclusion_details_[timestamp].txt
└── tracking_exports/                  # Comprehensive data flow tracking
    ├── cumulative_exclusions_all.csv
    ├── iteration_summaries_all.csv
    └── data_flow_complete.csv
```

**Key Metrics to Watch**:
- **F1 Score**: Should improve over iterations (target: +2-5% improvement)
- **False Positives**: Should decrease as model learns what NOT to predict
- **Training Size**: Grows as negative examples are added
- **Exclusions**: Track how much training data is being excluded from evaluation

### Example Results Progression:
```
Iteration 1: F1=0.850, Training=100 samples, 0 negatives
Iteration 2: F1=0.895, Training=115 samples, 15 "not_" negatives  
Iteration 3: F1=0.920, Training=128 samples, 28 "not_" negatives
Iteration 4: F1=0.925, Training=135 samples, 35 "not_" negatives (converged)
```

### Integration with Main Pipeline:

The iterative pipeline uses your existing `train_representatives.py` script internally:
1. **Saves** temporary training data (positive + negative examples)
2. **Calls** `train_representatives.py` with appropriate parameters
3. **Loads** the resulting representatives for prediction
4. **Repeats** this process, accumulating better training data




### Different Initialization Methods
```bash
# Try different starting points
python train_representatives.py --data data.pkl --init-method random         # Most exploration
python train_representatives.py --data data.pkl --init-method bounded_random # Balanced
python train_representatives.py --data data.pkl --init-method perturbed_means # Conservative
```

## 📊 Visualization

### Interactive HTML Diagrams
- **`docs/html/contrastive_learning_diagram.html`** - Complete process flow
- **`docs/html/contrastive_learning_slide.html`** - Presentation version

Open these in a browser to see interactive visualizations of how the algorithm works.

## 📂 Project Structure

```
contrastive_learning/
├── README.md                           # This file
├── TEAM_HANDOFF.md                     # Team transition info
├── train_representatives.py            # Main training script
├── run_iterative_pipeline.py           # Advanced pipeline
├── src/                                # Core code
├── scripts/                            # Analysis tools
├── tests/                              # Test suite
├── docs/html/                          # Interactive visualizations (KEEP THESE!)
└── results/                            # Training outputs
```

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Test specific components
pytest tests/test_trainer.py -v
pytest tests/test_data_loader.py -v

# Integration test with toy data
python train_representatives.py --data examples/toy_data.pkl --epochs 10
```

## 📊 Example Workflows

### For Classification Task:
```bash
# 1. Train representatives
python train_representatives.py --data embeddings.pkl --margin 0.3 --lambda-push 0.6

# 2. Load and use for classification
import pandas as pd
reps = pd.read_pickle('results/.../representatives.pkl')
# Use reps['finetuned_embedding'] for nearest neighbor classification
```

### For Parameter Optimization:
```bash
# 1. Run parameter sweep
python scripts/parameter_sweep.py --data embeddings.pkl

# 2. Analyze results
python scripts/analyze_sweep_results.py --results-dir sweep_results/

# 3. Use best parameters for final training
python train_representatives.py --data embeddings.pkl --init-method random --margin 0.22 --lambda-push 0.5


## 🎯 The Math (Simple Version)

The algorithm learns class representatives by:
1. **Pull**: Make representatives similar to same-class samples
2. **Push**: Make representatives different from other-class samples
3. **Margin**: Ensure minimum distance between different classes

**Loss Function**: L = -pull_force + λ × push_force

Higher `margin` and `lambda-push` = better class separation but needs more training.

---

**Quick Start Summary**: `python train_representatives.py --data your_data.pkl` and check the results folder! 