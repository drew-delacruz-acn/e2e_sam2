# Parameter Sweep for Contrastive Learning

This directory contains scripts for running comprehensive parameter sweeps to optimize contrastive learning performance.

## Overview

The parameter sweep systematically tests different combinations of:
- **4 initialization methods**: `class_means`, `random`, `bounded_random`, `perturbed_means`
- **3 learning rates**: `0.01`, `0.001`, `0.0001`
- **3 epoch settings**: `50`, `100`, `150`
- **3 margin values**: `0.15`, `0.22`, `0.3`
- **3 lambda values**: `0.25`, `0.5`, `0.75`

**Total combinations**: 4 × 3 × 3 × 3 × 3 = **324 experiments**

## Key Features

### 🧪 Parameter Sweep (`parameter_sweep.py`)
- **GPU Memory Management**: Automatically clears GPU memory after each experiment to prevent bloat
- **Resume Capability**: Can resume interrupted sweeps by skipping completed experiments
- **Progress Tracking**: Shows real-time progress and estimated completion time
- **Auto-naming**: Generates descriptive folder names based on parameters
- **Results Collection**: Automatically aggregates results into CSV and JSON formats

### 📊 Results Analysis (`analyze_sweep_results.py`)
- **Parameter Importance**: Analyzes which parameters have the most impact
- **Best Combinations**: Identifies top-performing parameter sets
- **Visualizations**: Creates plots and heatmaps for easy interpretation
- **Recommendations**: Provides actionable insights based on results

## Usage

### 1. Running the Parameter Sweep

```bash
# Basic usage
python parameter_sweep.py --data path/to/embeddings.pkl --output sweep_results/

# Resume interrupted sweep
python parameter_sweep.py --data path/to/embeddings.pkl --output sweep_results/ --resume
```

**Required:**
- `--data`: Path to PKL file containing embeddings

**Optional:**
- `--output`: Output directory (default: `sweep_results/`)
- `--resume`: Resume from previous run

### 2. Analyzing Results

```bash
# Full analysis with plots
python analyze_sweep_results.py --results-dir sweep_results/

# Analysis without plots
python analyze_sweep_results.py --results-dir sweep_results/ --no-plots

# Show top 20 results
python analyze_sweep_results.py --results-dir sweep_results/ --top-n 20
```

## Output Structure

```
sweep_results/
├── experiments/                    # Individual experiment results
│   ├── init_class_means_lr_0.01_margin_0.15_lambda_0.25_epochs_50/
│   ├── init_random_lr_0.001_margin_0.22_lambda_0.5_epochs_100/
│   └── ...
├── sweep_results.csv              # All results in tabular format
├── sweep_results.json             # All results in JSON format
├── sweep_summary.txt              # Human-readable summary
├── parameter_analysis.png         # Parameter comparison plots
└── init_lr_heatmap.png           # Initialization vs Learning Rate heatmap
```

### Individual Experiment Structure
Each experiment folder contains:
- `results.json` - Metrics and configuration
- `representatives.pkl` - Learned class representatives (DataFrame format)
- `loss_curve.png` - Training loss visualization
- `tsne_plot.png` - t-SNE visualization of representatives
- `embeddings_2d.png` - 2D embedding visualization

## Key Metrics Tracked

- **F1 Score**: Primary performance metric
- **Baseline F1**: Performance with original class means
- **Improvement**: F1 improvement over baseline
- **Final Loss**: Contrastive loss at end of training
- **Initial Loss**: Contrastive loss at start of training

## Parameter Details

### Initialization Methods
1. **`class_means`**: Initialize with class mean embeddings (default)
2. **`random`**: Random initialization from standard normal distribution
3. **`bounded_random`**: Random initialization within embedding space bounds
4. **`perturbed_means`**: Class means with small random perturbation

### Learning Rates
- **`0.01`**: High learning rate for fast convergence
- **`0.001`**: Medium learning rate for balanced training
- **`0.0001`**: Low learning rate for fine-tuned optimization

### Margin Values
- **`0.15`**: Conservative margin (original default)
- **`0.22`**: Moderate margin for better separation
- **`0.3`**: Aggressive margin for maximum separation

### Lambda Values (Push Term Weight)
- **`0.25`**: Light push term (original default)
- **`0.5`**: Moderate push term for balanced training
- **`0.75`**: Strong push term for aggressive separation

### Epochs
- **`50`**: Quick training for initial exploration
- **`100`**: Standard training duration
- **`150`**: Extended training for convergence

## GPU Memory Management

The sweep automatically manages GPU memory by:
1. Clearing CUDA cache before each experiment
2. Synchronizing GPU operations
3. Running garbage collection
4. Clearing memory again after completion/failure

This prevents memory bloat during long-running sweeps.

## Time Estimation

Based on previous experiments:
- **Single experiment**: ~30-60 seconds (depending on epochs)
- **Full sweep (324 experiments)**: ~3-5 hours
- **Estimated completion time**: Shown after first experiment

## Resume Functionality

If a sweep is interrupted:
1. Use `--resume` flag to continue
2. Script automatically detects completed experiments
3. Only runs remaining combinations
4. Preserves all previous results

## Analysis Features

### Parameter Importance Analysis
- Shows average F1 score for each parameter value
- Identifies which parameters have the most impact
- Provides statistical significance (mean ± std)

### Best Results Identification
- Lists top N performing combinations
- Shows complete parameter sets for reproduction
- Highlights improvement over baseline

### Visualization
- Box plots comparing parameter effects
- Heatmaps showing parameter interactions
- Easy-to-interpret charts for presentations

## Example Results Format

### CSV Output
```csv
experiment_name,init_method,lr,margin,lambda_push,epochs,final_f1,baseline_f1,improvement,final_loss,initial_loss
init_random_lr_0.001_margin_0.22_lambda_0.5_epochs_100,random,0.001,0.22,0.5,100,0.9876,0.9234,0.0642,-15.234,-8.123
```

### Summary Output
```
BEST RESULTS:
Best F1 Score: 0.9876
  Experiment: init_random_lr_0.001_margin_0.22_lambda_0.5_epochs_100
  Parameters: init=random, lr=0.001, margin=0.22, lambda=0.5, epochs=100

AVERAGE F1 BY INITIALIZATION METHOD:
class_means    : 0.9234 (n=81)
random         : 0.9456 (n=81)
bounded_random : 0.9345 (n=81)
perturbed_means: 0.9123 (n=81)
```

## Testing

Use `test_parameter_combinations.py` to verify the parameter grid:
```bash
python test_parameter_combinations.py
```

This confirms:
- Correct number of combinations (324)
- All combinations are unique
- Even distribution across parameters

## Tips for Success

1. **Start Small**: Test with a subset first using modified parameter grid
2. **Monitor Progress**: Check intermediate results to catch issues early
3. **Use Resume**: Don't restart from scratch if interrupted
4. **GPU Memory**: Ensure sufficient GPU memory for your model size
5. **Storage Space**: 324 experiments require significant disk space

## Troubleshooting

### Common Issues
- **GPU OOM**: Reduce batch size in training script
- **Disk Space**: Monitor available storage during sweep
- **Interrupted Sweep**: Use `--resume` to continue
- **Missing Dependencies**: Ensure all packages installed in venv

### Error Recovery
- Failed experiments are logged but don't stop the sweep
- GPU memory is cleared even after failures
- Results are collected from all successful experiments
- Analysis works with partial results

## Integration with Existing Workflow

This parameter sweep integrates seamlessly with:
- Existing `train_representatives.py` script
- Current data format (PKL embeddings)
- Established output structure
- DataFrame-based representatives format

The sweep uses the same underlying training logic, just systematically varies the parameters for comprehensive optimization. 