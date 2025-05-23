# Representative Initialization Methods

This document explains the different initialization methods available for contrastive learning representatives and how to use them.

## Overview

The contrastive learning system now supports four different methods for initializing class representatives:

1. **`class_means`** (default) - Initialize as class means
2. **`random`** - Random initialization from standard normal distribution  
3. **`bounded_random`** - Random initialization within embedding space bounds
4. **`perturbed_means`** - Class means with small random perturbation

## Initialization Methods

### 1. Class Means (`class_means`)
**Default method** - Initialize representatives as the mean of all embeddings for each class.

**Pros:**
- Stable and predictable training
- Fast convergence
- Good starting point based on actual data
- Preserves existing embedding structure

**Cons:**
- May get stuck in local optima
- Biased toward training distribution
- Assumes Gaussian-like class distributions

**Best for:** Well-separated classes, stable training, production use

### 2. Random (`random`)
Initialize representatives randomly from a standard normal distribution (mean=0, std=1).

**Pros:**
- No bias toward training data
- Can escape poor local optima
- May find better global optima
- Good for exploration

**Cons:**
- Requires more training epochs (3-5x)
- Higher variance in results
- May start far from optimal regions
- Less predictable convergence

**Best for:** When class means are poor starting points, suspected local optima issues

### 3. Bounded Random (`bounded_random`)
Initialize representatives randomly within the bounds of the embedding space.

**Pros:**
- Data-aware random initialization
- Good compromise between exploration and relevance
- Stays within meaningful embedding ranges
- More stable than pure random

**Cons:**
- Still requires more epochs than class means
- May not escape all local optima
- Depends on embedding space characteristics

**Best for:** Moderate exploration while staying data-relevant

### 4. Perturbed Means (`perturbed_means`)
Initialize as class means plus small random perturbation (10% of class standard deviation).

**Pros:**
- Slight exploration around class means
- Maintains most benefits of class means
- Can help escape nearby local optima
- Minimal impact on convergence speed

**Cons:**
- Limited exploration range
- May not help with major local optima issues
- Still biased toward class means

**Best for:** Fine-tuning around class means, slight exploration

## Usage

### Command Line

```bash
# Default (class means) - auto-generates folder name
python train_representatives.py --data data.pkl
# → results/init_class_means_lr_0.01_margin_0.15_lambda_0.25_epochs_50/

# Random initialization - auto-generates folder name
python train_representatives.py --data data.pkl --init-method random --epochs 150
# → results/init_random_lr_0.01_margin_0.15_lambda_0.25_epochs_150/

# Bounded random with custom parameters
python train_representatives.py --data data.pkl --init-method bounded_random --epochs 100 --margin 0.22
# → results/init_bounded_random_lr_0.01_margin_0.22_lambda_0.25_epochs_100/

# Manual output directory (disables auto-naming)
python train_representatives.py --data data.pkl --output my_experiment/
# → my_experiment/

# Disable auto-naming but use default results folder
python train_representatives.py --data data.pkl --no-auto-name
# → results/
```

### Programmatic Usage

```python
from src.trainer import ContrastiveTrainer

config = {'lr': 0.01, 'margin': 0.15, 'lambda_push': 0.25}
trainer = ContrastiveTrainer(config)

# Initialize with different methods
trainer.initialize_representatives(embeddings, labels, init_method='random')
trainer.initialize_representatives(embeddings, labels, init_method='bounded_random')
trainer.initialize_representatives(embeddings, labels, init_method='perturbed_means')
trainer.initialize_representatives(embeddings, labels, init_method='class_means')  # default
```

## Recommendations

### When to Use Each Method

| Scenario | Recommended Method | Epochs | Notes |
|----------|-------------------|---------|-------|
| Production/stable training | `class_means` | 50-60 | Default choice |
| Classes well-separated | `class_means` or `perturbed_means` | 50-60 | Stable convergence |
| Classes overlap significantly | `random` or `bounded_random` | 100-150 | More exploration needed |
| Stuck in local optima | `random` | 150+ | Maximum exploration |
| Fine-tuning existing results | `perturbed_means` | 60-80 | Slight improvement |

### Parameter Adjustments

When using random initialization methods, consider:

- **Increase epochs**: Random methods typically need 2-5x more epochs
- **Adjust learning rate**: May benefit from slightly higher LR (0.015 vs 0.01)
- **Multiple runs**: Random methods have higher variance - run multiple times
- **Early stopping**: Monitor validation F1 to avoid overfitting

## Testing and Comparison

### Test Script
Use the provided test script to compare initialization methods:

```bash
python test_initialization_methods.py --num-classes 3 --samples-per-class 20
```

### Comparison Script
Run experiments with all methods:

```bash
python example_random_init.py
```

### Analysis
Compare results by examining:
- Final F1 scores in `results.json`
- Loss curves in `loss_curve.png`
- t-SNE plots for representative quality
- Inter-representative similarity matrices

## Expected Results

Based on analysis of your dataset (18 classes, armor/clothing types):

| Method | Expected Max Similarity | Expected Convergence | Recommended Epochs |
|--------|------------------------|---------------------|-------------------|
| `class_means` | 0.35-0.40 | Fast (20-30 epochs) | 50-60 |
| `random` | 0.25-0.35 | Slow (80-120 epochs) | 150+ |
| `bounded_random` | 0.30-0.40 | Medium (40-80 epochs) | 100 |
| `perturbed_means` | 0.35-0.40 | Fast (25-35 epochs) | 60-80 |

## Implementation Details

### Code Changes Made

1. **`src/trainer.py`**: Extended `initialize_representatives()` method
2. **`train_representatives.py`**: Added `--init-method` command line argument
3. **Results tracking**: Added `init_method` to saved results

### Files Added

- `test_initialization_methods.py` - Test and compare methods
- `example_random_init.py` - Example usage script
- `test_auto_naming.py` - Test automatic folder naming
- `INITIALIZATION_METHODS.md` - This documentation

### New Features

- **Automatic folder naming**: Results are saved in descriptive folders based on parameters
- **Parameter-based organization**: Easy to identify and compare experiments
- **Flexible output control**: Can disable auto-naming or use custom paths

## Troubleshooting

### Common Issues

1. **Random methods not converging**: Increase epochs to 150+
2. **High variance in results**: Run multiple seeds and average
3. **Poor performance with random**: Try `bounded_random` instead
4. **Slow convergence**: Increase learning rate slightly (0.012-0.015)

### Debugging

Check initialization quality:
```python
# Analyze initial representatives
from src.loss_functions import cosine_similarity_matrix

# Inter-representative similarities (should be low)
sim_matrix = cosine_similarity_matrix(representatives, representatives)
print(f"Max inter-rep similarity: {sim_matrix.max()}")

# Similarity to data (should be reasonable)
data_sim = cosine_similarity_matrix(representatives, embeddings)
print(f"Mean data similarity: {data_sim.mean()}")
```

## Automatic Folder Naming

### How It Works

When using the default output directory (`results/`), the system automatically generates descriptive folder names based on your parameters:

**Format**: `init_{method}_lr_{lr}_margin_{margin}_lambda_{lambda}_epochs_{epochs}[_seed_{seed}]`

**Examples**:
- `init_class_means_lr_0.01_margin_0.15_lambda_0.25_epochs_50`
- `init_random_lr_0.01_margin_0.3_lambda_0.6_epochs_150`
- `init_bounded_random_lr_0.015_margin_0.22_lambda_0.35_epochs_100_seed_123`

### Benefits

- **No overwrites**: Each parameter combination gets its own folder
- **Easy identification**: Parameters visible in folder name
- **Automatic organization**: Results grouped logically
- **Comparison friendly**: Easy to compare different experiments

### Control Options

- **Auto-naming** (default): Use `python train_representatives.py --data data.pkl`
- **Custom path**: Use `--output my_folder/` (disables auto-naming)
- **Disable auto-naming**: Use `--no-auto-name` flag

## Future Enhancements

Potential additional initialization methods:
- K-means++ style initialization
- Furthest-first initialization
- Embedding space clustering-based init
- Learned initialization from previous experiments 