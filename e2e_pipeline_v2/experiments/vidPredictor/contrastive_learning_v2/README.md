# Contrastive Learning for Representative Learning

This experiment implements a contrastive learning approach to learn class representatives that stay close to positive samples and maintain a cosine margin from negative samples.

## 🎯 Objective

Learn optimal class representatives using the contrastive loss function:

```
L = Σ_c [ -mean(cos(r_c, P^c)) + λ * mean(max(0, cos(r_c, N^c) - margin)) ]
```

Where:
- `r_c` is the representative for class c
- `P^c` are positive samples (same class as c)  
- `N^c` are negative samples (different class from c)
- `margin` is the cosine similarity margin for negatives
- `λ` (lambda_push) controls the strength of the push term

## 📁 Project Structure

```
contrastive_learning_v2/
├── src/                          # Source modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and validation
│   ├── loss_functions.py        # Contrastive loss implementation
│   ├── trainer.py               # Training loop and evaluation
│   └── visualizer.py            # Plotting and visualization
├── tests/                       # Comprehensive test suite
│   ├── conftest.py             # Test fixtures
│   ├── test_data_loader.py     # Data loading tests
│   ├── test_loss_functions.py  # Loss function tests
│   ├── test_trainer.py         # Trainer tests
│   └── test_integration.py     # End-to-end tests
├── train_representatives.py    # Main experiment script
├── test_experiment.py          # Test with toy data
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Test Experiment

```bash
python test_experiment.py
```

This creates synthetic 2D data and runs the complete pipeline to verify everything works.

### 3. Run with Real Data

```bash
python train_representatives.py --data path/to/your/data.pkl --output results/
```

## 📊 Data Format

Your PKL file should contain a pandas DataFrame with:
- `class`: String class labels
- `fine_tuned_embeddings`: List/array of embedding vectors

Example:
```python
import pandas as pd
import numpy as np

data = [
    {'class': 'mirror', 'fine_tuned_embeddings': np.array([1.0, 2.0, ...])},
    {'class': 'lamp', 'fine_tuned_embeddings': np.array([3.0, 4.0, ...])},
    # ... more samples
]
df = pd.DataFrame(data)
```

## ⚙️ Configuration

Edit `config.yaml` or use command line arguments:

```bash
python train_representatives.py \
    --data data.pkl \
    --output results/ \
    --lr 0.01 \
    --margin 0.15 \
    --lambda-push 0.25 \
    --epochs 50 \
    --val-frac 0.3
```

### Key Parameters

- **Learning Rate (`lr`)**: Controls optimization step size (default: 0.01)
- **Margin (`margin`)**: Cosine similarity margin for negatives (default: 0.15)
- **Lambda Push (`lambda_push`)**: Weight for push term (default: 0.25)
- **Epochs**: Number of training iterations (default: 50)
- **Val Fraction**: Fraction of data for validation (default: 0.3)

## 📈 Output

The experiment generates:

1. **Results JSON** (`results.json`): Metrics and configuration
2. **Representatives PKL** (`representatives.pkl`): Learned class representatives
3. **Visualizations**:
   - `loss_curve.png`: Training loss over epochs
   - `tsne_plot.png`: t-SNE visualization of embeddings and representatives
   - `embeddings_2d.png`: Direct 2D plot (for 2D data)

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
python -m pytest -v

# Specific module tests
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/test_loss_functions.py -v
python -m pytest tests/test_trainer.py -v
python -m pytest tests/test_integration.py -v
```

## 🔬 Implementation Details

### Data Loader (`src/data_loader.py`)
- Validates PKL file format
- Performs stratified train/validation splits
- Handles edge cases (single samples per class, etc.)

### Loss Functions (`src/loss_functions.py`)
- Cosine similarity matrix computation
- Contrastive loss with pull and push terms
- Separate loss component analysis

### Trainer (`src/trainer.py`)
- Representative initialization as class means
- Adam optimizer with configurable learning rate
- F1 score evaluation using nearest representative classification

### Visualizer (`src/visualizer.py`)
- Training loss curves with trend lines
- t-SNE visualizations for high-dimensional data
- Direct 2D plots for toy data
- Similarity matrix heatmaps

## 📊 Example Results

With well-separated toy data:
- **Baseline F1**: 1.0000 (class means already optimal)
- **Final F1**: 1.0000 (maintained performance)
- **Loss**: Decreases during training
- **Visualizations**: Clear class separation in plots

## 🔧 Troubleshooting

### Common Issues

1. **Module not found**: Ensure you're in the correct directory
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Data format errors**: Check PKL file has required columns
4. **Memory issues**: Reduce batch size or use CPU device

### Performance Tips

- Use GPU if available (automatically detected)
- Adjust learning rate based on convergence
- Increase lambda_push for better class separation
- Use higher margin for more distinct boundaries

## 📚 References

- Contrastive Learning principles
- Cosine similarity metrics
- Representative learning approaches
- Test-driven development practices

## 🤝 Contributing

1. Run tests before submitting changes
2. Follow existing code style and documentation
3. Add tests for new features
4. Update README for significant changes 