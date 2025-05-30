# SAM2 Training and Inference Pipeline

A comprehensive toolkit for fine-tuning and running inference with the Segment Anything 2 (SAM2) model. This project provides a complete pipeline for training SAM2 on custom datasets with validation, model comparison, and inference capabilities.

## 🏗️ Project Structure

```
train_sam2/
├── scripts/                           # Main executable scripts
│   ├── train.py                       # Fine-tuning script with validation
│   ├── inference.py                   # Inference with fine-tuned models
│   ├── compare_models.py              # Model performance comparison
│   ├── prepare_dataset.py             # Dataset preparation utility
│   └── run_pipeline.py                # Full pipeline orchestrator
│
├── config/                            # Configuration files
│   ├── default_config.py              # Default settings
│   ├── training_configs.yaml          # Training hyperparameters
│   └── model_configs.yaml             # Model variants and paths
│
├── utils/                             # Utility modules
│   ├── data_utils.py                  # Dataset loading and processing
│   ├── model_utils.py                 # Model loading and device setup
│   ├── metrics.py                     # Training metrics and evaluation
│   └── visualization.py               # Visualization utilities
│
├── outputs/                           # Generated outputs (created automatically)
│   ├── models/                        # Saved model checkpoints
│   ├── logs/                          # Training logs and metrics
│   ├── visualizations/                # Generated plots and comparisons
│   └── results/                       # Evaluation results
│
└── README.md                          # This file
```

## 🚀 Quick Start

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SAM2 checkpoints**: Ensure you have SAM2 model checkpoints and configs in the appropriate locations (see [Model Configuration](#model-configuration))

### 5-Minute Example

```bash
# Quick demo with minimal setup
python examples/quick_start.py

# Or run the full pipeline step by step:
python scripts/run_pipeline.py --sequence bear --max_iterations 100
```

## 📖 Usage Guide

### 1. Dataset Preparation

Prepare your dataset splits for training and validation:

```bash
python scripts/prepare_dataset.py \
    --data_dir /path/to/davis/dataset \
    --sequence bear \
    --val_split 0.2 \
    --output_dir ./dataset_splits
```

**Supported Datasets**: 
- DAVIS 2017 (default)
- Custom datasets with similar structure

**Expected Directory Structure**:
```
DAVIS/
├── JPEGImages/480p/<sequence>/
│   └── *.jpg
└── Annotations/480p/<sequence>/
    └── *.png
```

### 2. Training

#### Basic Training
```bash
python scripts/train.py \
    --data_dir /path/to/dataset \
    --splits_dir ./dataset_splits \
    --max_iterations 10000 \
    --output_model ./outputs/models/my_model.torch
```

#### Advanced Training with Configuration
```bash
python scripts/train.py \
    --data_dir /path/to/dataset \
    --learning_rate 5e-6 \
    --max_iterations 5000 \
    --val_interval 10 \
    --early_stopping_patience 5
```

#### Full Pipeline (Recommended)
```bash
python scripts/run_pipeline.py \
    --data_dir /path/to/dataset \
    --sequence bear \
    --max_iterations 1000 \
    --val_split 0.2
```

### 3. Inference

#### Basic Inference
```bash
python scripts/inference.py \
    --image_path /path/to/image.jpg \
    --mask_path /path/to/mask.png \
    --sam2_checkpoint ./outputs/models/my_model.torch \
    --output_path ./outputs/visualizations/result.png
```

#### Model Comparison
```bash
python scripts/inference.py \
    --image_path /path/to/image.jpg \
    --mask_path /path/to/mask.png \
    --sam2_checkpoint ./outputs/models/my_model.torch \
    --original_checkpoint ../checkpoints/sam2.1_hiera_large.pt \
    --compare \
    --output_path ./outputs/visualizations/comparison.png
```

### 4. Model Evaluation

Compare performance between fine-tuned and original models:

```bash
python scripts/compare_models.py \
    --data_dir /path/to/dataset \
    --splits_dir ./dataset_splits \
    --finetuned_checkpoint ./outputs/models/my_model.torch \
    --pretrained_checkpoint ../checkpoints/sam2.1_hiera_large.pt \
    --output_dir ./outputs/results
```

## ⚙️ Configuration

### Training Configurations

The `config/training_configs.yaml` file contains different preset configurations:

- **`default`**: Standard settings for most use cases
- **`fine_tuning`**: Optimized for small datasets
- **`quick_test`**: Fast training for development
- **`robust`**: Conservative settings for large datasets
- **`aggressive`**: Higher learning rates for faster convergence

### Model Variants

Configure different SAM2 model sizes in `config/model_configs.yaml`:

- **Large** (672M params): Best accuracy, requires more compute
- **Base+** (80M params): Balanced accuracy and speed
- **Small** (46M params): Faster inference
- **Tiny** (39M params): Fastest, for edge deployment

### Key Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `learning_rate` | Optimizer learning rate | 1e-5 | 5e-6 to 2e-5 |
| `max_iterations` | Training iterations | 10000 | 1000-20000 |
| `val_interval` | Validation frequency | 10 | 5-20 |
| `gap_threshold` | Train-val IoU gap warning | 1.0 | 0.5-2.0 |
| `early_stopping_patience` | Stop after N warnings | 5 | 3-10 |

## 📊 Monitoring and Evaluation

### Training Metrics

The training script automatically tracks:
- **Training IoU**: Performance on training data
- **Validation IoU**: Performance on validation data
- **Gap Analysis**: Overfitting detection
- **Early Stopping**: Automatic stopping for overfitting

Metrics are saved to `outputs/logs/training_metrics.json` and can be visualized.

### Evaluation Metrics

Model comparison provides:
- **IoU (Intersection over Union)**: Segmentation accuracy
- **Dice Coefficient**: Alternative accuracy measure
- **Precision/Recall**: Detection performance
- **F1 Score**: Balanced accuracy metric

### Visualizations

Generated visualizations include:
- **Training curves**: IoU over time
- **Segmentation comparisons**: Side-by-side model outputs
- **Error analysis**: Failed predictions
- **Performance improvements**: Before/after comparisons

## 🛠️ Advanced Usage

### Custom Datasets

To use your own dataset:

1. **Organize data** in DAVIS format
2. **Update paths** in configuration files
3. **Modify data loading** in `utils/data_utils.py` if needed

Example for custom dataset:
```python
# In your custom script
from utils.data_utils import create_dataset_splits, save_splits

# Load your data
data = load_your_custom_data()

# Create splits
splits = create_dataset_splits(data, val_split=0.2, seed=42)

# Save for training
save_splits(splits, "./custom_splits")
```

### Hyperparameter Tuning

Use the configuration system for systematic tuning:

```python
# examples/hyperparameter_tuning.py
configs = [
    {"learning_rate": 1e-5, "max_iterations": 5000},
    {"learning_rate": 5e-6, "max_iterations": 10000},
    {"learning_rate": 2e-5, "max_iterations": 3000}
]

for config in configs:
    train_with_config(config)
    evaluate_model()
```

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Errors**:
- Reduce batch size in data loading
- Use smaller model variant (base/small/tiny)
- Enable gradient checkpointing

**Training Divergence**:
- Lower learning rate
- Increase validation frequency
- Check data quality

**Poor Performance**:
- Ensure sufficient training data
- Verify data annotations
- Try different model variants
- Adjust loss function weights

**Device Issues**:
- CUDA: Ensure compatible PyTorch version
- MPS (Apple Silicon): Some operations may fall back to CPU
- CPU: Training will be significantly slower

### Debug Mode

Enable detailed logging:
```bash
python scripts/train.py --debug
python scripts/compare_models.py --debug
```

### Performance Optimization

**Speed up training**:
- Use mixed precision (automatically enabled)
- Reduce image resolution in config
- Use smaller validation sets
- Enable model parallelism for multiple GPUs

**Reduce memory usage**:
- Lower batch size
- Use gradient accumulation
- Clear cache regularly


## 📋 Requirements

### Core Dependencies
- `torch>=2.0.0` - PyTorch framework
- `torchvision>=0.15.0` - Vision utilities
- `opencv-python>=4.5.0` - Image processing
- `numpy>=1.19.0` - Numerical operations
- `PyYAML>=6.0` - Configuration files
- `tqdm>=4.64.0` - Progress bars

### Optional Dependencies
- `matplotlib>=3.5.0` - Plotting and visualization
- `tensorboard` - Training visualization
- `wandb` - Experiment tracking

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Storage**: 10GB+ free space for models and outputs
- **OS**: Linux, macOS, Windows

