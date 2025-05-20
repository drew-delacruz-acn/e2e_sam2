# Supervised Contrastive Learning for Class Prototypes

This package implements a supervised contrastive learning pipeline that generates class prototypes from embedding vectors.

## Overview

This implementation turns a DataFrame (or its pickle) containing embeddings into a second DataFrame containing one representative embedding per object class, learned with supervised contrastive learning (SupCon). The pipeline follows these steps:

1. **Data Loading**: Load embeddings from DataFrame or pickle file
2. **Dataset Preparation**: Prepare embeddings for training  
3. **Model Training**: Train a projection head with supervised contrastive loss
4. **Prototype Generation**: Create class prototypes using mean or medoid aggregation
5. **Evaluation**: Compute cluster quality metrics (optional)

## Requirements

- Python 3.6+
- PyTorch
- pandas
- numpy
- scikit-learn
- pytorch-metric-learning

## Installation

The code requires `pytorch-metric-learning`. Install it with:

```bash
pip install pytorch-metric-learning
```

## Usage

### Basic Usage

```python
import pandas as pd
from contrastiveLearning.pipeline import run_pipeline

# Load your DataFrame with 'class' and 'embedding' columns
df = pd.read_pickle("your_embeddings.pkl")

# Generate class prototypes using mean
prototypes = run_pipeline(df, method="mean")

# Save the prototypes
prototypes.to_pickle("class_prototypes.pkl")
```

### Input Format

Your input DataFrame must have these columns:
- `class`: An integer class ID
- `embedding`: A numpy array or list containing the embedding vector

### Options

The pipeline supports two methods for prototype generation:
- `mean`: Creates prototypes by averaging all embeddings in a class (default)
- `medoid`: Uses the embedding closest to the class centroid (robust to outliers)

## Components

- `data_io.py`: Load embeddings from DataFrame or pickle
- `dataset.py`: Wrap embeddings in PyTorch Dataset
- `model.py`: Define projection head neural network
- `training.py`: Implement training with SupCon loss
- `prototype.py`: Build class prototypes
- `pipeline.py`: Orchestrate the entire process

## Example

See `example_supcon.py` for a complete demonstration. 