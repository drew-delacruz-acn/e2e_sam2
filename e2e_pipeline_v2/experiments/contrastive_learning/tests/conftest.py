import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import tempfile
import pickle


@pytest.fixture
def sample_dataframe():
    """Create synthetic but realistic test data with 3 classes."""
    np.random.seed(42)
    classes = ['mirror', 'lamp', 'wall']
    data = []
    
    for cls in classes:
        # Create embeddings clustered around class-specific centers
        center = np.random.randn(2048)
        for i in range(10):  # 10 samples per class
            # Add some noise but keep clusters separable
            embedding = center + 0.1 * np.random.randn(2048)
            data.append({
                'class': cls, 
                'finetuned_embedding': embedding.astype(np.float32)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def toy_separable_data():
    """Create simple 2D data that's perfectly separable for testing algorithms."""
    np.random.seed(42)
    data = []
    
    # Class A: points around (2, 2)
    for i in range(5):
        embedding = np.array([2.0, 2.0]) + 0.1 * np.random.randn(2)
        data.append({'class': 'A', 'finetuned_embedding': embedding.astype(np.float32)})
    
    # Class B: points around (-2, -2)  
    for i in range(5):
        embedding = np.array([-2.0, -2.0]) + 0.1 * np.random.randn(2)
        data.append({'class': 'B', 'finetuned_embedding': embedding.astype(np.float32)})
        
    return pd.DataFrame(data)


@pytest.fixture
def invalid_dataframe():
    """Create DataFrame with missing required columns for testing error handling."""
    return pd.DataFrame({
        'wrong_column': ['A', 'B', 'C'],
        'another_wrong_column': [1, 2, 3]
    })


@pytest.fixture
def dataframe_with_nans():
    """Create DataFrame with NaN embeddings for testing validation."""
    data = []
    embeddings = [
        np.array([1.0, 2.0, 3.0]),
        np.array([np.nan, 2.0, 3.0]),  # Contains NaN
        np.array([1.0, np.inf, 3.0])   # Contains inf
    ]
    
    for i, emb in enumerate(embeddings):
        data.append({
            'class': f'class_{i}',
            'finetuned_embedding': emb.astype(np.float32)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_pkl_file(sample_dataframe):
    """Create a temporary PKL file with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(sample_dataframe, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def single_sample_per_class():
    """Create DataFrame with only one sample per class for edge case testing."""
    np.random.seed(42)
    data = []
    
    for i, cls in enumerate(['A', 'B', 'C']):
        embedding = np.random.randn(10).astype(np.float32)
        data.append({'class': cls, 'finetuned_embedding': embedding})
    
    return pd.DataFrame(data)


@pytest.fixture
def device():
    """Get available device (CPU or CUDA)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for output files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) 