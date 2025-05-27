"""
Data loading and validation module for contrastive learning experiment.

This module provides functions to:
1. Load and validate PKL files containing embeddings
2. Split data into train/validation sets with stratification
3. Ensure data integrity and proper formatting
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import warnings


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame has the required format for contrastive learning.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If DataFrame is missing required columns or has invalid data
    """
    # Check required columns
    required_columns = ['class', 'fine_tuned_embeddings']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) == 0:
        raise ValueError("Empty dataset")
    
    # Check class labels are strings
    if not all(isinstance(cls, str) for cls in df['class']):
        raise ValueError("All class labels must be strings")
    
    # Check embeddings format and consistency
    embeddings = df['fine_tuned_embeddings'].tolist()
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings found")
    
    # Check all embeddings have same dimension
    embedding_lengths = [len(emb) for emb in embeddings]
    if len(set(embedding_lengths)) != 1:
        raise ValueError("All embeddings must have the same dimension")
    
    # Check for NaN/inf values
    for i, emb in enumerate(embeddings):
        emb_array = np.array(emb)
        if not np.issubdtype(emb_array.dtype, np.number):
            raise ValueError(f"Embedding {i} contains non-numeric values")
        
        if np.any(np.isnan(emb_array)):
            raise ValueError(f"Embedding {i} contains NaN values")
        
        if np.any(np.isinf(emb_array)):
            raise ValueError(f"Embedding {i} contains infinite values")
    
    # Check each class has at least one sample
    class_counts = df['class'].value_counts()
    if any(count == 0 for count in class_counts):
        raise ValueError("Found classes with zero samples")


def load_and_split_data(pkl_path: str, val_frac: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load PKL file and split into train/validation sets with stratification.
    
    Args:
        pkl_path: Path to PKL file containing DataFrame
        val_frac: Fraction of data to use for validation (0.0 to 1.0)
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_df, val_df, class_names)
        
    Raises:
        FileNotFoundError: If PKL file doesn't exist
        ValueError: If data format is invalid
    """
    # Check file exists
    if not Path(pkl_path).exists():
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")
    
    # Load PKL file
    try:
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load PKL file: {e}")
    
    # Validate data format
    validate_dataframe(df)
    
    # Get unique classes
    unique_classes = sorted(df['class'].unique())
    class_counts = df['class'].value_counts()
    
    # Check if we have enough samples for splitting
    min_samples_per_class = class_counts.min()
    if min_samples_per_class == 1 and val_frac > 0:
        warnings.warn(
            f"Some classes have only 1 sample. Putting all samples in training set.",
            UserWarning
        )
        return df.copy(), pd.DataFrame(columns=df.columns), unique_classes
    
    # Perform stratified split
    if val_frac == 0:
        return df.copy(), pd.DataFrame(columns=df.columns), unique_classes
    
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=val_frac,
            stratify=df['class'],
            random_state=random_state
        )
    except ValueError as e:
        # If stratification fails, fall back to random split with warning
        warnings.warn(
            f"Stratified split failed ({e}). Using random split instead.",
            UserWarning
        )
        train_df, val_df = train_test_split(
            df,
            test_size=val_frac,
            random_state=random_state
        )
    
    # Verify both splits have all classes (if possible)
    train_classes = set(train_df['class'].unique())
    val_classes = set(val_df['class'].unique())
    all_classes = set(unique_classes)
    
    if len(val_df) > 0:
        missing_from_train = all_classes - train_classes
        missing_from_val = all_classes - val_classes
        
        if missing_from_train:
            warnings.warn(f"Training set missing classes: {missing_from_train}")
        if missing_from_val:
            warnings.warn(f"Validation set missing classes: {missing_from_val}")
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), unique_classes 