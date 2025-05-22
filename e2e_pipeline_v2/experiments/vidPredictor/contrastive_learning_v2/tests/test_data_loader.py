import pytest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import tempfile

# Note: These imports will fail until we create the actual modules
# For now, we're writing the tests first (TDD approach)
# from src.data_loader import load_and_split_data, validate_dataframe


class TestDataValidation:
    """Test data validation functions."""
    
    def test_validate_dataframe_success(self, sample_dataframe):
        """Test validation passes with properly formatted data."""
        # This test defines the interface we expect
        # validate_dataframe should not raise any exceptions
        
        # Expected behavior:
        # - DataFrame has 'class' and 'fine_tuned_embeddings' columns
        # - All embeddings are same length
        # - All embeddings are numeric and finite
        # - At least one sample per class
        
        assert 'class' in sample_dataframe.columns
        assert 'fine_tuned_embeddings' in sample_dataframe.columns
        
        # Check embedding consistency
        embeddings = sample_dataframe['fine_tuned_embeddings'].tolist()
        embedding_lengths = [len(emb) for emb in embeddings]
        assert len(set(embedding_lengths)) == 1, "All embeddings should have same length"
        
        # Check all embeddings are finite
        for emb in embeddings:
            assert np.all(np.isfinite(emb)), "All embeddings should be finite"
    
    def test_validate_dataframe_missing_columns(self, invalid_dataframe):
        """Test validation fails with missing required columns."""
        # Should raise ValueError when required columns are missing
        
        required_columns = ['class', 'fine_tuned_embeddings']
        missing_columns = [col for col in required_columns if col not in invalid_dataframe.columns]
        
        assert len(missing_columns) > 0, "Test data should be missing required columns"
        
        # When we implement validate_dataframe, it should raise ValueError here
        # with pytest.raises(ValueError, match="Missing required columns"):
        #     validate_dataframe(invalid_dataframe)
    
    def test_validate_embeddings_format(self, sample_dataframe):
        """Test embeddings are properly formatted."""
        embeddings = sample_dataframe['fine_tuned_embeddings'].tolist()
        
        # All embeddings should be numpy arrays or lists of numbers
        for emb in embeddings:
            assert isinstance(emb, (np.ndarray, list))
            if isinstance(emb, np.ndarray):
                assert emb.dtype in [np.float32, np.float64]
    
    def test_validate_dataframe_with_nans(self, dataframe_with_nans):
        """Test validation handles NaN and infinite values."""
        # Should either raise error or warn about NaN/inf values
        
        embeddings = dataframe_with_nans['fine_tuned_embeddings'].tolist()
        
        # Verify test data actually contains NaN/inf
        has_nan = any(np.any(np.isnan(emb)) for emb in embeddings)
        has_inf = any(np.any(np.isinf(emb)) for emb in embeddings)
        
        assert has_nan or has_inf, "Test data should contain NaN or inf values"
        
        # When implemented, should handle this gracefully
        # Either raise error or clean the data with warning


class TestDataLoading:
    """Test data loading from PKL files."""
    
    def test_load_valid_pkl(self, temp_pkl_file):
        """Test loading a properly formatted PKL file."""
        # Should successfully load and return DataFrame
        
        # Verify the temp file exists and contains our test data
        assert Path(temp_pkl_file).exists()
        
        # Load manually to verify content
        with open(temp_pkl_file, 'rb') as f:
            loaded_df = pickle.load(f)
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 30  # 3 classes × 10 samples
        assert 'class' in loaded_df.columns
        assert 'fine_tuned_embeddings' in loaded_df.columns
        
        # When we implement load_and_split_data:
        # train_df, val_df, class_names = load_and_split_data(temp_pkl_file)
        # assert isinstance(train_df, pd.DataFrame)
        # assert isinstance(val_df, pd.DataFrame)
        # assert len(class_names) == 3
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        nonexistent_path = "/path/that/does/not/exist.pkl"
        
        # Should raise FileNotFoundError
        # with pytest.raises(FileNotFoundError):
        #     load_and_split_data(nonexistent_path)
        
        # For now, just verify the path doesn't exist
        assert not Path(nonexistent_path).exists()


class TestDataSplitting:
    """Test train/validation splitting functionality."""
    
    def test_stratified_split(self, sample_dataframe):
        """Test train/val split preserves class distribution."""
        val_frac = 0.3
        
        # Expected behavior after implementation:
        # train_df, val_df, class_names = load_and_split_data(df, val_frac=val_frac)
        
        # For now, test the logic we expect:
        class_counts = sample_dataframe['class'].value_counts()
        total_samples = len(sample_dataframe)
        expected_val_size = int(total_samples * val_frac)
        expected_train_size = total_samples - expected_val_size
        
        # All classes should be represented
        unique_classes = sample_dataframe['class'].unique()
        assert len(unique_classes) == 3
        
        # Each class should have enough samples for splitting
        min_samples_per_class = class_counts.min()
        assert min_samples_per_class >= 2, "Need at least 2 samples per class for splitting"
    
    def test_split_with_small_classes(self, single_sample_per_class):
        """Test behavior when some classes have only 1 sample."""
        # This is an edge case - what should happen?
        # Options: 1) Raise error, 2) Put single samples in train, 3) Warn user
        
        class_counts = single_sample_per_class['class'].value_counts()
        assert class_counts.max() == 1, "All classes should have only 1 sample"
        
        # When implemented, should handle this gracefully
        # Maybe put all samples in training set and warn about no validation data
    
    def test_split_preserves_all_classes(self, sample_dataframe):
        """Test that both train and val splits contain all classes."""
        # Critical requirement: stratified split should ensure
        # both train and val have samples from every class
        
        unique_classes = set(sample_dataframe['class'].unique())
        
        # After implementation:
        # train_df, val_df, _ = load_and_split_data(sample_dataframe, val_frac=0.3)
        # train_classes = set(train_df['class'].unique())
        # val_classes = set(val_df['class'].unique())
        # 
        # assert train_classes == unique_classes
        # assert val_classes == unique_classes
        
        # For now, verify our test data has multiple samples per class
        class_counts = sample_dataframe['class'].value_counts()
        assert all(count >= 3 for count in class_counts), "Need enough samples for proper splitting"


class TestDataIntegrity:
    """Test data integrity and consistency checks."""
    
    def test_embedding_dimensions_consistent(self, sample_dataframe):
        """Test all embeddings have the same dimensionality."""
        embeddings = sample_dataframe['fine_tuned_embeddings'].tolist()
        
        if len(embeddings) > 0:
            expected_dim = len(embeddings[0])
            for i, emb in enumerate(embeddings):
                assert len(emb) == expected_dim, f"Embedding {i} has wrong dimension"
    
    def test_class_labels_are_strings(self, sample_dataframe):
        """Test class labels are string type."""
        classes = sample_dataframe['class']
        assert all(isinstance(cls, str) for cls in classes)
    
    def test_no_empty_classes(self, sample_dataframe):
        """Test no class has zero samples."""
        class_counts = sample_dataframe['class'].value_counts()
        assert all(count > 0 for count in class_counts)
    
    def test_embeddings_are_numeric(self, sample_dataframe):
        """Test all embedding values are numeric."""
        embeddings = sample_dataframe['fine_tuned_embeddings'].tolist()
        
        for emb in embeddings:
            emb_array = np.array(emb)
            assert np.issubdtype(emb_array.dtype, np.number), "Embeddings should be numeric"


# Integration test placeholder
class TestDataLoaderIntegration:
    """Integration tests for the complete data loading pipeline."""
    
    def test_full_pipeline_with_real_data(self, temp_pkl_file):
        """Test the complete data loading pipeline."""
        # This will test the full workflow once implemented:
        # 1. Load PKL file
        # 2. Validate data format
        # 3. Split into train/val
        # 4. Return properly formatted data structures
        
        # For now, just verify our test setup works
        assert Path(temp_pkl_file).exists()
        
        with open(temp_pkl_file, 'rb') as f:
            df = pickle.load(f)
        
        assert len(df) > 0
        assert 'class' in df.columns
        assert 'fine_tuned_embeddings' in df.columns 