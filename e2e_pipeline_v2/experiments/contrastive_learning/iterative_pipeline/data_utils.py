import pickle
import pandas as pd
from pathlib import Path
from typing import Tuple, Union
import traceback

# Constants for column names
COL_CLASS = 'class'
COL_EMBEDDING = 'finetuned_embedding'
COL_VIDEO = 'video'
COL_FRAME = 'frame'
COL_OWL_LABEL = 'owl_label'
COL_TAG = 'tag'
COL_ACTUAL = 'actual'

def load_data_flexible(data_input: Union[str, pd.DataFrame], data_name: str) -> pd.DataFrame:
    """
    Load data from either a file path (pickle) or DataFrame directly.
    
    Args:
        data_input: Either a file path string or pandas DataFrame
        data_name: Name for logging purposes
        
    Returns:
        pandas DataFrame
        
    Raises:
        TypeError: If input is neither string nor DataFrame
        FileNotFoundError: If file path doesn't exist
        Exception: If pickle loading fails
    """
    if isinstance(data_input, pd.DataFrame):
        print(f"📊 Using provided DataFrame for {data_name}: {data_input.shape}")
        return data_input.copy()
    elif isinstance(data_input, str):
        print(f"📂 Loading {data_name} from file: {data_input}")
        try:
            with open(data_input, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            return data
        except Exception as e:
            print(f"❌ Error loading {data_name}: {e}")
            traceback.print_exc()
            raise
    else:
        raise TypeError(f"{data_name} must be either a file path (str) or pandas DataFrame, got {type(data_input)}")

def load_and_validate_data(
    definitiveObjects_input: Union[str, pd.DataFrame], 
    resnetPredictions_input: Union[str, pd.DataFrame], 
    trackingInfo_input: Union[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate the three required data inputs.
    Each input can be either a file path (str) or a pandas DataFrame.
    
    Args:
        definitiveObjects_input: Path to pickle file or DataFrame with initial positive training examples
        resnetPredictions_input: Path to pickle file or DataFrame with frame-level embeddings  
        trackingInfo_input: Path to pickle file or DataFrame with ground truth tracking info
        
    Returns:
        Tuple of (definitiveObjects, resnetPredictions, trackingInfo) DataFrames
        
    Raises:
        ValueError: If required columns are missing from any DataFrame
        TypeError: If inputs are not strings or DataFrames
    """ 
    print("🔄 Loading and validating data...")
    
    try:
        # Load definitiveObjects (flexible input)
        definitiveObjects = load_data_flexible(definitiveObjects_input, "definitiveObjects")
        required_cols_do = [COL_CLASS, COL_EMBEDDING]
        if not all(col in definitiveObjects.columns for col in required_cols_do):
            raise ValueError(f"definitiveObjects missing required columns: {required_cols_do}. Available: {list(definitiveObjects.columns)}")
        definitiveObjects = definitiveObjects[required_cols_do].copy()
        print(f"✅ Processed definitiveObjects: {definitiveObjects.shape}")
        
        # Load resnetPredictions (flexible input)
        resnetPredictions = load_data_flexible(resnetPredictions_input, "resnetPredictions")
        required_cols_rp = [COL_VIDEO, COL_FRAME, COL_OWL_LABEL, COL_EMBEDDING]
        
        # Handle column name variations
        if COL_OWL_LABEL not in resnetPredictions.columns and COL_CLASS in resnetPredictions.columns:
            print(f"ℹ️  '{COL_OWL_LABEL}' not found, using '{COL_CLASS}' column and renaming to '{COL_OWL_LABEL}'.")
            resnetPredictions = resnetPredictions.rename(columns={COL_CLASS: COL_OWL_LABEL})
        
        if not all(col in resnetPredictions.columns for col in required_cols_rp):
            raise ValueError(f"resnetPredictions missing required columns: {required_cols_rp}. Available: {list(resnetPredictions.columns)}")

        resnetPredictions = resnetPredictions[required_cols_rp].copy()
        resnetPredictions[COL_FRAME] = resnetPredictions[COL_FRAME].astype(int)
        print(f"✅ Processed resnetPredictions: {resnetPredictions.shape}")
        
        # Load trackingInfo (flexible input)
        trackingInfo = load_data_flexible(trackingInfo_input, "trackingInfo")
        
        # Handle column name variations
        if COL_FRAME in trackingInfo.columns and 'second' in trackingInfo.columns:
            trackingInfo = trackingInfo.rename(columns={COL_FRAME: 'second_temp', 'second': COL_FRAME, 'second_temp': 'second'})
        
        required_cols_ti = [COL_VIDEO, COL_TAG, COL_ACTUAL]
        if not all(col in trackingInfo.columns for col in required_cols_ti):
            raise ValueError(f"trackingInfo missing required columns: {required_cols_ti}. Available: {list(trackingInfo.columns)}")
        trackingInfo = trackingInfo[required_cols_ti].copy()
        
        if not pd.api.types.is_numeric_dtype(trackingInfo[COL_ACTUAL]):
            trackingInfo[COL_ACTUAL] = trackingInfo[COL_ACTUAL].astype(int)
        trackingInfo = trackingInfo.groupby([COL_VIDEO, COL_TAG])[COL_ACTUAL].max().reset_index()
        print(f"✅ Processed trackingInfo: {trackingInfo.shape}")
        
        print("✅ Data loading and validation complete!")
        return definitiveObjects, resnetPredictions, trackingInfo
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        raise

# Keep backward compatibility with old function signature for any existing code
def load_and_validate_data_legacy(
    definitiveObjects_path: str, 
    resnetPredictions_path: str, 
    trackingInfo_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Legacy function for backward compatibility. Use load_and_validate_data instead.
    This function maintains the old string-only interface.
    """
    return load_and_validate_data(definitiveObjects_path, resnetPredictions_path, trackingInfo_path)