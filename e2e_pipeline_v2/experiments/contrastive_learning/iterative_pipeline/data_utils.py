import pickle
import pandas as pd
from pathlib import Path
from typing import Tuple
import traceback

# Constants for column names
COL_CLASS = 'class'
COL_EMBEDDING = 'finetuned_embedding'
COL_VIDEO = 'video'
COL_FRAME = 'frame'
COL_OWL_LABEL = 'owl_label'
COL_TAG = 'tag'
COL_ACTUAL = 'actual'

def load_and_validate_data(definitiveObjects_path: str, 
                          resnetPredictions_path: str, 
                          trackingInfo_path: str,
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate the three required data files.
    """ 
    print("🔄 Loading and validating data files...")
    
    # Load definitiveObjects
    print(f"📂 Loading definitiveObjects from: {definitiveObjects_path}")
    try:
        with open(definitiveObjects_path, 'rb') as f:
            definitiveObjects = pickle.load(f)
        if isinstance(definitiveObjects, dict):
            definitiveObjects = pd.DataFrame(definitiveObjects)
        required_cols_do = [COL_CLASS, COL_EMBEDDING]
        if not all(col in definitiveObjects.columns for col in required_cols_do):
            raise ValueError(f"definitiveObjects missing one of required columns: {required_cols_do}. Available: {list(definitiveObjects.columns)}")
        definitiveObjects = definitiveObjects[required_cols_do].copy()
        print(f"✅ Loaded definitiveObjects: {definitiveObjects.shape}")
    except Exception as e: 
        print(f"❌ Error loading definitiveObjects: {e}")
        traceback.print_exc()
        raise
    
    # Load resnetPredictions
    print(f"📂 Loading resnetPredictions from: {resnetPredictions_path}")
    try:
        with open(resnetPredictions_path, 'rb') as f:
            resnetPredictions = pickle.load(f)
        if isinstance(resnetPredictions, dict):
            resnetPredictions = pd.DataFrame(resnetPredictions)
        
        required_cols_rp = [COL_VIDEO, COL_FRAME, COL_OWL_LABEL, COL_EMBEDDING]
        
        if COL_OWL_LABEL not in resnetPredictions.columns and COL_CLASS in resnetPredictions.columns:
            print(f"ℹ️  '{COL_OWL_LABEL}' not found, using '{COL_CLASS}' column from resnetPredictions and renaming to '{COL_OWL_LABEL}'.")
            resnetPredictions = resnetPredictions.rename(columns={COL_CLASS: COL_OWL_LABEL})
        
        if not all(col in resnetPredictions.columns for col in required_cols_rp):
            raise ValueError(f"resnetPredictions missing one of required columns: {required_cols_rp}. Available: {list(resnetPredictions.columns)}")

        resnetPredictions = resnetPredictions[required_cols_rp].copy()
        resnetPredictions[COL_FRAME] = resnetPredictions[COL_FRAME].astype(int)
        print(f"✅ Loaded resnetPredictions: {resnetPredictions.shape}")
    except Exception as e: 
        print(f"❌ Error loading resnetPredictions: {e}")
        traceback.print_exc()
        raise
        
    # Load trackingInfo
    print(f"📂 Loading trackingInfo from: {trackingInfo_path}")
    try:
        with open(trackingInfo_path, 'rb') as f:
            trackingInfo = pickle.load(f)
        if isinstance(trackingInfo, dict):
            trackingInfo = pd.DataFrame(trackingInfo)
        
        if COL_FRAME in trackingInfo.columns and 'second' in trackingInfo.columns:
            trackingInfo = trackingInfo.rename(columns={COL_FRAME: 'second_temp', 'second': COL_FRAME, 'second_temp': 'second'})
        
        required_cols_ti = [COL_VIDEO, COL_TAG, COL_ACTUAL]
        if not all(col in trackingInfo.columns for col in required_cols_ti):
            raise ValueError(f"trackingInfo missing one of required columns: {required_cols_ti}. Available: {list(trackingInfo.columns)}")
        trackingInfo = trackingInfo[required_cols_ti].copy()
        
        if not pd.api.types.is_numeric_dtype(trackingInfo[COL_ACTUAL]):
            trackingInfo[COL_ACTUAL] = trackingInfo[COL_ACTUAL].astype(int)
        trackingInfo = trackingInfo.groupby([COL_VIDEO, COL_TAG])[COL_ACTUAL].max().reset_index()
        print(f"✅ Loaded trackingInfo: {trackingInfo.shape}")
    except Exception as e: 
        print(f"❌ Error loading trackingInfo: {e}")
        traceback.print_exc()
        raise
    
    print("✅ Data loading and validation complete!")
    return definitiveObjects, resnetPredictions, trackingInfo