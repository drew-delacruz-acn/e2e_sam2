#!/usr/bin/env python3
"""
Iterative Contrastive Learning Pipeline

This pipeline implements an iterative approach to improve class representatives by:
1. Training contrastive learning models on current data
2. Evaluating against predictions using cosine similarity  
3. Extracting false positives and adding them to training data
4. Iterating until convergence

Usage:
    python iterative_contrastive_pipeline.py \
        --definitive-objects path/to/definitiveObjects.pkl \
        --resnet-predictions path/to/resnetPredictions.pkl \
        --tracking-info path/to/trackingInfo.pkl \
        --iterations 5 \
        --threshold 0.6
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Iterative Contrastive Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data files
    parser.add_argument('--definitive-objects', type=str, required=True,
                      help='Path to definitiveObjects.pkl file (training objects)')
    parser.add_argument('--resnet-predictions', type=str, required=True,
                      help='Path to resnetPredictions.pkl file (prediction data)')
    parser.add_argument('--tracking-info', type=str, required=True,
                      help='Path to trackingInfo.pkl file (ground truth)')
    
    # Pipeline parameters
    parser.add_argument('--iterations', type=int, default=5,
                      help='Maximum number of iterations (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.6,
                      help='Confidence threshold for predictions (default: 0.6)')
    parser.add_argument('--convergence-threshold', type=float, default=0.001,
                      help='F1 improvement threshold for convergence (default: 0.001)')
    
    # Exclusion strategy options
    exclusion_group = parser.add_mutually_exclusive_group()
    exclusion_group.add_argument('--exclude-training-from-eval', action='store_true', default=True,
                               help='Exclude training data from evaluation (default, recommended for research)')
    exclusion_group.add_argument('--include-training-in-eval', action='store_true',
                               help='Include training data in evaluation (useful for debugging)')
    exclusion_group.add_argument('--track-training-separately', action='store_true',
                               help='Track both clean and contaminated metrics')
    
    # Output options
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results (default: results)')
    parser.add_argument('--test-mode', action='store_true',
                      help='Run in test mode (only data loading and validation)')
    
    return parser.parse_args()


def load_and_validate_data(definitiveObjects_path: str, 
                          resnetPredictions_path: str, 
                          trackingInfo_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate the three required data files.
    
    Args:
        definitiveObjects_path: Path to definitiveObjects.pkl
        resnetPredictions_path: Path to resnetPredictions.pkl  
        trackingInfo_path: Path to trackingInfo.pkl
        
    Returns:
        Tuple of (definitiveObjects, resnetPredictions, trackingInfo) DataFrames
        
    Raises:
        FileNotFoundError: If any file doesn't exist
        ValueError: If required columns are missing
    """
    print("🔄 Loading and validating data files...")
    
    # Load definitiveObjects
    print(f"📂 Loading definitiveObjects from: {definitiveObjects_path}")
    try:
        with open(definitiveObjects_path, 'rb') as f:
            definitiveObjects = pickle.load(f)
        
        if isinstance(definitiveObjects, dict):
            # Convert dict to DataFrame if needed
            definitiveObjects = pd.DataFrame(definitiveObjects)
        
        # Validate required columns
        required_cols = ['class', 'finetuned_embedding']
        missing_cols = [col for col in required_cols if col not in definitiveObjects.columns]
        if missing_cols:
            print(f"❌ Missing columns in definitiveObjects: {missing_cols}")
            print(f"📊 Available columns: {list(definitiveObjects.columns)}")
            raise ValueError(f"definitiveObjects missing required columns: {missing_cols}")
        
        # Extract only the required columns
        definitiveObjects = definitiveObjects[required_cols].copy()
        
        print(f"✅ Loaded definitiveObjects: {definitiveObjects.shape} → using {required_cols}")
        print(f"📊 Class distribution: {definitiveObjects['class'].value_counts().head()}")
        
    except Exception as e:
        print(f"❌ Error loading definitiveObjects: {e}")
        raise
    
    # Load resnetPredictions  
    print(f"📂 Loading resnetPredictions from: {resnetPredictions_path}")
    try:
        with open(resnetPredictions_path, 'rb') as f:
            resnetPredictions = pickle.load(f)
        
        if isinstance(resnetPredictions, dict):
            resnetPredictions = pd.DataFrame(resnetPredictions)
        
        # Validate required columns
        required_cols = ['video', 'frame', 'owl_label', 'finetuned_embedding']
        missing_cols = [col for col in required_cols if col not in resnetPredictions.columns]
        if missing_cols:
            print(f"❌ Missing columns in resnetPredictions: {missing_cols}")
            print(f"📊 Available columns: {list(resnetPredictions.columns)}")
            raise ValueError(f"resnetPredictions missing required columns: {missing_cols}")
        
        # Extract only the required columns
        resnetPredictions = resnetPredictions[required_cols].copy()
        
        print(f"✅ Loaded resnetPredictions: {resnetPredictions.shape} → using {required_cols}")
        print(f"📊 Videos: {resnetPredictions['video'].nunique()}, Labels: {resnetPredictions['owl_label'].nunique()}")
        
    except Exception as e:
        print(f"❌ Error loading resnetPredictions: {e}")
        raise
    
    # Load trackingInfo
    print(f"📂 Loading trackingInfo from: {trackingInfo_path}")
    try:
        with open(trackingInfo_path, 'rb') as f:
            trackingInfo = pickle.load(f)
        
        if isinstance(trackingInfo, dict):
            trackingInfo = pd.DataFrame(trackingInfo)
        
        # Handle potential column naming variations
        if 'frame' in trackingInfo.columns and 'second' in trackingInfo.columns:
            # Swap if needed (based on the code patterns I saw)
            trackingInfo = trackingInfo.rename(columns={'frame': 'second_temp', 'second': 'frame', 'second_temp': 'second'})
        
        # Validate required columns  
        required_cols = ['video', 'tag', 'actual']
        missing_cols = [col for col in required_cols if col not in trackingInfo.columns]
        if missing_cols:
            print(f"❌ Missing columns in trackingInfo: {missing_cols}")
            print(f"📊 Available columns: {list(trackingInfo.columns)}")
            raise ValueError(f"trackingInfo missing required columns: {missing_cols}")
        
        # Extract only the required columns
        trackingInfo = trackingInfo[required_cols].copy()
        
        # Group by video and tag to get max actual value (as seen in existing code)
        trackingInfo = trackingInfo.groupby(['video', 'tag'])['actual'].max().reset_index()
        
        print(f"✅ Loaded trackingInfo: {trackingInfo.shape} → using {required_cols}")
        print(f"📊 Videos: {trackingInfo['video'].nunique()}, Tags: {trackingInfo['tag'].nunique()}")
        print(f"📊 Actual distribution: {trackingInfo['actual'].value_counts()}")
        
    except Exception as e:
        print(f"❌ Error loading trackingInfo: {e}")
        raise
    
    # Validation checks
    print("\n🔍 Cross-validation checks...")
    
    # Check class overlap
    definitive_classes = set(definitiveObjects['class'].unique())
    prediction_classes = set(resnetPredictions['owl_label'].unique())
    tracking_classes = set(trackingInfo['tag'].unique())
    
    print(f"📊 Definitive classes: {len(definitive_classes)}")
    print(f"📊 Prediction classes: {len(prediction_classes)}")
    print(f"📊 Tracking classes: {len(tracking_classes)}")
    
    # Check for embedding consistency
    if len(definitiveObjects) > 0:
        sample_emb = definitiveObjects['finetuned_embedding'].iloc[0]
        if isinstance(sample_emb, (list, np.ndarray)):
            emb_dim = len(sample_emb)
            print(f"📊 Embedding dimension: {emb_dim}")
        else:
            print(f"⚠️  Unexpected embedding type: {type(sample_emb)}")
    
    print("✅ Data loading and validation complete!")
    return definitiveObjects, resnetPredictions, trackingInfo


def main():
    """Main pipeline function."""
    args = parse_args()
    
    print("🚀 Starting Iterative Contrastive Learning Pipeline")
    print("="*60)
    
    # === MILESTONE 1: Basic Data Pipeline ===
    print("\n📋 MILESTONE 1: Basic Data Pipeline")
    print("-" * 40)
    
    try:
        definitiveObjects, resnetPredictions, trackingInfo = load_and_validate_data(
            args.definitive_objects,
            args.resnet_predictions, 
            args.tracking_info
        )
        
        # Print summary statistics
        print(f"\n📈 Data Summary:")
        print(f"   • Training objects: {len(definitiveObjects):,} samples across {definitiveObjects['class'].nunique()} classes")
        print(f"   • Prediction data: {len(resnetPredictions):,} predictions across {resnetPredictions['video'].nunique()} videos")
        print(f"   • Ground truth: {len(trackingInfo):,} video-class pairs")
        
        if args.test_mode:
            print("\n✅ Test mode complete - data loading successful!")
            return
        
        # TODO: Continue with Milestone 2
        print("\n🔄 Ready for Milestone 2: Single Iteration Training")
        print("   (Implementation coming next...)")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 