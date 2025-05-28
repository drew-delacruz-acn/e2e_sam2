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
import subprocess
import shutil
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
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of contrastive learning epochs (default: 50)')
    parser.add_argument('--margin', type=float, default=0.2,
                      help='Contrastive learning margin (default: 0.2)')
    
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
        
        # Convert frame to int to ensure consistency
        resnetPredictions['frame'] = resnetPredictions['frame'].astype(int)
        
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


def train_contrastive_representatives(training_data: pd.DataFrame, 
                                   output_dir: Path,
                                   epochs: int = 50,
                                   margin: float = 0.2) -> Path:
    """
    Train contrastive learning representatives using existing train_representatives.py.
    
    Args:
        training_data: DataFrame with 'class' and 'finetuned_embedding' columns
        output_dir: Directory to save results
        epochs: Number of training epochs
        margin: Contrastive learning margin
        
    Returns:
        Path to the saved representatives.pkl file
    """
    print(f"🏋️ Training contrastive representatives...")
    print(f"   📊 Training data: {len(training_data)} samples, {training_data['class'].nunique()} classes")
    
    # Create temporary training data file
    temp_data_path = output_dir / "temp_training_data.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data in the format expected by train_representatives.py
    with open(temp_data_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    # Run train_representatives.py
    cmd = [
        "python", "train_representatives.py",
        "--data", str(temp_data_path),
        "--output", str(output_dir),
        "--epochs", str(epochs),
        "--margin", str(margin),
        "--no-auto-name"  # Use the provided output directory directly
    ]
    
    print(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              cwd=Path(__file__).parent,
                              timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            print(f"❌ Training failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"train_representatives.py failed with return code {result.returncode}")
        
        print(f"✅ Training completed successfully")
        if result.stdout:
            # Print last few lines of output for progress indication
            lines = result.stdout.strip().split('\n')
            for line in lines[-3:]:
                if line.strip():
                    print(f"   {line}")
        
    except subprocess.TimeoutExpired:
        print(f"❌ Training timed out after 5 minutes")
        raise
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        # Clean up temporary file
        if temp_data_path.exists():
            temp_data_path.unlink()
    
    # Check if representatives.pkl was created
    representatives_path = output_dir / "representatives.pkl"
    if not representatives_path.exists():
        raise FileNotFoundError(f"Expected output file not found: {representatives_path}")
    
    print(f"✅ Representatives saved to: {representatives_path}")
    return representatives_path


def cosine_similarity_prediction(embedding: np.ndarray, 
                               class_embeddings: List[np.ndarray], 
                               class_names: List[str]) -> Tuple[str, float]:
    """
    Predict class using cosine similarity.
    
    Args:
        embedding: Input embedding
        class_embeddings: List of class representative embeddings
        class_names: List of class names corresponding to embeddings
        
    Returns:
        Tuple of (predicted_class, confidence_score)
    """
    # Convert to tensors and normalize
    input_tensor = F.normalize(torch.tensor(embedding, dtype=torch.float32).unsqueeze(0), dim=1)
    class_tensor = F.normalize(torch.tensor(np.vstack(class_embeddings), dtype=torch.float32), dim=1)
    
    # Compute cosine similarities
    similarities = F.cosine_similarity(input_tensor, class_tensor).numpy()
    
    # Get best match
    best_idx = np.argmax(similarities)
    return class_names[best_idx], similarities[best_idx]


def generate_predictions(resnet_data: pd.DataFrame,
                        representatives_path: Path,
                        threshold: float) -> pd.DataFrame:
    """
    Generate predictions using trained representatives.
    
    Args:
        resnet_data: Prediction data with embeddings
        representatives_path: Path to representatives.pkl
        threshold: Confidence threshold for predictions
        
    Returns:
        DataFrame with predictions added
    """
    print(f"🔮 Generating predictions with threshold {threshold}...")
    
    # Load representatives
    with open(representatives_path, 'rb') as f:
        representatives = pickle.load(f)
    
    # Handle different formats
    if isinstance(representatives, dict):
        if 'finetuned_embedding' in representatives:
            class_embeddings = list(representatives['finetuned_embedding'])
            class_names = list(representatives['class'])
        elif 'representative_embedding' in representatives:
            class_embeddings = list(representatives['representative_embedding'])
            class_names = list(representatives['class'])
        else:
            raise ValueError("Unexpected representatives format")
    else:
        # Assume DataFrame
        if 'finetuned_embedding' in representatives.columns:
            class_embeddings = list(representatives['finetuned_embedding'])
            class_names = list(representatives['class'])
        elif 'representative_embedding' in representatives.columns:
            class_embeddings = list(representatives['representative_embedding'])
            class_names = list(representatives['class'])
        else:
            raise ValueError("Unexpected representatives format")
    
    print(f"🏷️  Loaded {len(class_names)} representative classes: {class_names}")
    
    # Generate predictions for all data
    predictions = []
    for _, row in resnet_data.iterrows():
        pred_class, confidence = cosine_similarity_prediction(
            row['finetuned_embedding'], 
            class_embeddings, 
            class_names
        )
        predictions.append({
            'visual_predicted_object': pred_class,
            'visual_max_score': confidence
        })
    
    # Add predictions to dataframe
    pred_df = pd.DataFrame(predictions)
    result_df = resnet_data.copy()
    result_df[['visual_predicted_object', 'visual_max_score']] = pred_df
    
    print(f"✅ Generated {len(result_df)} predictions")
    
    # Keep highest scoring prediction per video-object pair
    print("🔄 Deduplicating: keeping highest scoring prediction per video-class pair...")
    idx = result_df.groupby(['video', 'visual_predicted_object'])['visual_max_score'].idxmax()
    result_df = result_df.loc[idx].reset_index(drop=True)
    print(f"✅ Reduced to {len(result_df)} unique video-class predictions")
    
    # Apply threshold filtering
    print(f"🎯 Applying threshold: {threshold}")
    above_threshold = result_df['visual_max_score'] > threshold
    result_df = result_df[above_threshold].copy()
    print(f"✅ {len(result_df)} predictions above threshold")
    
    return result_df


def evaluate_predictions(predictions: pd.DataFrame,
                        ground_truth: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions: DataFrame with predictions
        ground_truth: Ground truth DataFrame
        
    Returns:
        Tuple of (evaluation_results, metrics)
    """
    print("📊 Evaluating predictions against ground truth...")
    
    # Create evaluation matrix for all video-class combinations in ground truth
    eval_results = []
    
    for _, gt_row in ground_truth.iterrows():
        video = gt_row['video']
        true_class = gt_row['tag']
        actual = gt_row['actual']
        
        # Check if we have a prediction for this video-class pair
        pred_match = predictions[
            (predictions['video'] == video) & 
            (predictions['visual_predicted_object'] == true_class)
        ]
        
        if len(pred_match) > 0:
            # We predicted this class for this video
            predicted = 1
            confidence = pred_match['visual_max_score'].iloc[0]
            frame = pred_match['frame'].iloc[0]
        else:
            # We didn't predict this class for this video
            predicted = 0
            confidence = 0.0
            frame = None
        
        # Classify result
        if actual == 0 and predicted == 0:
            classification = 'TN'
        elif actual == 1 and predicted == 1:
            classification = 'TP'
        elif actual == 1 and predicted == 0:
            classification = 'FN'
        elif actual == 0 and predicted == 1:
            classification = 'FP'
        
        eval_results.append({
            'video': video,
            'class': true_class,
            'actual': actual,
            'predicted': predicted,
            'confidence': confidence,
            'frame': frame,
            'classification': classification
        })
    
    eval_df = pd.DataFrame(eval_results)
    
    # Calculate metrics
    y_true = eval_df['actual']
    y_pred = eval_df['predicted']
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Count classifications
    classification_counts = eval_df['classification'].value_counts()
    
    # Convert to regular Python integers for JSON serialization
    metrics = {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(classification_counts.get('TP', 0)),
        'fp': int(classification_counts.get('FP', 0)),
        'fn': int(classification_counts.get('FN', 0)),
        'tn': int(classification_counts.get('TN', 0))
    }
    
    print(f"📊 Evaluation Results:")
    print(f"   🎯 F1 Score: {f1:.4f}")
    print(f"   🎯 Precision: {precision:.4f}")
    print(f"   🎯 Recall: {recall:.4f}")
    print(f"   📈 TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
    
    return eval_df, metrics


def extract_false_positives(evaluation_results: pd.DataFrame,
                           predictions: pd.DataFrame,
                           exclusion_tracker: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Extract false positive cases and prepare them for training.
    
    Args:
        evaluation_results: Results from evaluation
        predictions: Original predictions DataFrame
        exclusion_tracker: Tracker for excluded data
        
    Returns:
        Tuple of (false_positive_data, exclusion_list)
    """
    print("🚨 Extracting false positives...")
    
    # Get false positive cases
    fp_cases = evaluation_results[evaluation_results['classification'] == 'FP'].copy()
    
    if len(fp_cases) == 0:
        print("✅ No false positives found!")
        return pd.DataFrame(columns=['class', 'finetuned_embedding']), []
    
    print(f"🚨 Found {len(fp_cases)} false positive cases")
    
    # Extract embeddings for false positives
    fp_data = []
    exclusion_list = []
    
    for _, fp_row in fp_cases.iterrows():
        video = fp_row['video']
        predicted_class = fp_row['class']  # This is the TRUE class (tag), not predicted class
        frame = fp_row['frame']
        
        if frame is None:
            print(f"⚠️  Skipping FP case with no frame: {video}, {predicted_class}")
            continue
        
        # Find the original prediction that caused this false positive
        # We need to find where we predicted this class for this video
        original_pred = predictions[
            (predictions['video'] == video) & 
            (predictions['visual_predicted_object'] == predicted_class)
        ]
        
        if len(original_pred) == 0:
            print(f"⚠️  Could not find original prediction for FP: {video}, {predicted_class}")
            continue
        
        # Use the first match (there should only be one after deduplication)
        orig_row = original_pred.iloc[0]
        
        # Add to false positive training data with TRUE class label
        fp_data.append({
            'class': predicted_class,  # Use the TRUE class (from ground truth)
            'finetuned_embedding': orig_row['finetuned_embedding']
        })
        
        # Track for exclusion
        exclusion_entry = {
            'video': video,
            'frame': int(orig_row['frame']),
            'class': predicted_class,
            'reason': 'false_positive'
        }
        exclusion_list.append(exclusion_entry)
    
    fp_df = pd.DataFrame(fp_data)
    
    print(f"✅ Extracted {len(fp_df)} false positive embeddings")
    print(f"📍 Created {len(exclusion_list)} exclusion entries")
    
    if len(fp_df) > 0:
        print(f"🏷️  FP classes: {fp_df['class'].value_counts().to_dict()}")
    
    return fp_df, exclusion_list


def run_single_iteration(iteration: int,
                        training_data: pd.DataFrame,
                        resnet_data: pd.DataFrame,
                        ground_truth: pd.DataFrame,
                        exclusion_tracker: Dict,
                        args) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    """
    Run a single iteration of the pipeline.
    
    Args:
        iteration: Iteration number
        training_data: Current training data
        resnet_data: Prediction data
        ground_truth: Ground truth data
        exclusion_tracker: Exclusion tracking dict
        args: Command line arguments
        
    Returns:
        Tuple of (new_training_data, metrics, new_exclusions)
    """
    print(f"\n🔄 ITERATION {iteration}")
    print("=" * 50)
    
    # Create iteration output directory
    output_dir = Path(args.output) / f"iteration_{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 2A: Train representatives
    print(f"\n🏋️ Phase 2A: Training representatives...")
    representatives_path = train_contrastive_representatives(
        training_data, output_dir, args.epochs, args.margin
    )
    
    # Phase 2B: Generate predictions
    print(f"\n🔮 Phase 2B: Generating predictions...")
    predictions = generate_predictions(resnet_data, representatives_path, args.threshold)
    
    # Phase 2C: Evaluate against ground truth
    print(f"\n📊 Phase 2C: Evaluating predictions...")
    evaluation_results, metrics = evaluate_predictions(predictions, ground_truth)
    
    # Phase 2D: Extract false positives
    print(f"\n🚨 Phase 2D: Extracting false positives...")
    fp_data, new_exclusions = extract_false_positives(evaluation_results, predictions, exclusion_tracker)
    
    # Save iteration results
    print(f"\n💾 Saving iteration results...")
    
    # Save evaluation results
    evaluation_results.to_csv(output_dir / "evaluation_results.csv", index=False)
    
    # Save false positives
    if len(fp_data) > 0:
        fp_data.to_csv(output_dir / "false_positives.csv", index=False)
        with open(output_dir / "false_positives.pkl", 'wb') as f:
            pickle.dump(fp_data, f)
    
    # Save exclusions
    with open(output_dir / "exclusions.json", 'w') as f:
        json.dump(new_exclusions, f, indent=2)
    
    # Save metrics
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create new training data
    if len(fp_data) > 0:
        new_training_data = pd.concat([training_data, fp_data], ignore_index=True)
        print(f"🔄 Updated training data: {len(training_data)} → {len(new_training_data)} samples (+{len(fp_data)})")
    else:
        new_training_data = training_data.copy()
        print(f"🔄 No false positives to add, training data unchanged: {len(training_data)} samples")
    
    # Print iteration summary
    print(f"\n📊 ITERATION {iteration} SUMMARY:")
    print(f"   🎯 F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    print(f"   🚨 False Positives: {metrics['fp']} cases extracted")
    print(f"   📍 Exclusion list: +{len(new_exclusions)} entries")
    print(f"   📈 Training data: {len(training_data)} → {len(new_training_data)} samples")
    
    return new_training_data, metrics, new_exclusions


def filter_evaluation_data(resnet_data: pd.DataFrame,
                          exclusion_tracker: Dict,
                          exclude_training: bool = True) -> pd.DataFrame:
    """
    Filter evaluation data based on exclusion strategy.
    
    Args:
        resnet_data: Original prediction data
        exclusion_tracker: Dictionary tracking excluded video-frame pairs
        exclude_training: Whether to exclude training data from evaluation
        
    Returns:
        Filtered DataFrame for evaluation
    """
    if not exclude_training or not exclusion_tracker:
        return resnet_data.copy()
    
    # Collect all excluded video-frame pairs
    excluded_pairs = set()
    total_exclusions = 0
    
    for iteration, exclusions in exclusion_tracker.items():
        for exclusion in exclusions:
            pair = (exclusion['video'], exclusion['frame'])
            excluded_pairs.add(pair)
            total_exclusions += 1
    
    if total_exclusions == 0:
        print("🚫 No exclusions to apply")
        return resnet_data.copy()
    
    # Filter out excluded pairs
    def is_not_excluded(row):
        pair = (row['video'], row['frame'])
        return pair not in excluded_pairs
    
    original_count = len(resnet_data)
    filtered_data = resnet_data[resnet_data.apply(is_not_excluded, axis=1)].copy()
    filtered_count = len(filtered_data)
    excluded_count = original_count - filtered_count
    
    print(f"🚫 Excluded {excluded_count} training samples from evaluation")
    print(f"📊 Evaluation data: {original_count} → {filtered_count} samples")
    
    return filtered_data


def run_iterative_pipeline(definitiveObjects: pd.DataFrame,
                          resnetPredictions: pd.DataFrame,
                          trackingInfo: pd.DataFrame,
                          args) -> Dict:
    """
    Run the complete iterative pipeline.
    
    Args:
        definitiveObjects: Training objects data
        resnetPredictions: Prediction data
        trackingInfo: Ground truth data
        args: Command line arguments
        
    Returns:
        Dictionary with pipeline summary
    """
    print(f"\n📋 MILESTONE 3: Multi-Iteration Loop")
    print("-" * 40)
    
    # Initialize tracking
    exclusion_tracker = {}
    current_training_data = definitiveObjects.copy()
    metrics_history = []
    converged = False
    
    # Create main output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for iteration in range(1, args.iterations + 1):
        print(f"\n🔄 ITERATION {iteration}")
        print("=" * 50)
        
        # Apply exclusion strategy to evaluation data
        if args.exclude_training_from_eval:
            eval_data = filter_evaluation_data(resnetPredictions, exclusion_tracker, exclude_training=True)
            eval_mode = "clean"
        elif args.include_training_in_eval:
            eval_data = resnetPredictions.copy()
            eval_mode = "contaminated"
            print("⚠️  Including training data in evaluation (contaminated metrics)")
        elif args.track_training_separately:
            # We'll handle this after running the iteration
            eval_data = resnetPredictions.copy()
            eval_mode = "both"
        else:
            eval_data = filter_evaluation_data(resnetPredictions, exclusion_tracker, exclude_training=True)
            eval_mode = "clean"
        
        # Run single iteration
        new_training_data, metrics, new_exclusions = run_single_iteration(
            iteration=iteration,
            training_data=current_training_data,
            resnet_data=eval_data,
            ground_truth=trackingInfo,
            exclusion_tracker=exclusion_tracker,
            args=args
        )
        
        # Update exclusion tracker
        if new_exclusions:
            exclusion_tracker[f'iteration_{iteration}'] = new_exclusions
        
        # Add iteration info to metrics
        metrics['iteration'] = iteration
        metrics['training_samples'] = len(current_training_data)
        metrics['new_training_samples'] = len(new_training_data)
        metrics['evaluation_samples'] = len(eval_data)
        metrics['eval_mode'] = eval_mode
        metrics_history.append(metrics)
        
        # Handle track_training_separately mode
        if args.track_training_separately:
            # Run clean evaluation
            clean_eval_data = filter_evaluation_data(resnetPredictions, exclusion_tracker, exclude_training=True)
            
            if len(clean_eval_data) != len(eval_data):
                print(f"\n🔄 Running clean evaluation for comparison...")
                
                # Load representatives and generate clean predictions
                representatives_path = output_dir / f"iteration_{iteration}" / "representatives.pkl"
                clean_predictions = generate_predictions(clean_eval_data, representatives_path, args.threshold)
                clean_eval_results, clean_metrics = evaluate_predictions(clean_predictions, trackingInfo)
                
                print(f"📊 CLEAN vs CONTAMINATED COMPARISON:")
                print(f"   Clean F1: {clean_metrics['f1']:.4f} ({len(clean_eval_data)} samples)")
                print(f"   Contaminated F1: {metrics['f1']:.4f} ({len(eval_data)} samples)")
                
                # Save clean metrics
                clean_metrics['iteration'] = iteration
                clean_metrics['eval_mode'] = 'clean'
                clean_metrics['evaluation_samples'] = len(clean_eval_data)
                
                with open(output_dir / f"iteration_{iteration}" / "clean_metrics.json", 'w') as f:
                    json.dump(clean_metrics, f, indent=2)
        
        # Check for convergence
        if iteration > 1:
            prev_f1 = metrics_history[-2]['f1']
            current_f1 = metrics['f1']
            f1_improvement = current_f1 - prev_f1
            
            print(f"\n📈 Convergence Check:")
            print(f"   Previous F1: {prev_f1:.4f}")
            print(f"   Current F1: {current_f1:.4f}")
            print(f"   Improvement: {f1_improvement:+.4f}")
            print(f"   Threshold: {args.convergence_threshold:.4f}")
            
            if f1_improvement < args.convergence_threshold:
                print(f"🎯 Converged after {iteration} iterations (improvement < {args.convergence_threshold})")
                converged = True
            else:
                print(f"🔄 Continuing (improvement >= {args.convergence_threshold})")
        
        # Update training data for next iteration
        current_training_data = new_training_data
        
        # Print iteration summary
        total_exclusions = sum(len(exclusions) for exclusions in exclusion_tracker.values())
        print(f"\n📊 ITERATION {iteration} SUMMARY:")
        print(f"   🎯 F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"   📊 Eval samples: {len(eval_data)} ({eval_mode})")
        print(f"   🚨 False Positives: {metrics['fp']} cases extracted")
        print(f"   📍 Total excluded: {total_exclusions} entries")
        print(f"   📈 Training data: {len(definitiveObjects)} → {len(current_training_data)} samples")
        
        if converged:
            break
    
    # Generate pipeline summary
    final_iteration = len(metrics_history)
    final_metrics = metrics_history[-1]
    initial_samples = len(definitiveObjects)
    final_samples = len(current_training_data)
    total_exclusions = sum(len(exclusions) for exclusions in exclusion_tracker.values())
    
    pipeline_summary = {
        'pipeline_info': {
            'total_iterations': final_iteration,
            'converged': converged,
            'convergence_threshold': args.convergence_threshold,
            'eval_strategy': 'clean' if args.exclude_training_from_eval else 'contaminated' if args.include_training_in_eval else 'both'
        },
        'data_summary': {
            'initial_training_samples': initial_samples,
            'final_training_samples': final_samples,
            'samples_added': final_samples - initial_samples,
            'total_exclusions': total_exclusions
        },
        'final_metrics': final_metrics,
        'metrics_progression': metrics_history
    }
    
    # Save pipeline summary
    with open(output_dir / "pipeline_summary.json", 'w') as f:
        json.dump(pipeline_summary, f, indent=2)
    
    # Save metrics progression CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(output_dir / "metrics_comparison.csv", index=False)
    
    # Save cumulative exclusions
    with open(output_dir / "cumulative_exclusions.json", 'w') as f:
        json.dump(exclusion_tracker, f, indent=2)
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"=" * 50)
    print(f"📊 Final Results:")
    print(f"   🔄 Iterations: {final_iteration}")
    print(f"   🎯 Final F1: {final_metrics['f1']:.4f}")
    print(f"   📈 Training samples: {initial_samples} → {final_samples} (+{final_samples - initial_samples})")
    print(f"   📍 Total exclusions: {total_exclusions}")
    print(f"   🎯 Converged: {'Yes' if converged else 'No'}")
    print(f"   📁 Results saved to: {output_dir}")
    
    return pipeline_summary


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
        
        # === MILESTONE 2 & 3: Complete Pipeline ===
        if args.iterations == 1:
            print("\n📋 MILESTONE 2: Single Iteration Training")
            print("-" * 40)
            
            # Initialize tracking
            exclusion_tracker = {}
            current_training_data = definitiveObjects.copy()
            
            # Run single iteration
            new_training_data, metrics, new_exclusions = run_single_iteration(
                iteration=1,
                training_data=current_training_data,
                resnet_data=resnetPredictions,
                ground_truth=trackingInfo,
                exclusion_tracker=exclusion_tracker,
                args=args
            )
            
            # Update exclusion tracker
            exclusion_tracker['iteration_1'] = new_exclusions
            
            print(f"\n✅ MILESTONE 2 COMPLETE!")
            print(f"   🎯 Iteration 1 F1: {metrics['f1']:.4f}")
            print(f"   🚨 False positives extracted: {metrics['fp']}")
            print(f"   📈 Training data updated: {len(definitiveObjects)} → {len(new_training_data)} samples")
            print(f"   📁 Results saved to: {Path(args.output) / 'iteration_1'}")
        else:
            # Run full iterative pipeline
            pipeline_summary = run_iterative_pipeline(
                definitiveObjects, resnetPredictions, trackingInfo, args
            )
            
            print(f"\n✅ MILESTONE 3 COMPLETE!")
            print(f"   🔄 Total iterations: {pipeline_summary['pipeline_info']['total_iterations']}")
            print(f"   🎯 Final F1: {pipeline_summary['final_metrics']['f1']:.4f}")
            print(f"   🎯 Converged: {pipeline_summary['pipeline_info']['converged']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 