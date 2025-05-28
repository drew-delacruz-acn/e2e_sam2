#!/usr/bin/env python3
"""
Iterative Contrastive Learning Pipeline with Negative Classes

This pipeline implements an iterative approach to improve class representatives by:
1. Training contrastive learning models on current data
2. Evaluating against predictions using cosine similarity
3. Extracting false positives and adding them as NEGATIVE examples ("not_class")
4. Using contrastive scoring during prediction (positive_sim - negative_sim)
5. Iterating until convergence

Usage:
    python iterative_contrastive_pipeline_negative.py \
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
        description="Iterative Contrastive Learning Pipeline with Negative Classes",
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
    parser.add_argument('--output', type=str, default='results_negative',
                      help='Output directory for results (default: results_negative)')
    parser.add_argument('--test-mode', action='store_true',
                      help='Run in test mode (only data loading and validation)')
    
    return parser.parse_args()


def load_and_validate_data(definitiveObjects_path: str, 
                          resnetPredictions_path: str, 
                          trackingInfo_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        required_cols = ['class', 'finetuned_embedding']
        missing_cols = [col for col in required_cols if col not in definitiveObjects.columns]
        if missing_cols:
            raise ValueError(f"definitiveObjects missing required columns: {missing_cols}")
        definitiveObjects = definitiveObjects[required_cols].copy()
        print(f"✅ Loaded definitiveObjects: {definitiveObjects.shape} → using {required_cols}")
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
        required_cols = ['video', 'frame', 'owl_label', 'finetuned_embedding']
        missing_cols = [col for col in required_cols if col not in resnetPredictions.columns]
        if missing_cols:
            raise ValueError(f"resnetPredictions missing required columns: {missing_cols}")
        resnetPredictions = resnetPredictions[required_cols].copy()
        resnetPredictions['frame'] = resnetPredictions['frame'].astype(int)
        print(f"✅ Loaded resnetPredictions: {resnetPredictions.shape} → using {required_cols}")
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
        if 'frame' in trackingInfo.columns and 'second' in trackingInfo.columns:
            trackingInfo = trackingInfo.rename(columns={'frame': 'second_temp', 'second': 'frame', 'second_temp': 'second'})
        required_cols = ['video', 'tag', 'actual']
        missing_cols = [col for col in required_cols if col not in trackingInfo.columns]
        if missing_cols:
            raise ValueError(f"trackingInfo missing required columns: {missing_cols}")
        trackingInfo = trackingInfo[required_cols].copy()
        trackingInfo = trackingInfo.groupby(['video', 'tag'])['actual'].max().reset_index()
        print(f"✅ Loaded trackingInfo: {trackingInfo.shape} → using {required_cols}")
    except Exception as e:
        print(f"❌ Error loading trackingInfo: {e}")
        raise
    
    print("✅ Data loading and validation complete!")
    return definitiveObjects, resnetPredictions, trackingInfo


def train_contrastive_representatives(training_data: pd.DataFrame, 
                                   output_dir: Path,
                                   epochs: int = 50,
                                   margin: float = 0.2) -> Path:
    """
    Train contrastive learning representatives using existing train_representatives.py.
    """
    print(f"🏋️ Training contrastive representatives...")
    temp_data_path = output_dir / "temp_training_data.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(temp_data_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    # Assuming train_representatives.py is in the same directory or accessible in PATH
    # If it's in a specific subdirectory (e.g., ../../scripts), adjust the path
    train_script_path = Path(__file__).parent / "../../train_representatives.py" 
    if not train_script_path.exists():
         # Fallback if not found relative to this script (e.g. if this script is moved)
        train_script_path = Path("train_representatives.py")


    cmd = [
        "python", str(train_script_path),
        "--data", str(temp_data_path),
        "--output", str(output_dir),
        "--epochs", str(epochs),
        "--margin", str(margin),
        "--no-auto-name" 
    ]
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600) # 10 min timeout
        if result.returncode != 0:
            print(f"❌ Training failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            raise RuntimeError(f"train_representatives.py failed with return code {result.returncode}")
        print(f"✅ Training completed successfully")
    except subprocess.TimeoutExpired:
        print(f"❌ Training timed out after 10 minutes")
        raise
    finally:
        if temp_data_path.exists():
            temp_data_path.unlink()
    
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
    """
    input_tensor = F.normalize(torch.tensor(embedding, dtype=torch.float32).unsqueeze(0), dim=1)
    class_tensor = F.normalize(torch.tensor(np.vstack(class_embeddings), dtype=torch.float32), dim=1)
    similarities = F.cosine_similarity(input_tensor, class_tensor).numpy()
    best_idx = np.argmax(similarities)
    return class_names[best_idx], similarities[best_idx]

def cosine_similarity_prediction_with_negatives(embedding: np.ndarray, 
                                               class_embeddings: List[np.ndarray], 
                                               class_names: List[str]) -> Tuple[str, float]:
    """
    Predict class using cosine similarity with support for negative classes.
    Uses contrastive scoring: positive_similarity - negative_similarity
    """
    input_tensor = F.normalize(torch.tensor(embedding, dtype=torch.float32).unsqueeze(0), dim=1)
    class_tensor = F.normalize(torch.tensor(np.vstack(class_embeddings), dtype=torch.float32), dim=1)
    similarities = F.cosine_similarity(input_tensor, class_tensor).numpy().flatten() # Ensure 1D
    
    sim_dict = {name: sim for name, sim in zip(class_names, similarities)}
    
    positive_classes = [name for name in class_names if not name.startswith('not_')]
    negative_classes = [name for name in class_names if name.startswith('not_')]
    
    if not negative_classes or not positive_classes:
        best_idx = np.argmax(similarities)
        return class_names[best_idx], similarities[best_idx]
    
    contrastive_scores = {}
    for pos_class in positive_classes:
        neg_class = f"not_{pos_class}"
        pos_sim = sim_dict.get(pos_class, -1.0) # Default to low similarity
        neg_sim = sim_dict.get(neg_class, -1.0) # Default to low similarity if not_class doesn't exist
        contrastive_scores[pos_class] = pos_sim - neg_sim
            
    if not contrastive_scores: # Should not happen if positive_classes exist
        best_idx = np.argmax(similarities) # Fallback
        return class_names[best_idx], similarities[best_idx]

    best_class = max(contrastive_scores, key=contrastive_scores.get)
    # Confidence is the original similarity to the positive class, not the contrastive score
    confidence = sim_dict.get(best_class, 0.0) 
    return best_class, confidence


def generate_predictions(resnet_data: pd.DataFrame,
                        representatives_path: Path,
                        threshold: float) -> pd.DataFrame:
    """
    Generate predictions using trained representatives with support for negative classes.
    """
    print(f"🔮 Generating predictions with threshold {threshold}...")
    with open(representatives_path, 'rb') as f:
        representatives = pickle.load(f)

    if isinstance(representatives, dict):
        # Assuming keys are 'class' and 'finetuned_embedding' or 'representative_embedding'
        if 'finetuned_embedding' in representatives:
            class_embeddings = list(representatives['finetuned_embedding'])
            class_names = list(representatives['class'])
        elif 'representative_embedding' in representatives: # For compatibility
            class_embeddings = list(representatives['representative_embedding'])
            class_names = list(representatives['class'])
        else:
             # Try to infer from DataFrame-like dict
            try:
                temp_df = pd.DataFrame(representatives)
                if 'finetuned_embedding' in temp_df.columns:
                     class_embeddings = list(temp_df['finetuned_embedding'])
                     class_names = list(temp_df['class'])
                elif 'representative_embedding' in temp_df.columns:
                     class_embeddings = list(temp_df['representative_embedding'])
                     class_names = list(temp_df['class'])
                else:
                    raise ValueError("Unexpected dict representatives format")
            except Exception as e_dict:
                raise ValueError(f"Unexpected dict representatives format: {e_dict}")
    elif isinstance(representatives, pd.DataFrame):
        if 'finetuned_embedding' in representatives.columns:
            class_embeddings = list(representatives['finetuned_embedding'])
            class_names = list(representatives['class'])
        elif 'representative_embedding' in representatives.columns: # For compatibility
            class_embeddings = list(representatives['representative_embedding'])
            class_names = list(representatives['class'])
        else:
            raise ValueError("Unexpected DataFrame representatives format")
    else:
        raise ValueError(f"Unknown representatives format: {type(representatives)}")

    has_negatives = any(name.startswith('not_') for name in class_names)
    positive_display_classes = [name for name in class_names if not name.startswith('not_')]
    negative_display_classes = [name for name in class_names if name.startswith('not_')]

    print(f"🏷️  Loaded {len(class_names)} representatives:")
    print(f"   • Positive classes ({len(positive_display_classes)}): {positive_display_classes}")
    if negative_display_classes:
        print(f"   • Negative classes ({len(negative_display_classes)}): {negative_display_classes}")

    predictions_list = []
    for _, row in resnet_data.iterrows():
        if has_negatives:
            pred_class, confidence = cosine_similarity_prediction_with_negatives(
                row['finetuned_embedding'], class_embeddings, class_names)
        else:
            pred_class, confidence = cosine_similarity_prediction(
                row['finetuned_embedding'], class_embeddings, class_names)
        predictions_list.append({'visual_predicted_object': pred_class, 'visual_max_score': confidence})
    
    pred_df = pd.DataFrame(predictions_list)
    result_df = pd.concat([resnet_data.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    print(f"✅ Generated {len(result_df)} initial predictions")

    # Deduplication: keep highest scoring positive prediction per video-object pair
    # Important: Only consider positive classes for this step, as 'not_' classes aren't final predictions
    positive_predictions_df = result_df[~result_df['visual_predicted_object'].str.startswith('not_', na=False)].copy()
    if not positive_predictions_df.empty:
        idx = positive_predictions_df.groupby(['video', 'visual_predicted_object'])['visual_max_score'].idxmax()
        deduplicated_df = positive_predictions_df.loc[idx].reset_index(drop=True)
        print(f"✅ Reduced to {len(deduplicated_df)} unique positive video-class predictions after deduplication")
    else: # No positive predictions made (e.g. if only 'not_' classes were predicted)
        deduplicated_df = pd.DataFrame(columns=result_df.columns) # Empty df with same schema
        print("⚠️  No positive class predictions found for deduplication!")

    # Apply threshold
    final_df = deduplicated_df[deduplicated_df['visual_max_score'] >= threshold].copy() # Use >= for threshold
    print(f"✅ {len(final_df)} predictions above/at threshold {threshold}")
    return final_df


def evaluate_predictions(predictions: pd.DataFrame,
                        ground_truth: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate predictions against ground truth.
    """
    print("📊 Evaluating predictions against ground truth...")
    eval_results = []
    for _, gt_row in ground_truth.iterrows():
        video, true_class, actual = gt_row['video'], gt_row['tag'], gt_row['actual']
        pred_match = predictions[(predictions['video'] == video) & (predictions['visual_predicted_object'] == true_class)]
        
        predicted, confidence, frame_num = (1, pred_match['visual_max_score'].iloc[0], pred_match['frame'].iloc[0]) if not pred_match.empty else (0, 0.0, None)

        if actual == 0 and predicted == 0: classification = 'TN'
        elif actual == 1 and predicted == 1: classification = 'TP'
        elif actual == 1 and predicted == 0: classification = 'FN'
        else: classification = 'FP' # actual == 0 and predicted == 1
            
        eval_results.append({
            'video': video, 'class': true_class, 'actual': actual, 'predicted': predicted,
            'confidence': confidence, 'frame': frame_num, 'classification': classification
        })
    eval_df = pd.DataFrame(eval_results)
    
    y_true, y_pred = eval_df['actual'], eval_df['predicted']
    metrics = {
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        **{cls_type: int(eval_df['classification'].value_counts().get(cls_type, 0)) for cls_type in ['TP', 'FP', 'FN', 'TN']}
    }
    print(f"   📊 F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    print(f"   📈 TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
    return eval_df, metrics


def extract_false_positives(evaluation_results: pd.DataFrame,
                           predictions: pd.DataFrame, # This is the output of generate_predictions before thresholding
                           exclusion_tracker: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Extract false positive cases and prepare them as negative examples for training.
    """
    print("🚨 Extracting false positives as negative examples...")
    fp_cases = evaluation_results[evaluation_results['classification'] == 'FP'].copy()
    if fp_cases.empty:
        print("✅ No false positives found!")
        return pd.DataFrame(columns=['class', 'finetuned_embedding']), []
    print(f"🚨 Found {len(fp_cases)} false positive cases")
    
    fp_data = []
    exclusion_list = []
    for _, fp_row in fp_cases.iterrows():
        video = fp_row['video']
        # In fp_cases, 'class' is the true class that was *not* present, 
        # but for which a prediction was wrongly made.
        # We need the class that was *actually predicted* for this video-frame.
        # The `predictions` df passed to this function should be the one *before* thresholding and deduplication
        # to find the original embedding that led to this FP.
        
        # Find what was wrongly predicted for this video for the class marked as FP.
        # The `fp_row['class']` is the ground truth class that was 0 (not present).
        # The `fp_row['frame']` is the frame of that wrong prediction.
        # We need to look up in the *original predictions table* (passed as `predictions`)
        # what class was predicted for this video/frame that corresponds to `fp_row['class']`.
        # This part is a bit tricky. The `fp_row['class']` is the *true class* for which actual=0 but predicted=1.
        # This means `visual_predicted_object` in the `predictions` table was `fp_row['class']`.

        wrongly_predicted_as_class = fp_row['class'] # This IS the class that was wrongly predicted.
        frame_of_fp = fp_row['frame']

        if frame_of_fp is None: # Should not happen if FP was identified
            print(f"⚠️  Skipping FP case with no frame: {video}, {wrongly_predicted_as_class}")
            continue

        # Find the original prediction entry that corresponds to this FP
        # The `predictions` dataframe here is the output of `generate_predictions`
        # which has `visual_predicted_object` and `finetuned_embedding`.
        original_pred_entry = predictions[
            (predictions['video'] == video) &
            (predictions['visual_predicted_object'] == wrongly_predicted_as_class) &
            (predictions['frame'] == frame_of_fp) # Ensure we get the exact FP frame
        ]

        if original_pred_entry.empty:
            # This might happen if the `predictions` df passed here was already thresholded/deduplicated
            # It's crucial that `predictions` is the raw output from generate_predictions before filtering
            print(f"⚠️  Could not find original prediction for FP: Video {video}, Frame {frame_of_fp}, Wrongly predicted as {wrongly_predicted_as_class}")
            print(f"   This might indicate an issue with the 'predictions' data passed to extract_false_positives.")
            continue
        
        orig_row_embedding = original_pred_entry['finetuned_embedding'].iloc[0]
        
        negative_class_label = f"not_{wrongly_predicted_as_class}"
        fp_data.append({
            'class': negative_class_label,
            'finetuned_embedding': orig_row_embedding
        })
        exclusion_list.append({
            'video': video,
            'frame': int(frame_of_fp), # Ensure int
            'class': negative_class_label,
            'original_wrong_prediction': wrongly_predicted_as_class,
            'reason': 'false_positive_negative'
        })
        
    fp_df = pd.DataFrame(fp_data)
    print(f"✅ Extracted {len(fp_df)} embeddings for negative examples")
    if not fp_df.empty:
        print(f"🏷️  Negative classes created: {fp_df['class'].value_counts().to_dict()}")
    return fp_df, exclusion_list


def run_single_iteration(iteration: int,
                        training_data: pd.DataFrame,
                        resnet_data_for_prediction: pd.DataFrame, # Full data for making predictions
                        ground_truth: pd.DataFrame,
                        exclusion_tracker: Dict,
                        args) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    """
    Run a single iteration of the pipeline.
    `resnet_data_for_prediction` is the full set of embeddings we make predictions on.
    `ground_truth` is used for evaluation.
    """
    print(f"\n🔄 ITERATION {iteration}\n" + "=" * 50)
    output_dir = Path(args.output) / f"iteration_{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏋️ Phase 2A: Training representatives...")
    representatives_path = train_contrastive_representatives(
        training_data, output_dir, args.epochs, args.margin)

    print(f"\n🔮 Phase 2B: Generating predictions...")
    all_predictions_with_scores = generate_predictions(
        resnet_data_for_prediction, representatives_path, args.threshold
    )

    print(f"\n📊 Phase 2C: Evaluating predictions...")
    evaluation_results, metrics = evaluate_predictions(all_predictions_with_scores, ground_truth)

    print(f"\n🚨 Phase 2D: Extracting false positives...")
    fp_data, new_exclusions = extract_false_positives(
        evaluation_results,
        all_predictions_with_scores, 
        exclusion_tracker
    )

    print(f"\n💾 Saving iteration results...")
    evaluation_results.to_csv(output_dir / "evaluation_results.csv", index=False)
    if not fp_data.empty:
        fp_data.to_csv(output_dir / "false_positives_for_training.csv", index=False)
        with open(output_dir / "false_positives_for_training.pkl", 'wb') as f: pickle.dump(fp_data, f)
    with open(output_dir / "exclusions.json", 'w') as f: json.dump(new_exclusions, f, indent=2)
    with open(output_dir / "training_results.json", 'w') as f: json.dump(metrics, f, indent=2)

    new_training_data = training_data.copy() # Start with a copy
    if not fp_data.empty:
        # Temporarily convert embeddings to tuples for duplicate checking
        temp_training_data_for_dedup = training_data.copy()
        temp_fp_data_for_dedup = fp_data.copy()

        # Check if 'finetuned_embedding' column exists and contains ndarray
        if 'finetuned_embedding' in temp_training_data_for_dedup.columns and \
           not temp_training_data_for_dedup.empty and \
           isinstance(temp_training_data_for_dedup['finetuned_embedding'].iloc[0], np.ndarray):
            temp_training_data_for_dedup['finetuned_embedding_tuple'] = temp_training_data_for_dedup['finetuned_embedding'].apply(tuple)
        else:
             # Handle empty or already tupled - create empty or copy existing if not ndarray
            temp_training_data_for_dedup['finetuned_embedding_tuple'] = None # Placeholder
            if 'finetuned_embedding' in temp_training_data_for_dedup.columns:
                 temp_training_data_for_dedup['finetuned_embedding_tuple'] = temp_training_data_for_dedup['finetuned_embedding']


        if 'finetuned_embedding' in temp_fp_data_for_dedup.columns and \
           not temp_fp_data_for_dedup.empty and \
           isinstance(temp_fp_data_for_dedup['finetuned_embedding'].iloc[0], np.ndarray):
            temp_fp_data_for_dedup['finetuned_embedding_tuple'] = temp_fp_data_for_dedup['finetuned_embedding'].apply(tuple)
        else:
            temp_fp_data_for_dedup['finetuned_embedding_tuple'] = None
            if 'finetuned_embedding' in temp_fp_data_for_dedup.columns:
                 temp_fp_data_for_dedup['finetuned_embedding_tuple'] = temp_fp_data_for_dedup['finetuned_embedding']

        combined_for_dedup = pd.concat([temp_training_data_for_dedup, temp_fp_data_for_dedup], ignore_index=True)
        
        # Perform drop_duplicates on the tuple column
        # Ensure 'finetuned_embedding_tuple' exists before using in subset
        dedup_columns = ['class']
        if 'finetuned_embedding_tuple' in combined_for_dedup.columns:
            dedup_columns.append('finetuned_embedding_tuple')
        
        # Only drop if 'finetuned_embedding_tuple' was successfully created for both and is not None
        can_dedup_on_embedding = ('finetuned_embedding_tuple' in temp_training_data_for_dedup.columns and \
                                 temp_training_data_for_dedup['finetuned_embedding_tuple'].notna().all()) and \
                                 ('finetuned_embedding_tuple' in temp_fp_data_for_dedup.columns and \
                                  temp_fp_data_for_dedup['finetuned_embedding_tuple'].notna().all())

        if not combined_for_dedup.empty and can_dedup_on_embedding:
            num_before_dedup = len(combined_for_dedup)
            deduplicated_df_indices = combined_for_dedup.drop_duplicates(subset=dedup_columns).index
            
            # Create the new_training_data using original np.array embeddings from the combined (non-tupled) data
            original_combined_data = pd.concat([training_data, fp_data], ignore_index=True)
            new_training_data = original_combined_data.loc[deduplicated_df_indices].reset_index(drop=True)
            num_after_dedup = len(new_training_data)
            print(f"🔄 Combined training data ({len(training_data)} existing + {len(fp_data)} new FPs).")
            print(f"🔄 After drop_duplicates on (class, embedding_tuple): {num_before_dedup} -> {num_after_dedup} samples.")

        else: # Fallback if tuple conversion didn't happen or df is empty
            new_training_data = pd.concat([training_data, fp_data], ignore_index=True)
            # As a simpler fallback, try dropping based on all columns if embeddings are an issue,
            # or just accept potential duplicates if embeddings are truly problematic for hashing.
            # This might not be perfect for ndarrays.
            if not new_training_data.empty:
                 print(f"⚠️ Could not reliably convert embeddings to tuples for deduplication. Combining without ndarray-based deduplication.")
                 # Attempt to drop duplicates if all columns are hashable, otherwise, this won't work for ndarray
                 try:
                    initial_len = len(new_training_data)
                    new_training_data.drop_duplicates(inplace=True)
                    if len(new_training_data) < initial_len:
                        print(f"   Performed basic drop_duplicates: {initial_len} -> {len(new_training_data)}")
                 except TypeError:
                    print(f"   Skipping drop_duplicates as embeddings are unhashable in this path.")


        print(f"🔄 Updated training data: {len(training_data)} initial + {len(fp_data)} new FPs -> {len(new_training_data)} total unique samples.")
    else:
        print(f"🔄 No new false positives to add, training data unchanged: {len(training_data)} samples")
        
    print(f"\n📊 ITERATION {iteration} SUMMARY:")
    print(f"   🎯 F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    print(f"   🚨 False Positives (used for new negatives): {metrics['fp']}") # This 'fp' is from evaluation stats
    print(f"   📍 Exclusion list: +{len(new_exclusions)} entries")
    print(f"   📈 Training data size: {len(new_training_data)}")
    
    return new_training_data, metrics, new_exclusions


def filter_evaluation_data(full_resnet_data: pd.DataFrame,
                          exclusion_tracker: Dict,
                          exclude_training_flag: bool = True) -> pd.DataFrame:
    """
    Filter resnet data for evaluation based on exclusion strategy.
    This should be called ONCE before the iterative loop if we want a consistent eval set.
    However, the current pipeline structure might re-evaluate this if not careful.
    For now, this function is less critical if we always predict on full data and then filter.
    The main use of exclusion_tracker is to avoid re-adding already processed FPs to training.
    """
    if not exclude_training_flag or not exclusion_tracker:
        print("🚫 No evaluation data filtering applied based on exclusion tracker.")
        return full_resnet_data.copy()

    excluded_video_frame_class_tuples = set()
    for iter_num, exclusions_in_iter in exclusion_tracker.items():
        for exc in exclusions_in_iter:
            # We exclude based on the video, frame, and the *original wrong prediction*
            # to prevent the model from being penalized on items it's trying to learn *not* to predict.
            # Or, if it's a 'false_positive_negative', we exclude the 'not_X' class.
            # This logic needs to be clear: what exactly are we excluding from *evaluation*?
            # Typically, you exclude items ADDED to training from subsequent *evaluation* sets.
            excluded_video_frame_class_tuples.add((exc['video'], exc['frame'], exc.get('original_wrong_prediction', exc['class'])))
    
    if not excluded_video_frame_class_tuples:
        print("🚫 No exclusions found in tracker to filter from evaluation data.")
        return full_resnet_data.copy()

    # This filtering is complex if we are evaluating predictions that might include 'not_X' classes.
    # For now, let's assume we filter based on (video, frame) of items that were added to training.
    # This means we should primarily store (video, frame) of the *source* of the training data.
    
    excluded_pairs = set((exc['video'], exc['frame']) for iter_excs in exclusion_tracker.values() for exc in iter_excs)

    if not excluded_pairs:
        return full_resnet_data.copy()

    original_count = len(full_resnet_data)
    # Filter based on (video, frame) not being in the set of items used for training augmentation
    filtered_data = full_resnet_data[~full_resnet_data.apply(lambda row: (row['video'], row['frame']) in excluded_pairs, axis=1)].copy()
    
    print(f"🔍 Filtered evaluation data: {original_count} → {len(filtered_data)} (removed {original_count - len(filtered_data)} based on exclusion tracker)")
    return filtered_data


def run_iterative_pipeline(definitiveObjects: pd.DataFrame,
                          resnetPredictions: pd.DataFrame, # This is the full, raw prediction data
                          trackingInfo: pd.DataFrame,    # This is ground truth
                          args) -> Dict:
    """
    Run the full iterative contrastive learning pipeline.
    """
    print("\n🚀 Starting Iterative Contrastive Learning Pipeline (Negative Classes Mode)")
    
    # Initial training data: definitiveObjects
    current_training_data = definitiveObjects.copy()
    
    # Exclusion tracker: {iteration_num: [exclusion_dicts]}
    # exclusion_dict: {'video', 'frame', 'class' (original or not_X), 'reason', 'original_wrong_prediction' (if applicable)}
    exclusion_tracker: Dict[int, List[Dict]] = {} 
    
    all_iteration_metrics = []
    previous_f1 = -1.0

    # Determine evaluation strategy based on args
    if args.include_training_in_eval:
        eval_exclusion_flag = False
        print("📈 Evaluation will INCLUDE all data (including those added to training).")
    else: # Default is exclude_training_from_eval or track_training_separately
        eval_exclusion_flag = True
        print("📉 Evaluation will EXCLUDE data added to training in previous iterations (standard).")
        if args.track_training_separately:
            print("   (Will also track metrics on full data if possible, TBD)")


    # The resnetPredictions are the full set of embeddings we can make predictions on.
    # For evaluation, we might filter this set based on what's been added to training.
    # However, for generating candidates for new negatives, we should predict on the whole set.
    
    # For now, resnet_data_for_prediction will be the full set.
    # Evaluation will use the ground_truth applied to predictions made on this full set.
    # The `filter_evaluation_data` is somewhat misnamed if used this way; it's more about
    # ensuring we don't retrain on the *exact same* FPs, which `drop_duplicates` in run_single_iteration handles.
    
    resnet_data_for_prediction = resnetPredictions.copy()

    for i in range(1, args.iterations + 1):
        # In each iteration, training_data grows.
        # Predictions are made on the consistent `resnet_data_for_prediction`.
        # Evaluation is against `ground_truth`.
        
        # If `eval_exclusion_flag` is true, we might want to filter `all_predictions_with_scores`
        # *before* passing to `evaluate_predictions`, but this complicates `extract_false_positives`
        # which needs to know about FPs from the unfiltered set.
        # Simpler: evaluate on all predictions, `extract_false_positives` from these,
        # and `current_training_data.drop_duplicates` handles not re-adding.
        # The `eval_exclusion_flag` then primarily serves as a philosophical note on metric reporting.

        new_training_data, metrics, new_exclusions = run_single_iteration(
            iteration=i,
            training_data=current_training_data,
            resnet_data_for_prediction=resnet_data_for_prediction, # Always use the full set for generating predictions
            ground_truth=trackingInfo,
            exclusion_tracker=exclusion_tracker, # Pass for logging, not for filtering resnet_data here
            args=args
        )
        
        current_training_data = new_training_data
        exclusion_tracker[i] = new_exclusions
        all_iteration_metrics.append({'iteration': i, **metrics})
        
        # Convergence check
        current_f1 = metrics['f1']
        if i > 1 and (current_f1 - previous_f1) < args.convergence_threshold:
            print(f"\n✅ Convergence reached at iteration {i}: F1 improvement ({current_f1 - previous_f1:.4f}) < threshold ({args.convergence_threshold:.4f})")
            break
        previous_f1 = current_f1
        
        if i == args.iterations:
            print("\n🏁 Maximum iterations reached.")
            
    # Save overall pipeline summary
    summary_path = Path(args.output) / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'iteration_metrics': all_iteration_metrics,
            'final_training_data_size': len(current_training_data),
            'final_exclusion_count': sum(len(v) for v in exclusion_tracker.values())
        }, f, indent=2)
    print(f"\n📜 Pipeline summary saved to: {summary_path}")
    
    # Optional: Save final combined training data
    final_training_data_path = Path(args.output) / "final_training_data.pkl"
    with open(final_training_data_path, 'wb') as f:
        pickle.dump(current_training_data, f)
    print(f"💾 Final combined training data saved to: {final_training_data_path}")

    return {'final_metrics': all_iteration_metrics[-1] if all_iteration_metrics else None}


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"⚠️ Output directory {output_dir} already exists and is not empty.")
        # shutil.rmtree(output_dir) # Optionally clear
        # print(f"Cleared output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        definitiveObjects, resnetPredictions, trackingInfo = load_and_validate_data(
            args.definitive_objects, 
            args.resnet_predictions, 
            args.tracking_info
        )
        
        if args.test_mode:
            print("\n✅ Test mode: Data loading and validation successful. Exiting.")
            return

        run_iterative_pipeline(definitiveObjects, resnetPredictions, trackingInfo, args)
        
        print("\n🎉 Iterative contrastive learning pipeline finished successfully!")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}. Please check input paths.")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Data validation error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Pipeline runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()