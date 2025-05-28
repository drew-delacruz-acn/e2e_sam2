#!/usr/bin/env python3
"""
Iterative Contrastive Learning Pipeline with Negative Classes

This pipeline implements an iterative approach to improve class representatives by:
1. Training contrastive learning models on current data
2. Evaluating against predictions using cosine similarity
3. Extracting false positives and adding them as NEGATIVE examples ("not_class")
4. Using contrastive scoring during prediction (positive_sim - negative_sim) OR
   using refined positive representatives directly.
5. Iterating until convergence

Usage:
    python iterative_contrastive_pipeline_negative.py \
        --definitive-objects path/to/definitiveObjects.pkl \
        --resnet-predictions path/to/resnetPredictions.pkl \
        --tracking-info path/to/trackingInfo.pkl \
        --iterations 5 \
        --threshold 0.6 \
        --secondary-threshold 0.4 \
        --margin 0.2 \
        --secondary-margin 0.3
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
import traceback

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
                      help='Confidence threshold for predictions, used in the first iteration (default: 0.6)')
    parser.add_argument('--secondary-threshold', type=float, default=None,
                      help='Confidence threshold for predictions for iterations AFTER the first one '
                           '(i.e., after false positives have been added). '
                           'If not set, the primary --threshold is used for all iterations (default: None).')
    parser.add_argument('--convergence-threshold', type=float, default=0.001,
                      help='F1 improvement threshold for convergence (default: 0.001)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of contrastive learning epochs (default: 50)')
    parser.add_argument('--margin', type=float, default=0.2,
                      help='Contrastive learning margin for the first iteration (default: 0.2)')
    parser.add_argument('--secondary-margin', type=float, default=None,
                      help='Contrastive learning margin for iterations AFTER the first one. '
                           'If not set, the primary --margin is used for all iterations.')
    
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
        traceback.print_exc()
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
        traceback.print_exc()
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
        traceback.print_exc()
        raise
    
    print("✅ Data loading and validation complete!")
    return definitiveObjects, resnetPredictions, trackingInfo


def train_contrastive_representatives(training_data: pd.DataFrame, 
                                   output_dir: Path,
                                   epochs: int, 
                                   margin_to_use: float # Accept specific margin
                                   ) -> Path:
    """
    Train contrastive learning representatives using existing train_representatives.py.
    """
    print(f"🏋️ Training contrastive representatives (Epochs: {epochs}, Margin: {margin_to_use})...")
    temp_data_path = output_dir / "temp_training_data.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(temp_data_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    # Try to find train_representatives.py in a few common locations relative to this script
    possible_train_script_paths = [
        Path(__file__).parent / "train_representatives.py", # Same directory
        Path(__file__).parent.parent / "train_representatives.py", # One level up (e.g. if this is in a 'pipelines' subdir)
        Path(__file__).parent.parent.parent / "train_representatives.py" # Two levels up (project root if this is nested)
    ]
    train_script_path = None
    for p in possible_train_script_paths:
        if p.exists():
            train_script_path = p
            break
    if train_script_path is None:
        # Fallback to just the name, relying on PATH or current dir if script moved
        train_script_path_fallback = Path("train_representatives.py")
        if train_script_path_fallback.exists():
            train_script_path = train_script_path_fallback
        else:
            raise FileNotFoundError(f"train_representatives.py not found at expected locations: {possible_train_script_paths} or as 'train_representatives.py'")

    cmd = [
        "python", str(train_script_path),
        "--data", str(temp_data_path),
        "--output", str(output_dir),
        "--epochs", str(epochs),
        "--margin", str(margin_to_use),
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
    if not class_embeddings or not class_names: return "unknown", 0.0 # Handle empty inputs
    input_tensor = F.normalize(torch.tensor(embedding, dtype=torch.float32).unsqueeze(0), dim=1)
    class_tensor = F.normalize(torch.tensor(np.vstack(class_embeddings), dtype=torch.float32), dim=1)
    similarities = F.cosine_similarity(input_tensor, class_tensor).numpy().flatten()
    best_idx = np.argmax(similarities)
    return class_names[best_idx], similarities[best_idx]


def generate_predictions(resnet_data: pd.DataFrame,
                        representatives_path: Path,
                        threshold: float,
                        iteration_number: int) -> pd.DataFrame:
    """
    Generate predictions using ONLY POSITIVE CLASS representatives.
    "not_" class representatives (if they exist in the pkl file) are ignored for prediction,
    having served their purpose during the training of positive class representatives.
    """
    print(f"🔮 Generating predictions (Iteration {iteration_number}) with threshold {threshold} using ONLY POSITIVE representatives...")
    with open(representatives_path, 'rb') as f:
        representatives_data = pickle.load(f)
    if isinstance(representatives_data, dict):
        try: temp_df = pd.DataFrame(representatives_data)
        except: raise ValueError("Cannot convert dict representatives to DataFrame")
        if 'finetuned_embedding' in temp_df.columns: reps_df = temp_df
        elif 'representative_embedding' in temp_df.columns: reps_df = temp_df.rename(columns={'representative_embedding':'finetuned_embedding'})
        else: raise ValueError("Dict representatives missing embedding column")
    elif isinstance(representatives_data, pd.DataFrame):
        if 'representative_embedding' in representatives_data.columns and 'finetuned_embedding' not in representatives_data.columns:
            reps_df = representatives_data.rename(columns={'representative_embedding':'finetuned_embedding'})
        else: reps_df = representatives_data
    else: raise ValueError(f"Unknown representatives format: {type(representatives_data)}")
    if not all(col in reps_df.columns for col in ['class', 'finetuned_embedding']):
        raise ValueError("Representatives DataFrame missing 'class' or 'finetuned_embedding'")
    positive_representatives_df = reps_df[~reps_df['class'].str.startswith('not_', na=False)].copy()
    if positive_representatives_df.empty:
        print("⚠️ No positive class representatives found after filtering! Cannot make predictions.")
        empty_pred_cols = list(resnet_data.columns) + ['visual_predicted_object', 'visual_max_score']
        return pd.DataFrame(columns=empty_pred_cols)
    class_embeddings = list(positive_representatives_df['finetuned_embedding'])
    class_names = list(positive_representatives_df['class'])
    print(f"🏷️  Using {len(class_names)} POSITIVE representatives for prediction: {class_names}")
    
    predictions_list = []
    for _, row in resnet_data.iterrows():
        pred_class, confidence = cosine_similarity_prediction( 
            row['finetuned_embedding'], 
            class_embeddings, 
            class_names
        )
        predictions_list.append({
            'visual_predicted_object': pred_class, 
            'visual_max_score': confidence
        })
    
    pred_df = pd.DataFrame(predictions_list)
    
    result_df = pd.concat([resnet_data.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    print(f"✅ Generated {len(result_df)} initial predictions using positive reps.")

    if not result_df.empty and 'visual_predicted_object' in result_df.columns:
        idx = result_df.groupby(['video', 'visual_predicted_object'])['visual_max_score'].idxmax()
        deduplicated_df = result_df.loc[idx].reset_index(drop=True)
        print(f"✅ Deduplicated to {len(deduplicated_df)} unique video-class predictions.")
    else:
        deduplicated_df = pd.DataFrame(columns=result_df.columns if not result_df.empty else list(resnet_data.columns) + ['visual_predicted_object', 'visual_max_score'])
        print("⚠️ No predictions made or 'visual_predicted_object' missing, possibly due to no positive representatives or empty resnet_data.")


    final_df = deduplicated_df[deduplicated_df['visual_max_score'] >= threshold].copy()
    print(f"✅ {len(final_df)} predictions at/above threshold {threshold}.")
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
                           predictions_source: pd.DataFrame, 
                           exclusion_tracker: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Extract false positive cases and prepare them as negative examples for training.
    """
    print("🚨 Extracting false positives as negative examples...")

    print(f"DEBUG (extract_false_positives): predictions_source index is_unique: {predictions_source.index.is_unique}")
    if not predictions_source.empty:
        print(f"DEBUG (extract_false_positives): predictions_source head:\\n{predictions_source.head()}")
        if not predictions_source.index.is_unique:
            print(f"DEBUG (extract_false_positives): Duplicated indices in predictions_source:\\n{predictions_source.index[predictions_source.index.duplicated()].unique()}")
            # --- POTENTIAL QUICK FIX ---
            print(f"WARNING (extract_false_positives): predictions_source came with non-unique index. Resetting index.")
            predictions_source = predictions_source.reset_index(drop=True)
            print(f"DEBUG (extract_false_positives): predictions_source index is_unique (after reset): {predictions_source.index.is_unique}")
            # --- END POTENTIAL QUICK FIX ---
    else:
        print("DEBUG (extract_false_positives): predictions_source is empty.")

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
        original_pred_entry = predictions_source[
            (predictions_source['video'] == video) &
            (predictions_source['visual_predicted_object'] == wrongly_predicted_as_class) &
            (predictions_source['frame'] == frame_of_fp) # Ensure we get the exact FP frame
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
                        resnet_data_for_prediction: pd.DataFrame,
                        ground_truth: pd.DataFrame,
                        exclusion_tracker: Dict, # For logging, not direct filtering here
                        current_iter_threshold: float, 
                        current_iter_margin: float, # Added for secondary margin
                        args) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    """Run a single iteration of the pipeline."""
    print(f"\n🔄 ITERATION {iteration} (Using Threshold: {current_iter_threshold}, Margin: {current_iter_margin})\n" + "=" * 50)
    output_dir = Path(args.output) / f"iteration_{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏋️ Phase 2A: Training representatives...")
    representatives_path = train_contrastive_representatives(training_data, output_dir, args.epochs, current_iter_margin)

    print(f"\n🔮 Phase 2B: Generating predictions...")
    # This is the result *after* thresholding and deduplication of positive classes by generate_predictions
    predictions_for_eval = generate_predictions(resnet_data_for_prediction, representatives_path, current_iter_threshold, iteration_number=iteration)
    
    print(f"DEBUG (run_single_iteration): predictions_for_eval index is_unique: {predictions_for_eval.index.is_unique}")
    if not predictions_for_eval.empty:
        print(f"DEBUG (run_single_iteration): predictions_for_eval head:\\n{predictions_for_eval.head()}")
        if not predictions_for_eval.index.is_unique:
            print(f"DEBUG (run_single_iteration): Duplicated indices in predictions_for_eval:\\n{predictions_for_eval.index[predictions_for_eval.index.duplicated()].unique()}")
    else:
        print("DEBUG (run_single_iteration): predictions_for_eval is empty.")


    print(f"\n📊 Phase 2C: Evaluating predictions...")
    evaluation_results, metrics = evaluate_predictions(predictions_for_eval, ground_truth)

    print(f"\n🚨 Phase 2D: Extracting false positives...")
    # Pass predictions_for_eval as the basis for identifying FP embeddings
    fp_data, new_exclusions = extract_false_positives(
        evaluation_results, 
        predictions_for_eval, # This is predictions_source in the next function
        exclusion_tracker
    )

    print(f"\n💾 Saving iteration results...")
    evaluation_results.to_csv(output_dir / "evaluation_results.csv", index=False)
    if not fp_data.empty:
        fp_data.to_csv(output_dir / "false_positives_for_training.csv", index=False)
        with open(output_dir / "false_positives_for_training.pkl", 'wb') as f: pickle.dump(fp_data, f)
    if new_exclusions: 
        with open(output_dir / "exclusions.json", 'w') as f: json.dump(new_exclusions, f, indent=2)
    with open(output_dir / "training_results.json", 'w') as f: json.dump(metrics, f, indent=2)

    current_training_data_len = len(training_data)
    new_training_data_iter = training_data.copy() # Initialize with current training data

    if not fp_data.empty:
        print(f"DEBUG: training_data index is_unique: {training_data.index.is_unique}")
        print(f"DEBUG: training_data.shape: {training_data.shape}")
        if not training_data.empty: print(f"DEBUG: training_data head:\n{training_data.head()}")
        
        print(f"DEBUG: fp_data index is_unique: {fp_data.index.is_unique}")
        print(f"DEBUG: fp_data.shape: {fp_data.shape}")
        if not fp_data.empty: print(f"DEBUG: fp_data head:\n{fp_data.head()}")

        temp_td = training_data.copy()
        temp_fp = fp_data.copy()
        
        # Prepare 'emb_tuple' column for temp_td
        if 'finetuned_embedding' in temp_td.columns and not temp_td.empty and isinstance(temp_td['finetuned_embedding'].iloc[0], np.ndarray):
            temp_td['emb_tuple'] = temp_td['finetuned_embedding'].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
        elif 'finetuned_embedding' in temp_td.columns: # Column exists but might be empty or not ndarray
             temp_td['emb_tuple'] = temp_td['finetuned_embedding'] # Copy as is, might not be hashable for all
        else: # Column doesn't exist
            temp_td['emb_tuple'] = pd.Series(dtype='object', index=temp_td.index)


        # Prepare 'emb_tuple' column for temp_fp
        if 'finetuned_embedding' in temp_fp.columns and not temp_fp.empty and isinstance(temp_fp['finetuned_embedding'].iloc[0], np.ndarray):
            temp_fp['emb_tuple'] = temp_fp['finetuned_embedding'].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
        elif 'finetuned_embedding' in temp_fp.columns:
            temp_fp['emb_tuple'] = temp_fp['finetuned_embedding']
        else:
            temp_fp['emb_tuple'] = pd.Series(dtype='object', index=temp_fp.index)

        dedup_cols = ['class']
        # Check if 'emb_tuple' can be reliably used for deduplication
        can_use_emb_tuple = (
            'emb_tuple' in temp_td.columns and temp_td['emb_tuple'].notna().any() and
            'emb_tuple' in temp_fp.columns and temp_fp['emb_tuple'].notna().any() and
            all(isinstance(x, tuple) for x in temp_td['emb_tuple'].dropna()) and # Ensure they are actually tuples
            all(isinstance(x, tuple) for x in temp_fp['emb_tuple'].dropna())
        )

        if can_use_emb_tuple:
            print("DEBUG: Using primary deduplication path with 'emb_tuple'.")
            dedup_cols.append('emb_tuple')
            
            # Ensure all necessary columns for concat exist in both DataFrames
            cols_for_concat_temp = ['class', 'finetuned_embedding', 'emb_tuple']
            
            # Create DataFrames with only the necessary columns, handling missing ones gracefully
            df_list_for_concat = []
            for df_orig, name in [(temp_td, "temp_td"), (temp_fp, "temp_fp")]:
                cols_present = [col for col in cols_for_concat_temp if col in df_orig.columns]
                if not df_orig.empty and cols_present:
                    df_list_for_concat.append(df_orig[cols_present])
                else: # Create an empty DataFrame with expected columns if original is empty or lacks key cols
                    print(f"DEBUG: {name} is empty or lacks essential columns for primary dedup path concat.")
                    # df_list_for_concat.append(pd.DataFrame(columns=cols_for_concat_temp)) # This might lead to issues if dtypes differ later
            
            if len(df_list_for_concat) == 2: # Both DFs were prepared
                temp_combined_for_dedup = pd.concat(df_list_for_concat, ignore_index=True)
                
                num_before_dedup = len(temp_combined_for_dedup)
                # Drop duplicates based on the 'emb_tuple' and 'class'
                deduplicated_temp = temp_combined_for_dedup.drop_duplicates(subset=dedup_cols, keep='first')
                # Select original columns (without 'emb_tuple') and reset index
                new_training_data_iter = deduplicated_temp[['class', 'finetuned_embedding']].reset_index(drop=True)
                num_after_dedup = len(new_training_data_iter)
                print(f"🔄 Combined training data. Before dedup: {num_before_dedup}, After dedup on (class, embedding_tuple): {num_after_dedup} samples.")
            else:
                print("⚠️ Could not prepare both DataFrames for primary deduplication path. Moving to fallback.")
                can_use_emb_tuple = False # Force fallback

        if not can_use_emb_tuple: 
            print("⚠️ Fallback deduplication: Tuple conversion for embeddings failed or one of the DFs was incompatible.")
            # This concat gets a clean 0..N-1 index due to ignore_index=True
            new_training_data_fallback = pd.concat([training_data, fp_data], ignore_index=True) 
            
            print(f"DEBUG: Fallback new_training_data_fallback (after concat, before astype dedup) index is_unique: {new_training_data_fallback.index.is_unique}")
            if not new_training_data_fallback.empty: print(f"DEBUG: Fallback new_training_data_fallback head:\n{new_training_data_fallback.head()}")
            
            try:
                initial_len = len(new_training_data_fallback)
                if not new_training_data_fallback.empty:
                    # This is a very broad attempt; may not work well if 'finetuned_embedding' is the only differentiator
                    # and it's an ndarray.
                    # Create a temporary column that's a string representation of all other columns for a row
                    # This is computationally intensive and approximate.
                    
                    # A simpler fallback: drop based on 'class' only, if 'finetuned_embedding' is too problematic
                    # This is not ideal as it might drop legitimate different embeddings for the same class.
                    if 'class' in new_training_data_fallback.columns:
                        print("DEBUG: Fallback attempting drop_duplicates on 'class' only.")
                        new_training_data_iter = new_training_data_fallback.drop_duplicates(subset=['class'], keep='first').reset_index(drop=True)
                    else: # If no 'class' column, just take the concat
                        new_training_data_iter = new_training_data_fallback
                else:
                    new_training_data_iter = new_training_data_fallback # Should be empty

                if len(new_training_data_iter) < initial_len:
                    print(f"   Performed basic drop_duplicates (on 'class' or was empty): {initial_len} -> {len(new_training_data_iter)}")
                else:
                    print(f"   Basic drop_duplicates did not reduce size or was skipped.")

            except Exception as e_basic_dedup:
                 print(f"   Basic drop_duplicates failed: {e_basic_dedup}. Proceeding with data as is from concat (potential duplicates).")
                 new_training_data_iter = new_training_data_fallback # Use the concatenated version
                 print(f"DEBUG: new_training_data state after basic_dedup exception")
                 if not new_training_data_iter.empty: print(f"DEBUG: Index: {new_training_data_iter.index.is_unique}, Head:\n{new_training_data_iter.head()}")


        added_count = len(new_training_data_iter) - current_training_data_len
        print(f"🔄 Updated training data: {current_training_data_len} init + {len(fp_data)} new FPs -> {len(new_training_data_iter)} total ({added_count} unique added).")
    # else: # fp_data is empty, new_training_data remains training_data.copy()
        # print(f"🔄 No new FPs to add. Training data size: {len(new_training_data)}.") # Covered by initialization
        
    print(f"\n📊 ITERATION {iteration} SUMMARY: F1:{metrics['f1']:.4f}, P:{metrics['precision']:.4f}, R:{metrics['recall']:.4f}. Added FPs to train: {len(fp_data)}. New exclusions: {len(new_exclusions)}. Training size: {len(new_training_data_iter)}")
    return new_training_data_iter, metrics, new_exclusions


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
                          resnetPredictions: pd.DataFrame, 
                          trackingInfo: pd.DataFrame,    
                          args) -> Dict:
    """
    Run the full iterative contrastive learning pipeline.
    """
    print("\n🚀 Starting Iterative Contrastive Learning Pipeline (Negative Classes Mode)")
    current_training_data = definitiveObjects.copy()
    exclusion_tracker: Dict[int, List[Dict]] = {} 
    all_iteration_metrics = []
    previous_f1 = -1.0

    if args.include_training_in_eval:
        print("📈 Evaluation will INCLUDE all data (including those added to training).")
    else: 
        print("📉 Evaluation will EXCLUDE data added to training in previous iterations (standard behavior implied by not re-adding duplicates).")

    resnet_data_for_prediction = resnetPredictions.copy()

    for i in range(1, args.iterations + 1):
        current_iter_threshold_to_use = args.threshold 
        current_iter_margin_to_use = args.margin 

        if i > 1: 
            if args.secondary_threshold is not None: current_iter_threshold_to_use = args.secondary_threshold
            if args.secondary_margin is not None: current_iter_margin_to_use = args.secondary_margin
        
        if i == 1: print(f"💡 Iteration {i}: Using Primary Threshold: {current_iter_threshold_to_use}, Primary Margin: {current_iter_margin_to_use}")
        else: print(f"💡 Iteration {i}: Using Threshold: {current_iter_threshold_to_use} (Secondary if set, else Primary), Margin: {current_iter_margin_to_use} (Secondary if set, else Primary)")

        new_training_data, metrics, new_exclusions = run_single_iteration(
            iteration=i,
            training_data=current_training_data,
            resnet_data_for_prediction=resnet_data_for_prediction,
            ground_truth=trackingInfo,
            exclusion_tracker=exclusion_tracker, 
            current_iter_threshold=current_iter_threshold_to_use,
            current_iter_margin=current_iter_margin_to_use, 
            args=args
        )
        
        current_training_data = new_training_data
        if new_exclusions: 
            exclusion_tracker[i] = new_exclusions
        all_iteration_metrics.append({'iteration': i, **metrics})
        
        current_f1 = metrics['f1']
        if i > 1 and previous_f1 >= 0 and abs(current_f1 - previous_f1) < args.convergence_threshold: # abs for safety
            print(f"\n✅ Convergence at iter {i}: F1 imprv ({current_f1 - previous_f1:.4f}) < thresh ({args.convergence_threshold:.4f})")
            break
        previous_f1 = current_f1
        
        if i == args.iterations:
            print("\n🏁 Maximum iterations reached.")
            
    # Save overall pipeline summary
    summary = {'config': vars(args), 'iteration_metrics': all_iteration_metrics,
               'final_training_data_size': len(current_training_data),
               'final_exclusion_count': sum(len(v) for v in exclusion_tracker.values())}
    summary_path = Path(args.output) / "pipeline_summary.json"
    with open(summary_path, 'w') as f: json.dump(summary, f, indent=2)
    print(f"\n📜 Pipeline summary saved to {summary_path}. Final F1: {all_iteration_metrics[-1]['f1']:.4f if all_iteration_metrics else 'N/A'}.")
    
    # Optional: Save final combined training data
    final_training_data_path = Path(args.output) / "final_training_data.pkl"
    with open(final_training_data_path, 'wb') as f:
        pickle.dump(current_training_data, f)
    print(f"💾 Final training data saved to {final_training_data_path}.")

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
        print("--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Data validation error: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Pipeline runtime error: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")
        sys.exit(1)
    except Exception as e: # Generic catch-all
        print(f"❌ An unexpected error occurred: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")
        sys.exit(1)

if __name__ == "__main__":
    main()