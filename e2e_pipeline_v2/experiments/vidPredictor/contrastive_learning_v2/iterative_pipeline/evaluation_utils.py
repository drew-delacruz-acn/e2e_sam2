import pandas as pd
import numpy as np # For type hinting if embeddings are expected
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score

# Column name constants (can be shared or defined per module)
COL_VIDEO = 'video'
COL_CLASS = 'class' # Often used for the 'true' or 'predicted' class label
COL_TAG = 'tag' # Often used for ground truth class label
COL_ACTUAL = 'actual'
COL_PREDICTED = 'predicted'
COL_VISUAL_PRED_OBJECT = 'visual_predicted_object'
COL_VISUAL_MAX_SCORE = 'visual_max_score'
COL_FRAME = 'frame'
COL_CLASSIFICATION = 'classification'
COL_EMBEDDING = 'finetuned_embedding'

# Import debug logging function
try:
    from .pipeline_manager import debug_log
except ImportError:
    # Fallback if circular import
    def debug_log(message: str):
        print(message)

def evaluate_predictions(predictions: pd.DataFrame,
                        ground_truth: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate predictions against ground truth for presence/absence at FRAME LEVEL.
    """
    print("📊 Evaluating predictions against ground truth (FRAME-LEVEL)...")

    if predictions.empty:
        print("⚠️ Predictions DataFrame is empty. Cannot evaluate. Returning zero metrics and empty eval_df.")
        metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        if not ground_truth.empty:
            metrics['FN'] = int(ground_truth[ground_truth[COL_ACTUAL] == 1][COL_ACTUAL].count())
            metrics['TN'] = int(ground_truth[ground_truth[COL_ACTUAL] == 0][COL_ACTUAL].count())
        return pd.DataFrame(columns=[COL_VIDEO, COL_CLASS, COL_ACTUAL, COL_PREDICTED, 'confidence', COL_FRAME, COL_CLASSIFICATION]), metrics

    eval_results = []
    gt_eval = ground_truth.copy()
    if COL_TAG not in gt_eval.columns and COL_CLASS in gt_eval.columns: 
        gt_eval = gt_eval.rename(columns={COL_CLASS: COL_TAG})
    if not pd.api.types.is_numeric_dtype(gt_eval[COL_ACTUAL]):
            gt_eval[COL_ACTUAL] = gt_eval[COL_ACTUAL].astype(int)

    debug_log(f"📊 FRAME-LEVEL EVALUATION DEBUG:")
    debug_log(f"   Ground truth rows: {len(gt_eval)}")
    debug_log(f"   Predictions rows: {len(predictions)}")
    debug_log(f"   Ground truth columns: {list(gt_eval.columns)}")
    debug_log(f"   Predictions columns: {list(predictions.columns)}")

    # Check if frame columns exist
    if COL_FRAME not in gt_eval.columns:
        debug_log(f"⚠️ WARNING: No frame column in ground truth. Available: {list(gt_eval.columns)}")
        print(f"⚠️ WARNING: No frame column in ground truth. Falling back to video-class evaluation.")
        # Fall back to original logic if no frame data
        use_frame_matching = False
    elif COL_FRAME not in predictions.columns:
        debug_log(f"⚠️ WARNING: No frame column in predictions. Available: {list(predictions.columns)}")
        print(f"⚠️ WARNING: No frame column in predictions. Falling back to video-class evaluation.")
        use_frame_matching = False
    else:
        use_frame_matching = True
        debug_log(f"✅ Frame columns found in both datasets. Using frame-level matching.")

    frame_matches = 0
    frame_mismatches = 0
    
    for _, gt_row in gt_eval.iterrows():
        video_val, true_class_val, actual_val = gt_row[COL_VIDEO], gt_row[COL_TAG], gt_row[COL_ACTUAL]
        frame_val = gt_row.get(COL_FRAME) if use_frame_matching else None
        
        if use_frame_matching and frame_val is not None:
            # FRAME-LEVEL MATCHING: Match video + frame + class
            # Handle type conversion - convert both to strings for comparison
            frame_str = str(frame_val)
            pred_match = predictions[
                (predictions[COL_VIDEO] == video_val) & 
                (predictions[COL_FRAME].astype(str) == frame_str) &  # Convert to string for comparison
                (predictions[COL_VISUAL_PRED_OBJECT] == true_class_val)
            ]
            if not pred_match.empty:
                frame_matches += 1
            else:
                frame_mismatches += 1
        else:
            # VIDEO-CLASS MATCHING: Original logic (fallback)
            pred_match = predictions[
                (predictions[COL_VIDEO] == video_val) & 
                (predictions[COL_VISUAL_PRED_OBJECT] == true_class_val)
            ]
        
        predicted_val = 0
        confidence_val = 0.0
        frame_val_for_output = frame_val if use_frame_matching else None
        
        if not pred_match.empty:
            predicted_val = 1
            confidence_val = pred_match[COL_VISUAL_MAX_SCORE].iloc[0]
            if COL_FRAME in pred_match.columns and not use_frame_matching: 
                frame_val_for_output = pred_match[COL_FRAME].iloc[0]
        
        classification_val = ''
        if actual_val == 0 and predicted_val == 0: classification_val = 'TN'
        elif actual_val == 1 and predicted_val == 1: classification_val = 'TP'
        elif actual_val == 1 and predicted_val == 0: classification_val = 'FN'
        elif actual_val == 0 and predicted_val == 1: classification_val = 'FP'
        
        eval_results.append({
            COL_VIDEO: video_val,
            COL_CLASS: true_class_val, 
            COL_ACTUAL: actual_val,
            COL_PREDICTED: predicted_val,
            'confidence': confidence_val,
            COL_FRAME: frame_val_for_output,
            COL_CLASSIFICATION: classification_val
        })
    
    eval_df = pd.DataFrame(eval_results)
    
    if use_frame_matching:
        debug_log(f"📊 FRAME-LEVEL MATCHING RESULTS:")
        debug_log(f"   Frame matches found: {frame_matches}")
        debug_log(f"   Frame mismatches: {frame_mismatches}")
        debug_log(f"   Total evaluations: {len(eval_results)}")
        debug_log(f"   Match rate: {frame_matches/len(eval_results)*100:.1f}%")
    
    if not eval_df.empty:
        y_true = eval_df[COL_ACTUAL]
        y_pred = eval_df[COL_PREDICTED]
        tp_count = int(eval_df[eval_df[COL_CLASSIFICATION] == 'TP'].shape[0])
        fp_count = int(eval_df[eval_df[COL_CLASSIFICATION] == 'FP'].shape[0])
        fn_count = int(eval_df[eval_df[COL_CLASSIFICATION] == 'FN'].shape[0])
        tn_count = int(eval_df[eval_df[COL_CLASSIFICATION] == 'TN'].shape[0])
        metrics = {
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'TP': tp_count, 'FP': fp_count, 'FN': fn_count, 'TN': tn_count
        }
    else: 
        metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        if not ground_truth.empty:
            metrics['FN'] = int(ground_truth[ground_truth[COL_ACTUAL] == 1][COL_ACTUAL].count())
            metrics['TN'] = int(ground_truth[ground_truth[COL_ACTUAL] == 0][COL_ACTUAL].count())

    eval_mode = "FRAME-LEVEL" if use_frame_matching else "VIDEO-CLASS"
    print(f"   📊 {eval_mode} EVALUATION: F1: {metrics['f1']:.4f}, P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}. Counts: TP:{metrics['TP']}, FP:{metrics['FP']}, FN:{metrics['FN']}, TN:{metrics['TN']}")
    debug_log(f"📊 {eval_mode} EVALUATION COMPLETE: {len(eval_df)} samples evaluated")
    
    return eval_df, metrics

def extract_false_positives(evaluation_results: pd.DataFrame,
                           predictions_for_eval: pd.DataFrame, 
                           exclusion_tracker: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Extract false positive cases and prepare them as negative examples for training.
    """
    print("🚨 Extracting false positives as negative examples...")
    fp_cases = evaluation_results[evaluation_results[COL_CLASSIFICATION] == 'FP'].copy()
    if fp_cases.empty:
        print("✅ No false positives found!")
        return pd.DataFrame(columns=[COL_CLASS, COL_EMBEDDING]), []
    print(f"🚨 Found {len(fp_cases)} false positive cases (from evaluation_results).")
    fp_data_list = [] 
    exclusion_list = []
    required_pred_cols = [COL_VIDEO, COL_FRAME, COL_VISUAL_PRED_OBJECT, COL_EMBEDDING]
    if not all(col in predictions_for_eval.columns for col in required_pred_cols):
        missing_cols_str = ", ".join(set(required_pred_cols) - set(predictions_for_eval.columns))
        print(f"⚠️ 'predictions_for_eval' DataFrame is missing required columns for FP extraction: {missing_cols_str}. Available: {list(predictions_for_eval.columns)}")
        return pd.DataFrame(columns=[COL_CLASS, COL_EMBEDDING]), []

    for _, fp_row in fp_cases.iterrows():
        video_val = fp_row[COL_VIDEO]
        wrongly_predicted_as_class_val = fp_row[COL_CLASS] 
        frame_of_fp_val = fp_row[COL_FRAME]
        if frame_of_fp_val is None or pd.isna(frame_of_fp_val):
            print(f"⚠️ Skipping FP with no valid frame: Video {video_val}, Class {wrongly_predicted_as_class_val}")
            continue
        
        # Safe frame value conversion with proper validation
        try:
            if isinstance(frame_of_fp_val, (int, float)):
                frame_of_fp_val = int(frame_of_fp_val)
            elif isinstance(frame_of_fp_val, str):
                frame_of_fp_val = int(float(frame_of_fp_val))  # Handle string numbers like "123.0"
            else:
                print(f"⚠️ Skipping FP with invalid frame type: Video {video_val}, Class {wrongly_predicted_as_class_val}, Frame type: {type(frame_of_fp_val)}")
                continue
        except (ValueError, TypeError) as e:
            print(f"⚠️ Skipping FP with unconvertible frame value: Video {video_val}, Class {wrongly_predicted_as_class_val}, Frame: {frame_of_fp_val}, Error: {e}")
            continue
        original_pred_entry_df = predictions_for_eval[
            (predictions_for_eval[COL_VIDEO] == video_val) &
            (predictions_for_eval[COL_VISUAL_PRED_OBJECT] == wrongly_predicted_as_class_val) &
            (predictions_for_eval[COL_FRAME] == frame_of_fp_val) 
        ]
        if original_pred_entry_df.empty:
            print(f"⚠️ Could not find original embedding for FP in 'predictions_for_eval': "
                  f"Video '{video_val}', Frame {frame_of_fp_val}, Predicted Class '{wrongly_predicted_as_class_val}'. "
                  f"Check if 'predictions_for_eval' data is consistent with 'evaluation_results'.")
            continue
        if COL_EMBEDDING not in original_pred_entry_df.columns:
            print(f"⚠️ 'COL_EMBEDDING' column missing in found original_pred_entry for FP: V:{video_val}, F:{frame_of_fp_val}. Columns: {original_pred_entry_df.columns}")
            continue
        orig_row_embedding_val = original_pred_entry_df[COL_EMBEDDING].iloc[0]
        if not isinstance(orig_row_embedding_val, np.ndarray):
            print(f"⚠️ Embedding for FP is not a numpy array. Type: {type(orig_row_embedding_val)}. Skipping.")
            continue
        negative_class_label = f"not_{wrongly_predicted_as_class_val}"
        fp_data_list.append({
            COL_CLASS: negative_class_label,
            COL_EMBEDDING: orig_row_embedding_val
        })
        exclusion_list.append({
            COL_VIDEO: video_val,
            COL_FRAME: frame_of_fp_val, 
            COL_CLASS: negative_class_label, 
            'original_wrong_prediction': wrongly_predicted_as_class_val,
            'reason': 'false_positive_negative'
        })
    fp_df_output = pd.DataFrame(fp_data_list) 
    if not fp_df_output.empty:
        if COL_CLASS not in fp_df_output.columns: fp_df_output[COL_CLASS] = None
        if COL_EMBEDDING not in fp_df_output.columns: fp_df_output[COL_EMBEDDING] = None
        fp_df_output = fp_df_output[[COL_CLASS, COL_EMBEDDING]] 
    print(f"✅ Extracted {len(fp_df_output)} embeddings for negative examples.")
    if not fp_df_output.empty:
        print(f"🏷️ Negative classes created: {fp_df_output[COL_CLASS].value_counts().to_dict()}")
    return fp_df_output, exclusion_list 

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
    debug_log(f"\n🔍 FILTER_EVALUATION_DATA CALLED:")
    debug_log(f"   resnet_data size: {len(resnet_data)}")
    debug_log(f"   exclude_training: {exclude_training}")
    debug_log(f"   exclusion_tracker: {type(exclusion_tracker)}")
    debug_log(f"   exclusion_tracker empty: {not exclusion_tracker}")
    debug_log(f"   exclusion_tracker keys: {list(exclusion_tracker.keys()) if exclusion_tracker else 'None'}")
    
    if not exclude_training or not exclusion_tracker:
        debug_log("🚫 No evaluation data filtering applied")
        debug_log(f"   Reason: exclude_training={exclude_training}, exclusion_tracker_empty={not exclusion_tracker}")
        return resnet_data.copy()
    
    # Collect all excluded video-frame pairs from all iterations
    excluded_pairs = set()
    total_exclusions = 0
    
    # Debug: Track exclusion details
    debug_exclusions = []
    
    for iteration_key, exclusions in exclusion_tracker.items():
        debug_log(f"🔍 DEBUG: Processing {len(exclusions)} exclusions from iteration {iteration_key}")
        for exclusion in exclusions:
            video = exclusion[COL_VIDEO]
            frame = exclusion[COL_FRAME]
            
            # Debug: Check for type mismatches
            debug_exclusions.append({
                'iteration': iteration_key,
                'video': video,
                'frame': frame,
                'video_type': type(video).__name__,
                'frame_type': type(frame).__name__
            })
            
            # Store both as string and original type for matching
            pair_str = (video, str(frame))  # String version for matching predictions
            pair_orig = (video, frame)      # Original version
            excluded_pairs.add(pair_str)
            excluded_pairs.add(pair_orig)
            total_exclusions += 1
    
    if total_exclusions == 0:
        debug_log("🚫 No exclusions to apply")
        return resnet_data.copy()
    
    debug_log(f"🔍 DEBUG: Total exclusion pairs created: {len(excluded_pairs)} (includes type variants)")
    debug_log(f"🔍 DEBUG: Sample exclusion types:")
    for i, exc in enumerate(debug_exclusions[:3]):
        debug_log(f"   {i+1}: Video={exc['video']} ({exc['video_type']}), Frame={exc['frame']} ({exc['frame_type']})")
    
    # Debug: Check data types in resnet_data
    if COL_VIDEO in resnet_data.columns and COL_FRAME in resnet_data.columns:
        sample_video = resnet_data[COL_VIDEO].iloc[0] if len(resnet_data) > 0 else None
        sample_frame = resnet_data[COL_FRAME].iloc[0] if len(resnet_data) > 0 else None
        debug_log(f"🔍 DEBUG: Sample resnet_data types: Video={sample_video} ({type(sample_video).__name__}), Frame={sample_frame} ({type(sample_frame).__name__})")
    
    # Filter out excluded pairs with detailed tracking
    def is_not_excluded(row):
        # Try both string and original type for frame matching
        pair_str = (row[COL_VIDEO], str(row[COL_FRAME]))
        pair_orig = (row[COL_VIDEO], row[COL_FRAME])
        return pair_str not in excluded_pairs and pair_orig not in excluded_pairs
    
    original_count = len(resnet_data)
    
    # Apply filter with debug tracking
    mask = resnet_data.apply(is_not_excluded, axis=1)
    filtered_data = resnet_data[mask].copy()
    
    filtered_count = len(filtered_data)
    excluded_count = original_count - filtered_count
    
    debug_log(f"🚫 Excluded {excluded_count} training samples from evaluation")
    debug_log(f"📊 Evaluation data: {original_count} → {filtered_count} samples")
    debug_log(f"🔍 DEBUG: Expected exclusions={total_exclusions}, Actual exclusions={excluded_count}")
    
    if excluded_count != total_exclusions:
        debug_log(f"⚠️ WARNING: Exclusion count mismatch! Expected {total_exclusions}, got {excluded_count}")
        debug_log(f"   Difference: {abs(excluded_count - total_exclusions)} samples")
        
        # Additional debugging: check for duplicates in exclusions
        exclusion_pairs_list = [(exc[COL_VIDEO], exc[COL_FRAME]) for iter_exclusions in exclusion_tracker.values() for exc in iter_exclusions]
        unique_exclusion_pairs = set(exclusion_pairs_list)
        debug_log(f"🔍 DEBUG: Total exclusion entries: {len(exclusion_pairs_list)}")
        debug_log(f"🔍 DEBUG: Unique exclusion pairs: {len(unique_exclusion_pairs)}")
        
        if len(exclusion_pairs_list) != len(unique_exclusion_pairs):
            debug_log(f"🔍 DEBUG: Found {len(exclusion_pairs_list) - len(unique_exclusion_pairs)} duplicate exclusions")
    
    return filtered_data 