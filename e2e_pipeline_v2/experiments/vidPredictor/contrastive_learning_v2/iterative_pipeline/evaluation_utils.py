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

    print("📊 FRAME-LEVEL EVALUATION DEBUG:")
    print("   Ground truth rows: {}".format(len(gt_eval)))
    print("   Predictions rows: {}".format(len(predictions)))
    print("   Ground truth columns: {}".format(list(gt_eval.columns)))
    print("   Predictions columns: {}".format(list(predictions.columns)))

    # Check if frame columns exist
    if COL_FRAME not in gt_eval.columns:
        print("⚠️ WARNING: No frame column in ground truth. Available: {}".format(list(gt_eval.columns)))
        print("⚠️ WARNING: No frame column in ground truth. Falling back to video-class evaluation.")
        # Fall back to original logic if no frame data
        use_frame_matching = False
    elif COL_FRAME not in predictions.columns:
        print("⚠️ WARNING: No frame column in predictions. Available: {}".format(list(predictions.columns)))
        print("⚠️ WARNING: No frame column in predictions. Falling back to video-class evaluation.")
        use_frame_matching = False
    else:
        use_frame_matching = True
        print("✅ Frame columns found in both datasets. Using frame-level matching.")

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
        print("📊 FRAME-LEVEL MATCHING RESULTS:")
        print("   Frame matches found: {}".format(frame_matches))
        print("   Frame mismatches: {}".format(frame_mismatches))
        print("   Total evaluations: {}".format(len(eval_results)))
        print("   Match rate: {:.1f}%".format(frame_matches/len(eval_results)*100))
    
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
    # print("   📊 {eval_mode} EVALUATION: F1: {:.4f}, P: {:.4f}, R: {:.4f}. Counts: TP:{}, FP:{}, FN:{}, TN:{}".format(eval_mode, metrics['f1'], metrics['precision'], metrics['recall'], metrics['TP'], metrics['FP'], metrics['FN'], metrics['TN']))
    # print("📊 {eval_mode} EVALUATION COMPLETE: {} samples evaluated".format(len(eval_df)))
    
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
    print("🚨 Found {} false positive cases (from evaluation_results).".format(len(fp_cases)))
    fp_data_list = [] 
    exclusion_list = []
    required_pred_cols = [COL_VIDEO, COL_FRAME, COL_VISUAL_PRED_OBJECT, COL_EMBEDDING]
    if not all(col in predictions_for_eval.columns for col in required_pred_cols):
        missing_cols_str = ", ".join(set(required_pred_cols) - set(predictions_for_eval.columns))
        print("⚠️ 'predictions_for_eval' DataFrame is missing required columns for FP extraction: {}".format(missing_cols_str))
        print("   Available: {}".format(list(predictions_for_eval.columns)))
        return pd.DataFrame(columns=[COL_CLASS, COL_EMBEDDING]), []

    for _, fp_row in fp_cases.iterrows():
        video_val = fp_row[COL_VIDEO]
        wrongly_predicted_as_class_val = fp_row[COL_CLASS] 
        frame_of_fp_val = fp_row[COL_FRAME]
        if frame_of_fp_val is None or pd.isna(frame_of_fp_val):
            print("⚠️ Skipping FP with no valid frame: Video {}, Class {}".format(video_val, wrongly_predicted_as_class_val))
            continue
        
        # Safe frame value conversion with proper validation
        try:
            if isinstance(frame_of_fp_val, (int, float)):
                frame_of_fp_val = int(frame_of_fp_val)
            elif isinstance(frame_of_fp_val, str):
                frame_of_fp_val = int(float(frame_of_fp_val))  # Handle string numbers like "123.0"
            else:
                print("⚠️ Skipping FP with invalid frame type: Video {}, Class {}, Frame type: {}".format(video_val, wrongly_predicted_as_class_val, type(frame_of_fp_val)))
                continue
        except (ValueError, TypeError) as e:
            print("⚠️ Skipping FP with unconvertible frame value: Video {}, Class {}, Frame: {}, Error: {}".format(video_val, wrongly_predicted_as_class_val, frame_of_fp_val, e))
            continue
        original_pred_entry_df = predictions_for_eval[
            (predictions_for_eval[COL_VIDEO] == video_val) &
            (predictions_for_eval[COL_VISUAL_PRED_OBJECT] == wrongly_predicted_as_class_val) &
            (predictions_for_eval[COL_FRAME] == frame_of_fp_val) 
        ]
        if original_pred_entry_df.empty:
            print("⚠️ Could not find original embedding for FP in 'predictions_for_eval': "
                  "Video '{}', Frame {}, Predicted Class '{}'. "
                  "Check if 'predictions_for_eval' data is consistent with 'evaluation_results'.".format(video_val, frame_of_fp_val, wrongly_predicted_as_class_val))
            continue
        if COL_EMBEDDING not in original_pred_entry_df.columns:
            print("⚠️ 'COL_EMBEDDING' column missing in found original_pred_entry for FP: V:{} F:{}".format(video_val, frame_of_fp_val))
            print("   Columns: {}".format(original_pred_entry_df.columns))
            continue
        orig_row_embedding_val = original_pred_entry_df[COL_EMBEDDING].iloc[0]
        if not isinstance(orig_row_embedding_val, np.ndarray):
            print("⚠️ Embedding for FP is not a numpy array. Type: {}".format(type(orig_row_embedding_val)))
            continue
        negative_class_label = "not_{}".format(wrongly_predicted_as_class_val)
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
    print("✅ Extracted {} embeddings for negative examples.".format(len(fp_df_output)))
    if not fp_df_output.empty:
        print("🏷️ Negative classes created: {}".format(fp_df_output[COL_CLASS].value_counts().to_dict()))
    return fp_df_output, exclusion_list 

def filter_evaluation_data_enhanced(resnet_data: pd.DataFrame,
                                   exclusion_tracker: Dict,
                                   exclude_training: bool = True,
                                   exclusion_strategy: str = 'frame-level') -> pd.DataFrame:
    """
    Enhanced filter evaluation data based on exclusion strategy.
    
    Args:
        resnet_data: Original prediction data
        exclusion_tracker: Dictionary tracking excluded video-frame pairs
        exclude_training: Whether to exclude training data from evaluation
        exclusion_strategy: 'frame-level', 'video-level', or 'compare-both'
        
    Returns:
        Filtered DataFrame for evaluation
    """
    print(f"\n🔍 ENHANCED_FILTER_EVALUATION_DATA CALLED:")
    print(f"   resnet_data size: {len(resnet_data)}")
    print(f"   exclude_training: {exclude_training}")
    print(f"   exclusion_strategy: {exclusion_strategy}")
    print(f"   exclusion_tracker: {type(exclusion_tracker)}")
    print(f"   exclusion_tracker empty: {not exclusion_tracker}")
    print(f"   exclusion_tracker keys: {list(exclusion_tracker.keys()) if exclusion_tracker else 'None'}")
    
    if not exclude_training or not exclusion_tracker:
        print("🚫 No evaluation data filtering applied")
        print(f"   Reason: exclude_training={exclude_training}, exclusion_tracker_empty={not exclusion_tracker}")
        return resnet_data.copy()
    
    # Collect exclusions by strategy
    if exclusion_strategy == 'frame-level':
        return _apply_frame_level_exclusions(resnet_data, exclusion_tracker)
    elif exclusion_strategy == 'video-level':
        return _apply_video_level_exclusions(resnet_data, exclusion_tracker)
    else:
        # For 'compare-both', default to frame-level (comparison handled in pipeline_manager)
        print(f"📊 Using frame-level for compare-both strategy")
        return _apply_frame_level_exclusions(resnet_data, exclusion_tracker)


def _apply_frame_level_exclusions(resnet_data: pd.DataFrame, exclusion_tracker: Dict) -> pd.DataFrame:
    """Apply frame-level exclusions (original behavior)."""
    print(f"🔍 Applying FRAME-LEVEL exclusions...")
    
    # Collect all excluded video-frame pairs from all iterations
    excluded_pairs = set()
    total_exclusions = 0
    
    # Debug: Track exclusion details
    debug_exclusions = []
    
    for iteration_key, exclusions in exclusion_tracker.items():
        print(f"🔍 DEBUG: Processing {len(exclusions)} exclusions from iteration {iteration_key}")
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
        print("🚫 No exclusions to apply")
        return resnet_data.copy()
    
    print(f"🔍 DEBUG: Total exclusion pairs created: {len(excluded_pairs)} (includes type variants)")
    print(f"🔍 DEBUG: Sample exclusion types:")
    for i, exc in enumerate(debug_exclusions[:3]):
        print(f"   {i+1}: Video={exc['video']} ({exc['video_type']}), Frame={exc['frame']} ({exc['frame_type']})")
    
    # Debug: Check data types in resnet_data
    if COL_VIDEO in resnet_data.columns and COL_FRAME in resnet_data.columns:
        sample_video = resnet_data[COL_VIDEO].iloc[0] if len(resnet_data) > 0 else None
        sample_frame = resnet_data[COL_FRAME].iloc[0] if len(resnet_data) > 0 else None
        print(f"🔍 DEBUG: Sample resnet_data types: Video={sample_video} ({type(sample_video).__name__}), Frame={sample_frame} ({type(sample_frame).__name__})")
    
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
    
    print(f"🚫 FRAME-LEVEL: Excluded {excluded_count} training samples from evaluation")
    print(f"📊 FRAME-LEVEL: Evaluation data: {original_count} → {filtered_count} samples")
    print(f"🔍 DEBUG: Expected exclusions={total_exclusions}, Actual exclusions={excluded_count}")
    
    return filtered_data


def _apply_video_level_exclusions(resnet_data: pd.DataFrame, exclusion_tracker: Dict) -> pd.DataFrame:
    """Apply video-level exclusions (exclude entire videos if any frame is problematic)."""
    print(f"🚫 Applying VIDEO-LEVEL exclusions...")
    
    # Collect all excluded videos (not just video-frame pairs)
    excluded_videos = set()
    total_frame_exclusions = 0
    
    for iteration_key, exclusions in exclusion_tracker.items():
        print(f"🔍 DEBUG: Processing {len(exclusions)} exclusions from iteration {iteration_key}")
        for exclusion in exclusions:
            video = exclusion[COL_VIDEO]
            excluded_videos.add(video)
            total_frame_exclusions += 1
    
    if not excluded_videos:
        print("🚫 No video exclusions to apply")
        return resnet_data.copy()
    
    print(f"🔍 DEBUG: Videos to exclude: {len(excluded_videos)}")
    print(f"🔍 DEBUG: Original frame exclusions: {total_frame_exclusions}")
    print(f"🔍 DEBUG: Sample excluded videos: {list(excluded_videos)[:5]}")
    
    # Filter out entire videos
    def is_video_not_excluded(row):
        return row[COL_VIDEO] not in excluded_videos
    
    original_count = len(resnet_data)
    
    # Apply filter
    mask = resnet_data.apply(is_video_not_excluded, axis=1)
    filtered_data = resnet_data[mask].copy()
    
    filtered_count = len(filtered_data)
    excluded_count = original_count - filtered_count
    
    print(f"🚫 VIDEO-LEVEL: Excluded {excluded_count} samples from {len(excluded_videos)} videos")
    print(f"📊 VIDEO-LEVEL: Evaluation data: {original_count} → {filtered_count} samples")
    print(f"📊 VIDEO-LEVEL: Amplification factor: {excluded_count / total_frame_exclusions:.1f}x more data excluded than frame-level")
    
    return filtered_data


def filter_evaluation_data(resnet_data: pd.DataFrame,
                          exclusion_tracker: Dict,
                          exclude_training: bool = True,
                          exclusion_strategy: str = 'frame-level') -> pd.DataFrame:
    """
    Filter evaluation data based on exclusion strategy.
    
    Args:
        resnet_data: Original prediction data
        exclusion_tracker: Dictionary tracking excluded video-frame pairs
        exclude_training: Whether to exclude training data from evaluation
        exclusion_strategy: 'frame-level' or 'video-level' (NEW parameter)
        
    Returns:
        Filtered DataFrame for evaluation
    """
    # Use the enhanced filter function
    return filter_evaluation_data_enhanced(resnet_data, exclusion_tracker, exclude_training, exclusion_strategy) 