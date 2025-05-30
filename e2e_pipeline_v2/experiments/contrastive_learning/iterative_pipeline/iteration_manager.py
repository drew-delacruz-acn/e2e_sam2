import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple

# Assuming other necessary utility functions will be imported from their respective modules
# e.g., from .training_utils import train_contrastive_representatives
# from .prediction_utils import generate_predictions
# from .evaluation_utils import evaluate_predictions, extract_false_positives
from .tracking_utils import get_tracker
from .pipeline_manager import debug_log

# Column name constants
COL_CLASS = 'class'
COL_EMBEDDING = 'finetuned_embedding'


def run_single_iteration(
    iteration: int,
    training_data: pd.DataFrame,
    resnet_data_for_prediction: pd.DataFrame,
    ground_truth: pd.DataFrame,
    exclusion_tracker: Dict, 
    current_iter_threshold: float, 
    current_iter_margin: float,
    output_base_dir: Path, 
    epochs: int, 
    train_reps_func, 
    generate_preds_func, 
    evaluate_preds_func, 
    extract_fps_func 
) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    """
    Run a single iteration of the pipeline: train, predict, evaluate, extract FPs.
    """
    print(f"\n🔄 ITERATION {iteration} (Threshold: {current_iter_threshold}, Margin: {current_iter_margin})\n" + "=" * 50)
    
    iteration_output_dir = output_base_dir / f"iteration_{iteration}"
    iteration_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏋️ Phase 2A: Training representatives...")
    representatives_path = train_reps_func( 
        training_data, 
        iteration_output_dir, 
        epochs, 
        current_iter_margin
    )

    print(f"\n🔮 Phase 2B: Generating predictions...")
    predictions_for_eval = generate_preds_func( 
        resnet_data_for_prediction, 
        representatives_path, 
        current_iter_threshold, 
        iteration_number=iteration
    )
    
    print(f"DEBUG (run_single_iteration): predictions_for_eval index is_unique: {predictions_for_eval.index.is_unique}")
    if not predictions_for_eval.empty:
        print(f"DEBUG (run_single_iteration): predictions_for_eval head:\n{predictions_for_eval.head(2)}")
        if not predictions_for_eval.index.is_unique:
            print(f"DEBUG (run_single_iteration): Duplicated indices in predictions_for_eval:\n{predictions_for_eval.index[predictions_for_eval.index.duplicated()].unique()}")
    else:
        print("DEBUG (run_single_iteration): predictions_for_eval is empty.")

    print(f"\n📊 Phase 2C: Evaluating predictions...")
    evaluation_results, metrics = evaluate_preds_func(predictions_for_eval, ground_truth) 

    print(f"\n🚨 Phase 2D: Extracting false positives...")
    fp_data, new_exclusions = extract_fps_func( 
        evaluation_results, 
        predictions_for_eval, 
        exclusion_tracker 
    )
    
    # DEBUG: Log what was returned from extract_false_positives
    debug_log(f"🔍 DEBUG ITERATION: extract_fps_func returned:")
    debug_log(f"   fp_data shape: {fp_data.shape if hasattr(fp_data, 'shape') else 'Not a DataFrame'}")
    debug_log(f"   new_exclusions type: {type(new_exclusions)}")
    debug_log(f"   new_exclusions length: {len(new_exclusions) if new_exclusions else 0}")
    debug_log(f"   new_exclusions is None: {new_exclusions is None}")
    debug_log(f"   new_exclusions is empty list: {new_exclusions == []}")
    if new_exclusions and len(new_exclusions) > 0:
        debug_log(f"   Sample exclusion: {new_exclusions[0]}")
    else:
        debug_log(f"   ⚠️ WARNING: new_exclusions is empty!")

    # 📊 TRACKING: Export false positives extracted
    tracker = get_tracker()
    if tracker:
        tracker.export_false_positives_extracted(iteration, fp_data, evaluation_results)

    print(f"\n💾 Saving iteration results to {iteration_output_dir}...")
    evaluation_results.to_csv(iteration_output_dir / "evaluation_results.csv", index=False)
    if not fp_data.empty:
        fp_data.to_csv(iteration_output_dir / "false_positives_for_training.csv", index=False)
        with open(iteration_output_dir / "false_positives_for_training.pkl", 'wb') as f:
            pickle.dump(fp_data, f)
    
    # DEBUG: Log exclusions saving logic
    debug_log(f"🔍 DEBUG ITERATION: Checking if exclusions should be saved...")
    debug_log(f"   new_exclusions: {new_exclusions}")
    debug_log(f"   bool(new_exclusions): {bool(new_exclusions)}")
    debug_log(f"   Condition 'if new_exclusions': {bool(new_exclusions)}")
    
    if new_exclusions: 
        debug_log(f"🔍 DEBUG ITERATION: Saving {len(new_exclusions)} exclusions to exclusions.json")
        with open(iteration_output_dir / "exclusions.json", 'w') as f:
            json.dump(new_exclusions, f, indent=2)
    else:
        debug_log(f"🔍 DEBUG ITERATION: NOT saving exclusions.json because new_exclusions is falsy")
        debug_log(f"   This is why no exclusions.json file was created!")
    
    with open(iteration_output_dir / "training_results.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    current_training_data_len = len(training_data)
    new_training_data_iter = training_data.copy() 

    if not fp_data.empty:
        temp_td = new_training_data_iter.copy()
        temp_fp = fp_data.copy()
        
        for df, _ in [(temp_td, "temp_td"), (temp_fp, "temp_fp")]:
            if COL_EMBEDDING in df.columns and not df.empty and isinstance(df[COL_EMBEDDING].iloc[0], np.ndarray):
                df['emb_tuple'] = df[COL_EMBEDDING].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
            elif COL_EMBEDDING in df.columns:
                df['emb_tuple'] = df[COL_EMBEDDING] 
            else: df['emb_tuple'] = pd.Series(dtype='object', index=df.index)
        
        dedup_cols = [COL_CLASS]
        can_use_emb_tuple = (
            'emb_tuple' in temp_td.columns and temp_td['emb_tuple'].notna().any() and
            'emb_tuple' in temp_fp.columns and temp_fp['emb_tuple'].notna().any() and
            all(isinstance(x, tuple) for x in temp_td['emb_tuple'].dropna() if pd.notna(x)) and
            all(isinstance(x, tuple) for x in temp_fp['emb_tuple'].dropna() if pd.notna(x))
        )
        
        if can_use_emb_tuple:
            dedup_cols.append('emb_tuple')
            cols_to_keep_from_temp = [COL_CLASS, COL_EMBEDDING, 'emb_tuple']
            df_list_for_concat = []
            if all(c in temp_td.columns for c in cols_to_keep_from_temp): df_list_for_concat.append(temp_td[cols_to_keep_from_temp])
            if all(c in temp_fp.columns for c in cols_to_keep_from_temp): df_list_for_concat.append(temp_fp[cols_to_keep_from_temp])

            if df_list_for_concat: 
                combined_temp = pd.concat(df_list_for_concat, ignore_index=True)
                if not combined_temp.empty:
                    new_training_data_iter = combined_temp.drop_duplicates(subset=dedup_cols, keep='first')[[COL_CLASS, COL_EMBEDDING]].reset_index(drop=True)
                else: 
                    new_training_data_iter = pd.DataFrame(columns=[COL_CLASS, COL_EMBEDDING]) 
            else: 
                can_use_emb_tuple = False 
        
        if not can_use_emb_tuple: 
            print("⚠️ Fallback deduplication path chosen.")
            combined_orig = pd.concat([training_data, fp_data], ignore_index=True)
            if COL_CLASS in combined_orig.columns:
                new_training_data_iter = combined_orig.drop_duplicates(subset=[COL_CLASS], keep='first').reset_index(drop=True)
            else: new_training_data_iter = combined_orig

        added_count = len(new_training_data_iter) - current_training_data_len
        print(f"🔄 Training data update: {current_training_data_len} initial + {len(fp_data)} new FPs -> {len(new_training_data_iter)} total ({added_count} unique added).")
    else:
        print(f"🔄 No new FPs. Training data size: {len(new_training_data_iter)}.")
        
    print(f"\n📊 ITERATION {iteration} SUMMARY: F1:{metrics['f1']:.4f}, P:{metrics['precision']:.4f}, R:{metrics['recall']:.4f}. FPs for train: {len(fp_data)}. New Excl: {len(new_exclusions)}. Train Size: {len(new_training_data_iter)}")
    return new_training_data_iter, metrics, new_exclusions 