import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Optional

# Assuming other necessary utility functions will be imported from their respective modules
# e.g., from .training_utils import train_contrastive_representatives
# from .prediction_utils import generate_predictions
# from .evaluation_utils import evaluate_predictions, extract_false_positives
from .tracking_utils import get_tracker
from .debug_utils import debug_log

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
    extract_fps_func,
    exclusion_strategy: str = 'frame-level',
    max_fps_per_class: Optional[int] = None
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
    evaluation_results, metrics = evaluate_preds_func(predictions_for_eval, ground_truth, exclusion_tracker, exclusion_strategy) 

    print(f"\n🚨 Phase 2D: Extracting false positives...")
    fp_data, new_exclusions = extract_fps_func( 
        evaluation_results, 
        predictions_for_eval, 
        exclusion_tracker,
        max_fps_per_class
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

    debug_log(f"🔍 TRAINING_DATA DEBUG: Starting training data update process")
    debug_log(f"   Iteration {iteration}: current_training_data_len = {current_training_data_len}")
    debug_log(f"   fp_data.empty = {fp_data.empty}")
    debug_log(f"   fp_data length = {len(fp_data) if not fp_data.empty else 0}")

    if not fp_data.empty:
        debug_log(f"🔍 TRAINING_DATA DEBUG: Creating temp DataFrames")
        temp_td = new_training_data_iter.copy()
        temp_fp = fp_data.copy()
        
        # DEBUG: Log initial DataFrame info
        debug_log(f"   temp_td shape: {temp_td.shape}")
        debug_log(f"   temp_td columns: {list(temp_td.columns)}")
        debug_log(f"   temp_fp shape: {temp_fp.shape}")
        debug_log(f"   temp_fp columns: {list(temp_fp.columns)}")
        
        for df, name in [(temp_td, "temp_td"), (temp_fp, "temp_fp")]:
            debug_log(f"🔍 TRAINING_DATA DEBUG: Processing {name}")
            debug_log(f"   {name} has COL_EMBEDDING: {COL_EMBEDDING in df.columns}")
            debug_log(f"   {name} is empty: {df.empty}")
            
            if COL_EMBEDDING in df.columns and not df.empty:
                first_embedding = df[COL_EMBEDDING].iloc[0]
                debug_log(f"   {name} first embedding type BEFORE: {type(first_embedding)}")
                debug_log(f"   {name} first embedding is ndarray BEFORE: {isinstance(first_embedding, np.ndarray)}")
                
                # SOLUTION 1: Normalize all embeddings to numpy arrays first
                debug_log(f"   🔧 FIXING: Converting all embeddings to numpy arrays first")
                df[COL_EMBEDDING] = df[COL_EMBEDDING].apply(lambda x: 
                    np.array(x) if isinstance(x, list) else x
                )
                
                # Verify conversion
                first_embedding_after = df[COL_EMBEDDING].iloc[0]
                debug_log(f"   {name} first embedding type AFTER: {type(first_embedding_after)}")
                debug_log(f"   {name} first embedding is ndarray AFTER: {isinstance(first_embedding_after, np.ndarray)}")
                
                # Then convert to tuples consistently
                df['emb_tuple'] = df[COL_EMBEDDING].apply(lambda x: 
                    tuple(x) if isinstance(x, np.ndarray) else x
                )
                debug_log(f"   ✅ {name} standardized embeddings and added emb_tuple column")
                
                # Verify final tuple conversion
                first_tuple = df['emb_tuple'].iloc[0]
                debug_log(f"   {name} first emb_tuple type: {type(first_tuple)}")
                debug_log(f"   {name} first emb_tuple is tuple: {isinstance(first_tuple, tuple)}")
            else:
                df['emb_tuple'] = pd.Series(dtype='object', index=df.index)
                debug_log(f"   {name} added empty emb_tuple column (missing COL_EMBEDDING or empty)")
        
        dedup_cols = [COL_CLASS]
        debug_log(f"🔍 TRAINING_DATA DEBUG: Checking emb_tuple compatibility")
        debug_log(f"   temp_td has emb_tuple: {'emb_tuple' in temp_td.columns}")
        debug_log(f"   temp_fp has emb_tuple: {'emb_tuple' in temp_fp.columns}")
        
        if 'emb_tuple' in temp_td.columns:
            debug_log(f"   temp_td emb_tuple notna count: {temp_td['emb_tuple'].notna().sum()}")
            if temp_td['emb_tuple'].notna().any():
                first_tuple = temp_td['emb_tuple'].dropna().iloc[0]
                debug_log(f"   temp_td first emb_tuple type: {type(first_tuple)}")
                debug_log(f"   temp_td first emb_tuple is tuple: {isinstance(first_tuple, tuple)}")
        
        if 'emb_tuple' in temp_fp.columns:
            debug_log(f"   temp_fp emb_tuple notna count: {temp_fp['emb_tuple'].notna().sum()}")
            if temp_fp['emb_tuple'].notna().any():
                first_tuple = temp_fp['emb_tuple'].dropna().iloc[0]
                debug_log(f"   temp_fp first emb_tuple type: {type(first_tuple)}")
                debug_log(f"   temp_fp first emb_tuple is tuple: {isinstance(first_tuple, tuple)}")
        
        can_use_emb_tuple = (
            'emb_tuple' in temp_td.columns and temp_td['emb_tuple'].notna().any() and
            'emb_tuple' in temp_fp.columns and temp_fp['emb_tuple'].notna().any() and
            all(isinstance(x, tuple) for x in temp_td['emb_tuple'].dropna() if x is not None) and
            all(isinstance(x, tuple) for x in temp_fp['emb_tuple'].dropna() if x is not None)
        )
        
        debug_log(f"🔍 TRAINING_DATA DEBUG: can_use_emb_tuple = {can_use_emb_tuple}")
        
        # Additional validation logging
        if 'emb_tuple' in temp_td.columns and temp_td['emb_tuple'].notna().any():
            td_tuple_types = [type(x) for x in temp_td['emb_tuple'].dropna()[:3]]
            debug_log(f"   temp_td sample emb_tuple types: {td_tuple_types}")
            td_all_tuples = all(isinstance(x, tuple) for x in temp_td['emb_tuple'].dropna() if x is not None)
            debug_log(f"   temp_td all emb_tuples are tuples: {td_all_tuples}")
        
        if 'emb_tuple' in temp_fp.columns and temp_fp['emb_tuple'].notna().any():
            fp_tuple_types = [type(x) for x in temp_fp['emb_tuple'].dropna()[:3]]
            debug_log(f"   temp_fp sample emb_tuple types: {fp_tuple_types}")
            fp_all_tuples = all(isinstance(x, tuple) for x in temp_fp['emb_tuple'].dropna() if x is not None)
            debug_log(f"   temp_fp all emb_tuples are tuples: {fp_all_tuples}")
        
        if can_use_emb_tuple:
            debug_log(f"   🎉 SUCCESS: Data types are now compatible - using sophisticated deduplication!")
            dedup_cols.append('emb_tuple')
            cols_to_keep_from_temp = [COL_CLASS, COL_EMBEDDING, 'emb_tuple']
            debug_log(f"   cols_to_keep_from_temp: {cols_to_keep_from_temp}")
            
            df_list_for_concat = []
            
            # Check temp_td columns
            td_has_all_cols = all(c in temp_td.columns for c in cols_to_keep_from_temp)
            debug_log(f"   temp_td has all required columns: {td_has_all_cols}")
            debug_log(f"   temp_td missing columns: {[c for c in cols_to_keep_from_temp if c not in temp_td.columns]}")
            if td_has_all_cols:
                df_list_for_concat.append(temp_td[cols_to_keep_from_temp])
                debug_log(f"   ✅ Added temp_td to concat list (shape: {temp_td[cols_to_keep_from_temp].shape})")
            else:
                debug_log(f"   ❌ temp_td REJECTED - missing required columns!")
            
            # Check temp_fp columns  
            fp_has_all_cols = all(c in temp_fp.columns for c in cols_to_keep_from_temp)
            debug_log(f"   temp_fp has all required columns: {fp_has_all_cols}")
            debug_log(f"   temp_fp missing columns: {[c for c in cols_to_keep_from_temp if c not in temp_fp.columns]}")
            if fp_has_all_cols:
                df_list_for_concat.append(temp_fp[cols_to_keep_from_temp])
                debug_log(f"   ✅ Added temp_fp to concat list (shape: {temp_fp[cols_to_keep_from_temp].shape})")
            else:
                debug_log(f"   ❌ temp_fp REJECTED - missing required columns!")

            debug_log(f"🔍 TRAINING_DATA DEBUG: df_list_for_concat length: {len(df_list_for_concat)}")
            debug_log(f"   DataFrames in concat list: {[df.shape for df in df_list_for_concat]}")

            if df_list_for_concat: 
                combined_temp = pd.concat(df_list_for_concat, ignore_index=True)
                debug_log(f"   Combined temp shape before dedup: {combined_temp.shape}")
                debug_log(f"   Combined temp classes: {combined_temp[COL_CLASS].value_counts().to_dict()}")
                
                if not combined_temp.empty:
                    debug_log(f"   Applying deduplication on columns: {dedup_cols}")
                    new_training_data_iter = combined_temp.drop_duplicates(subset=dedup_cols, keep='first')[[COL_CLASS, COL_EMBEDDING]].reset_index(drop=True)
                    debug_log(f"   ✅ SOPHISTICATED PATH SUCCESS: Final shape {new_training_data_iter.shape}")
                    debug_log(f"   Final classes: {new_training_data_iter[COL_CLASS].value_counts().to_dict()}")
                else: 
                    debug_log(f"   ❌ Combined temp is empty!")
                    new_training_data_iter = pd.DataFrame(columns=[COL_CLASS, COL_EMBEDDING]) 
            else: 
                debug_log(f"   ❌ df_list_for_concat is empty - falling back!")
                can_use_emb_tuple = False 
        
        if not can_use_emb_tuple: 
            debug_log(f"🔍 TRAINING_DATA DEBUG: Using fallback deduplication path")
            print("⚠️ Fallback deduplication path chosen.")
            combined_orig = pd.concat([training_data, fp_data], ignore_index=True)
            debug_log(f"   Fallback combined shape: {combined_orig.shape}")
            debug_log(f"   Fallback combined classes: {combined_orig[COL_CLASS].value_counts().to_dict()}")
            
            if COL_CLASS in combined_orig.columns:
                new_training_data_iter = combined_orig.drop_duplicates(subset=[COL_CLASS], keep='first').reset_index(drop=True)
                debug_log(f"   ✅ FALLBACK PATH: Final shape {new_training_data_iter.shape}")
                debug_log(f"   Final classes: {new_training_data_iter[COL_CLASS].value_counts().to_dict()}")
            else: 
                debug_log(f"   ❌ COL_CLASS missing from combined_orig!")
                new_training_data_iter = combined_orig

        added_count = len(new_training_data_iter) - current_training_data_len
        debug_log(f"🔍 TRAINING_DATA DEBUG: FINAL SUMMARY")
        debug_log(f"   Started with: {current_training_data_len} samples")
        debug_log(f"   Added FPs: {len(fp_data)} samples") 
        debug_log(f"   Final result: {len(new_training_data_iter)} samples")
        debug_log(f"   Net change: {added_count} samples")
        debug_log(f"   Expected minimum: {current_training_data_len + len(fp_data)} samples")
        if len(new_training_data_iter) < current_training_data_len:
            debug_log(f"   🚨 CRITICAL BUG: Training data SHRANK by {current_training_data_len - len(new_training_data_iter)} samples!")
    else:
        print(f"🔄 No new FPs. Training data size: {len(new_training_data_iter)}.")
        
    print(f"\n📊 ITERATION {iteration} SUMMARY: F1:{metrics['f1']:.4f}, P:{metrics['precision']:.4f}, R:{metrics['recall']:.4f}. FPs for train: {len(fp_data)}. New Excl: {len(new_exclusions)}. Train Size: {len(new_training_data_iter)}")
    return new_training_data_iter, metrics, new_exclusions 