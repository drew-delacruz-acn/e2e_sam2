import pandas as pd
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

from .iteration_manager import run_single_iteration
from .training_utils import train_contrastive_representatives
from .prediction_utils import generate_predictions
from .evaluation_utils import evaluate_predictions, extract_false_positives, filter_evaluation_data
from .config import PipelineConfig 

def run_iterative_pipeline(
    definitiveObjects: pd.DataFrame,
    resnetPredictions: pd.DataFrame, 
    trackingInfo: pd.DataFrame,    
    config: PipelineConfig 
) -> Dict:
    """
    Run the full iterative contrastive learning pipeline.
    """
    print("\n🚀 Starting Iterative Contrastive Learning Pipeline (Negative Classes Mode)")
    
    current_training_data = definitiveObjects.copy()
    exclusion_tracker: Dict[int, List[Dict]] = {} 
    all_iteration_metrics: List[Dict] = []
    previous_f1 = -1.0 

    output_base_dir = Path(config.output) 

    # Debug: Print configuration values
    print(f"\n🔍 FILTERING DEBUG: Configuration values:")
    print(f"   exclude_training_from_eval: {config.exclude_training_from_eval}")
    print(f"   include_training_in_eval: {config.include_training_in_eval}")
    print(f"   track_training_separately: {config.track_training_separately}")

    if config.include_training_in_eval:
        print("📈 Evaluation will INCLUDE all data (training data not explicitly excluded from resnetPredictions).")
    else: 
        print("📉 Evaluation behavior: Standard, FPs added to training are not re-evaluated as FPs in the same way if they are learned.")

    resnet_data_for_prediction = resnetPredictions.copy()

    for i in range(1, config.iterations + 1):
        current_iter_threshold_to_use = config.threshold 
        current_iter_margin_to_use = config.margin 

        if i > 1: 
            if config.secondary_threshold is not None:
                current_iter_threshold_to_use = config.secondary_threshold
            if config.secondary_margin is not None:
                current_iter_margin_to_use = config.secondary_margin
        
        if i == 1:
            print(f"💡 Iteration {i}: Using Primary Threshold: {current_iter_threshold_to_use}, Primary Margin: {current_iter_margin_to_use}")
        else:
            threshold_type = "Secondary" if config.secondary_threshold is not None and i > 1 else "Primary"
            margin_type = "Secondary" if config.secondary_margin is not None and i > 1 else "Primary"
            print(f"💡 Iteration {i}: Using Threshold: {current_iter_threshold_to_use} ({threshold_type}), Margin: {current_iter_margin_to_use} ({margin_type})")

        # Debug: Print exclusion tracker state
        print(f"\n🔍 FILTERING DEBUG (Iteration {i}):")
        print(f"   Exclusion tracker keys: {list(exclusion_tracker.keys())}")
        print(f"   Total exclusions so far: {sum(len(v) for v in exclusion_tracker.values())}")
        for iter_key, exclusions in exclusion_tracker.items():
            print(f"   Iteration {iter_key}: {len(exclusions)} exclusions")

        # Apply exclusion strategy to evaluation data
        print(f"\n🔍 FILTERING DEBUG: Applying exclusion strategy...")
        original_eval_size = len(resnetPredictions)
        print(f"   Original resnetPredictions size: {original_eval_size}")
        
        if config.exclude_training_from_eval:
            print(f"   🔍 Calling filter_evaluation_data with exclude_training=True")
            eval_data = filter_evaluation_data(resnetPredictions, exclusion_tracker, exclude_training=True)
            eval_mode = "clean"
            print("📉 Using CLEAN evaluation (excluding training data)")
        elif config.include_training_in_eval:
            print(f"   🔍 Skipping filtering - include_training_in_eval=True")
            eval_data = resnetPredictions.copy()
            eval_mode = "contaminated"
            print("📈 Using CONTAMINATED evaluation (including training data)")
        elif config.track_training_separately:
            print(f"   🔍 Using both evaluations - track_training_separately=True")
            # Use full data first, then run clean evaluation separately
            eval_data = resnetPredictions.copy()
            eval_mode = "both"
            print("📊 Using BOTH evaluations (will run clean and contaminated)")
        else:
            print(f"   🔍 Default to clean - calling filter_evaluation_data")
            # Default to clean
            eval_data = filter_evaluation_data(resnetPredictions, exclusion_tracker, exclude_training=True)
            eval_mode = "clean"
            print("📉 Using CLEAN evaluation (default)")

        filtered_eval_size = len(eval_data)
        print(f"🔍 FILTERING DEBUG: Evaluation data size after filtering:")
        print(f"   Original: {original_eval_size}")
        print(f"   Filtered: {filtered_eval_size}")
        print(f"   Difference: {original_eval_size - filtered_eval_size}")
        
        if original_eval_size == filtered_eval_size and exclusion_tracker:
            print(f"   ⚠️ WARNING: No size change despite having exclusions!")
        elif original_eval_size != filtered_eval_size:
            print(f"   ✅ Filtering applied: {original_eval_size - filtered_eval_size} samples removed")

        new_training_data, metrics, new_exclusions = run_single_iteration(
            iteration=i, 
            training_data=current_training_data,
            resnet_data_for_prediction=eval_data,
            ground_truth=trackingInfo, 
            exclusion_tracker=exclusion_tracker, 
            current_iter_threshold=current_iter_threshold_to_use,
            current_iter_margin=current_iter_margin_to_use, 
            output_base_dir=output_base_dir, 
            epochs=config.epochs,           
            train_reps_func=train_contrastive_representatives,
            generate_preds_func=generate_predictions,
            evaluate_preds_func=evaluate_predictions,
            extract_fps_func=extract_false_positives
        )
        
        # Add evaluation mode to metrics with CORRECTED evaluation sample count
        # The actual evaluation happens on the data AFTER generate_predictions processing
        # So we need to get the true evaluation size from the metrics or calculate it
        actual_evaluation_size = len(eval_data)  # This will be corrected below
        
        print(f"🔍 PIPELINE DEBUG: eval_data input size: {len(eval_data)}")
        print(f"🔍 PIPELINE DEBUG: Checking if actual evaluation size differs...")
        
        # The true evaluation size should be reflected in the total samples evaluated
        total_eval_samples = metrics.get('TP', 0) + metrics.get('FP', 0) + metrics.get('FN', 0) + metrics.get('TN', 0)
        if total_eval_samples > 0 and total_eval_samples != len(eval_data):
            print(f"🔍 PIPELINE DEBUG: Data reduction detected!")
            print(f"   Input to iteration: {len(eval_data)} samples")
            print(f"   Actual evaluation: {total_eval_samples} samples") 
            print(f"   Reduction: {len(eval_data) - total_eval_samples} samples lost in generate_predictions")
            actual_evaluation_size = total_eval_samples
        
        current_training_data = new_training_data
        if new_exclusions: 
            exclusion_tracker[i] = new_exclusions 
        
        # Add evaluation mode to metrics
        metrics_to_store = {'iteration': i, 'eval_mode': eval_mode, 'evaluation_samples': actual_evaluation_size, **metrics}
        all_iteration_metrics.append(metrics_to_store)
        
        # Handle track_training_separately mode
        if config.track_training_separately:
            # Run additional clean evaluation for comparison
            clean_eval_data = filter_evaluation_data(resnetPredictions, exclusion_tracker, exclude_training=True)
            
            if len(clean_eval_data) != len(eval_data):
                print(f"\n🔄 Running clean evaluation for comparison...")
                
                # Load representatives and generate clean predictions  
                representatives_path = output_base_dir / f"iteration_{i}" / "representatives.pkl"
                clean_predictions = generate_predictions(clean_eval_data, representatives_path, current_iter_threshold_to_use, i)
                clean_eval_results, clean_metrics = evaluate_predictions(clean_predictions, trackingInfo)
                
                print(f"📊 CLEAN vs CONTAMINATED COMPARISON:")
                print(f"   Clean F1: {clean_metrics['f1']:.4f} ({len(clean_eval_data)} samples)")
                print(f"   Contaminated F1: {metrics['f1']:.4f} ({len(eval_data)} samples)")
                
                # Save clean metrics
                clean_metrics['iteration'] = i
                clean_metrics['eval_mode'] = 'clean'
                clean_metrics['evaluation_samples'] = len(clean_eval_data)
                
                with open(output_base_dir / f"iteration_{i}" / "clean_metrics.json", 'w') as f:
                    json.dump(clean_metrics, f, indent=2)
            else:
                print(f"\n📊 Clean and contaminated data are the same size - no exclusions to compare")
        
        current_f1 = metrics.get('f1', 0.0) 
        if i > 1 and previous_f1 >= 0 and abs(current_f1 - previous_f1) < config.convergence_threshold:
            print(f"\n✅ Convergence reached at iteration {i}: F1 improvement ({current_f1 - previous_f1:.4f}) is less than threshold ({config.convergence_threshold:.4f}).")
            break
        previous_f1 = current_f1
        
        if i == config.iterations:
            print("\n🏁 Maximum iterations reached.")
            
    pipeline_summary_data = {
        'config': config.__dict__, 
        'iteration_metrics': all_iteration_metrics,
        'final_training_data_size': len(current_training_data),
        'final_exclusion_count': sum(len(v) for v in exclusion_tracker.values()),
        'evaluation_strategy': {
            'exclude_training_from_eval': config.exclude_training_from_eval,
            'include_training_in_eval': config.include_training_in_eval,
            'track_training_separately': config.track_training_separately
        }
    }
    summary_path = output_base_dir / "pipeline_summary.json"
    try:
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary_data, f, indent=2)
        
        final_f1_str = 'N/A'
        if all_iteration_metrics and 'f1' in all_iteration_metrics[-1]:
            f1_val = all_iteration_metrics[-1]['f1']
            if isinstance(f1_val, (float, np.floating)):
                final_f1_str = f"{f1_val:.4f}"
            else:
                final_f1_str = str(f1_val)
        elif all_iteration_metrics:
            final_f1_raw = all_iteration_metrics[-1].get('f1')
            if isinstance(final_f1_raw, (float, np.floating)):
                final_f1_str = f"{final_f1_raw:.4f}"
            elif final_f1_raw is not None:
                final_f1_str = str(final_f1_raw)

        print(f"\n📜 Pipeline summary saved to {summary_path}. Final F1: {final_f1_str}.")
    except Exception as e:
        print(f"❌ Error saving pipeline summary: {e}")

    # Save exclusion tracker data
    exclusion_path = output_base_dir / "cumulative_exclusions.json"
    try:
        with open(exclusion_path, 'w') as f:
            json.dump(exclusion_tracker, f, indent=2)
        print(f"📍 Exclusion tracker saved to {exclusion_path}.")
    except Exception as e:
        print(f"❌ Error saving exclusion tracker: {e}")

    final_data_path = output_base_dir / "final_training_data.pkl"
    try:
        with open(final_data_path, 'wb') as f:
            pickle.dump(current_training_data, f)
        print(f"💾 Final combined training data saved to {final_data_path}.")
    except Exception as e:
        print(f"❌ Error saving final training data: {e}")

    # Generate analysis logs for debugging
    try:
        original_data_size = len(resnetPredictions)
        chat_log_file = create_analysis_logs(output_base_dir, all_iteration_metrics, exclusion_tracker, config, original_data_size)
        print(f"\n🔍 Analysis complete! Copy {chat_log_file} contents to chat for debugging.")
    except Exception as e:
        print(f"❌ Error creating analysis logs: {e}")

    return {'final_metrics': all_iteration_metrics[-1] if all_iteration_metrics else None} 

def create_analysis_logs(output_base_dir: Path, all_iteration_metrics: List[Dict], 
                        exclusion_tracker: Dict, config: PipelineConfig, original_data_size: int):
    """
    Create compact, analysis-ready logs for debugging evaluation strategies.
    These logs are designed to be easily copied from VM and pasted into chat.
    """
    logs_dir = output_base_dir / "analysis_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. EVALUATION STRATEGY SUMMARY (most important for debugging)
    eval_summary = []
    eval_summary.append("=== EVALUATION STRATEGY ANALYSIS ===")
    eval_summary.append(f"Timestamp: {timestamp}")
    eval_summary.append(f"Strategy: exclude_training={config.exclude_training_from_eval}, include_training={config.include_training_in_eval}, track_separately={config.track_training_separately}")
    eval_summary.append(f"Original Data Size: {original_data_size}")
    eval_summary.append("")
    
    # Iteration-by-iteration breakdown
    eval_summary.append("ITER | MODE         | EVAL_SIZE | EXCLUSIONS | F1      | PRECISION | RECALL")
    eval_summary.append("-----|--------------|-----------|------------|---------|-----------|--------")
    
    cumulative_exclusions = 0
    for i, metrics in enumerate(all_iteration_metrics, 1):
        if i in exclusion_tracker:
            cumulative_exclusions += len(exclusion_tracker[i])
        
        eval_size = metrics.get('evaluation_samples', 'N/A')
        mode = metrics.get('eval_mode', 'unknown')
        f1 = metrics.get('f1', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        eval_summary.append(f"{i:4d} | {mode:12s} | {eval_size:9d} | {cumulative_exclusions:10d} | {f1:7.4f} | {precision:9.4f} | {recall:7.4f}")
    
    eval_summary.append("")
    eval_summary.append("=== KEY DIAGNOSTICS ===")
    
    # Check for unexpected patterns
    if all_iteration_metrics:
        final_exclusions = sum(len(v) for v in exclusion_tracker.values())
        final_eval_size = all_iteration_metrics[-1].get('evaluation_samples', original_data_size)
        expected_eval_size = original_data_size - final_exclusions
        
        eval_summary.append(f"Expected final eval size: {expected_eval_size}")
        eval_summary.append(f"Actual final eval size: {final_eval_size}")
        eval_summary.append(f"Size difference: {abs(expected_eval_size - final_eval_size)}")
        
        if abs(expected_eval_size - final_eval_size) > 0:
            eval_summary.append("⚠️  WARNING: Eval size mismatch - filtering may not be working correctly!")
        else:
            eval_summary.append("✅ Eval size matches expectations - filtering working correctly")
    
    # Save evaluation summary
    eval_file = logs_dir / f"evaluation_analysis_{timestamp}.txt"
    with open(eval_file, 'w') as f:
        f.write('\n'.join(eval_summary))
    
    # 2. EXCLUSION DETAILS (for debugging data quality)
    exclusion_details = []
    exclusion_details.append("=== EXCLUSION TRACKING DETAILS ===")
    exclusion_details.append(f"Total iterations: {len(all_iteration_metrics)}")
    exclusion_details.append(f"Total exclusions: {sum(len(v) for v in exclusion_tracker.values())}")
    exclusion_details.append("")
    
    for iteration, exclusions in exclusion_tracker.items():
        exclusion_details.append(f"ITERATION {iteration}: {len(exclusions)} exclusions")
        
        # Group by class for analysis
        class_counts = {}
        for exc in exclusions:
            cls = exc.get('class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            exclusion_details.append(f"  {cls}: {count} samples")
        
        # Show first few examples
        exclusion_details.append("  Sample exclusions:")
        for exc in exclusions[:3]:  # First 3 examples
            video = exc.get('video', 'N/A')[:20]  # Truncate long video names
            frame = exc.get('frame', 'N/A')
            cls = exc.get('class', 'N/A')
            exclusion_details.append(f"    {video}:{frame} -> {cls}")
        
        if len(exclusions) > 3:
            exclusion_details.append(f"    ... and {len(exclusions) - 3} more")
        exclusion_details.append("")
    
    exclusion_file = logs_dir / f"exclusion_details_{timestamp}.txt"
    with open(exclusion_file, 'w') as f:
        f.write('\n'.join(exclusion_details))
    
    # 3. METRICS COMPARISON (CSV format for easy analysis)
    metrics_csv = []
    metrics_csv.append("iteration,eval_mode,f1,precision,recall,tp,fp,fn,tn,eval_samples,exclusions")
    
    cumulative_exclusions = 0
    for i, metrics in enumerate(all_iteration_metrics, 1):
        if i in exclusion_tracker:
            cumulative_exclusions += len(exclusion_tracker[i])
        
        row = [
            str(i),
            metrics.get('eval_mode', 'unknown'),
            f"{metrics.get('f1', 0):.6f}",
            f"{metrics.get('precision', 0):.6f}",
            f"{metrics.get('recall', 0):.6f}",
            str(metrics.get('TP', 0)),
            str(metrics.get('FP', 0)),
            str(metrics.get('FN', 0)),
            str(metrics.get('TN', 0)),
            str(metrics.get('evaluation_samples', 0)),
            str(cumulative_exclusions)
        ]
        metrics_csv.append(','.join(row))
    
    metrics_file = logs_dir / f"metrics_comparison_{timestamp}.csv"
    with open(metrics_file, 'w') as f:
        f.write('\n'.join(metrics_csv))
    
    # 4. COMPACT SUMMARY FOR CHAT (single copyable block)
    chat_summary = []
    chat_summary.append("=== PIPELINE ANALYSIS SUMMARY (Copy to Chat) ===")
    chat_summary.append(f"Config: exclude_training={config.exclude_training_from_eval}, include_training={config.include_training_in_eval}")
    chat_summary.append(f"Data: {original_data_size} original samples, {len(all_iteration_metrics)} iterations")
    
    if all_iteration_metrics:
        first_f1 = all_iteration_metrics[0].get('f1', 0)
        last_f1 = all_iteration_metrics[-1].get('f1', 0)
        final_exclusions = sum(len(v) for v in exclusion_tracker.values())
        final_eval_size = all_iteration_metrics[-1].get('evaluation_samples', 0)
        
        chat_summary.append(f"Results: F1 {first_f1:.4f} -> {last_f1:.4f}, {final_exclusions} exclusions, {final_eval_size} final eval samples")
        
        # Flag unexpected patterns
        if config.exclude_training_from_eval and final_eval_size >= original_data_size:
            chat_summary.append("🚨 ISSUE: Clean mode should have fewer eval samples!")
        elif config.include_training_in_eval and final_eval_size < original_data_size:
            chat_summary.append("🚨 ISSUE: Contaminated mode should have all eval samples!")
    
    chat_summary.append("")
    chat_summary.append("Iteration Details:")
    for i, metrics in enumerate(all_iteration_metrics, 1):
        if i in exclusion_tracker:
            new_exclusions = len(exclusion_tracker[i])
        else:
            new_exclusions = 0
        
        eval_size = metrics.get('evaluation_samples', 0)
        f1 = metrics.get('f1', 0)
        mode = metrics.get('eval_mode', 'unknown')
        
        chat_summary.append(f"  Iter {i}: {mode} mode, {eval_size} samples, F1={f1:.4f}, +{new_exclusions} exclusions")
    
    chat_file = logs_dir / f"chat_summary_{timestamp}.txt"
    with open(chat_file, 'w') as f:
        f.write('\n'.join(chat_summary))
    
    # Print paths to all created files
    print(f"\n📊 Analysis logs created in {logs_dir}:")
    print(f"  📋 Evaluation analysis: {eval_file.name}")
    print(f"  📍 Exclusion details: {exclusion_file.name}")
    print(f"  📈 Metrics CSV: {metrics_file.name}")
    print(f"  💬 Chat summary: {chat_file.name}")
    print(f"\n📋 To analyze in chat, copy contents of: {chat_file}")
    
    return chat_file 