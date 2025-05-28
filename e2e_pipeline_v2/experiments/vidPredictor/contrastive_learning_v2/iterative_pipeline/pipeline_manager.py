import pandas as pd
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple
import numpy as np

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

        # Apply exclusion strategy to evaluation data
        if config.exclude_training_from_eval:
            eval_data = filter_evaluation_data(resnetPredictions, exclusion_tracker, exclude_training=True)
            eval_mode = "clean"
            print("📉 Using CLEAN evaluation (excluding training data)")
        elif config.include_training_in_eval:
            eval_data = resnetPredictions.copy()
            eval_mode = "contaminated"
            print("📈 Using CONTAMINATED evaluation (including training data)")
        elif config.track_training_separately:
            # Use full data first, then run clean evaluation separately
            eval_data = resnetPredictions.copy()
            eval_mode = "both"
            print("📊 Using BOTH evaluations (will run clean and contaminated)")
        else:
            # Default to clean
            eval_data = filter_evaluation_data(resnetPredictions, exclusion_tracker, exclude_training=True)
            eval_mode = "clean"
            print("📉 Using CLEAN evaluation (default)")

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
        
        current_training_data = new_training_data
        if new_exclusions: 
            exclusion_tracker[i] = new_exclusions 
        
        # Add evaluation mode to metrics
        metrics_to_store = {'iteration': i, 'eval_mode': eval_mode, 'evaluation_samples': len(eval_data), **metrics}
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

    return {'final_metrics': all_iteration_metrics[-1] if all_iteration_metrics else None} 