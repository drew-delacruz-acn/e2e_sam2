import pandas as pd
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple

from .iteration_manager import run_single_iteration
from .training_utils import train_contrastive_representatives
from .prediction_utils import generate_predictions
from .evaluation_utils import evaluate_predictions, extract_false_positives
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

        new_training_data, metrics, new_exclusions = run_single_iteration(
            iteration=i, 
            training_data=current_training_data,
            resnet_data_for_prediction=resnet_data_for_prediction,
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
        
        metrics_to_store = {'iteration': i, **metrics}
        all_iteration_metrics.append(metrics_to_store)
        
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
        'final_exclusion_count': sum(len(v) for v in exclusion_tracker.values())
    }
    summary_path = output_base_dir / "pipeline_summary.json"
    try:
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary_data, f, indent=2)
        print(f"\n📜 Pipeline summary saved to {summary_path}. Final F1: {all_iteration_metrics[-1]['f1']:.4f if all_iteration_metrics else 'N/A'}.")
    except Exception as e:
        print(f"❌ Error saving pipeline summary: {e}")

    final_data_path = output_base_dir / "final_training_data.pkl"
    try:
        with open(final_data_path, 'wb') as f:
            pickle.dump(current_training_data, f)
        print(f"💾 Final combined training data saved to {final_data_path}.")
    except Exception as e:
        print(f"❌ Error saving final training data: {e}")

    return {'final_metrics': all_iteration_metrics[-1] if all_iteration_metrics else None} 