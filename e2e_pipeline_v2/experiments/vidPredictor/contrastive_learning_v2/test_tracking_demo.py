#!/usr/bin/env python3
"""
Demo script to show the comprehensive tracking system with sample data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from iterative_pipeline.tracking_utils import IterationTracker

def create_sample_data():
    """Create sample data to demonstrate tracking system."""
    
    # Sample evaluation data (before filtering)
    sample_eval_data = pd.DataFrame({
        'video': ['Scene_001', 'Scene_001', 'Scene_002', 'Scene_002', 'Scene_003'] * 20,
        'frame': list(range(10, 110, 1)),
        'owl_label': ['Loki_Armor', 'TVA_Collar', 'TemPad', 'Loki_Dagger', 'TVA_Monitor'] * 20,
        'finetuned_embedding': [np.random.randn(512) for _ in range(100)]
    })
    
    # Sample evaluation data (after filtering) - smaller
    sample_eval_data_filtered = sample_eval_data.iloc[:80].copy()  # 20 samples filtered out
    
    # Sample exclusions
    sample_exclusions = [
        {'video': 'Scene_001', 'frame': 15, 'class': 'not_Loki_Armor', 'original_wrong_prediction': 'Loki_Armor', 'reason': 'false_positive_negative'},
        {'video': 'Scene_002', 'frame': 25, 'class': 'not_TVA_Collar', 'original_wrong_prediction': 'TVA_Collar', 'reason': 'false_positive_negative'},
        {'video': 'Scene_003', 'frame': 35, 'class': 'not_TemPad', 'original_wrong_prediction': 'TemPad', 'reason': 'false_positive_negative'},
    ]
    
    # Sample false positives data
    sample_fp_data = pd.DataFrame({
        'class': ['not_Loki_Armor', 'not_TVA_Collar', 'not_TemPad'],
        'finetuned_embedding': [np.random.randn(512), np.random.randn(512), np.random.randn(512)]
    })
    
    # Sample evaluation results
    sample_eval_results = pd.DataFrame({
        'video': ['Scene_001', 'Scene_002', 'Scene_003'],
        'class': ['Loki_Armor', 'TVA_Collar', 'TemPad'],
        'actual': [0, 0, 0],  # All false positives
        'predicted': [1, 1, 1],
        'confidence': [0.85, 0.92, 0.78],
        'frame': [15, 25, 35],
        'classification': ['FP', 'FP', 'FP']
    })
    
    # Sample metrics
    sample_metrics = {
        'f1': 0.6234,
        'precision': 0.5891,
        'recall': 0.6612,
        'TP': 45,
        'FP': 12,
        'FN': 23,
        'TN': 120
    }
    
    return sample_eval_data, sample_eval_data_filtered, sample_exclusions, sample_fp_data, sample_eval_results, sample_metrics

def demo_tracking_system():
    """Demonstrate the comprehensive tracking system."""
    
    print("🚀 COMPREHENSIVE TRACKING SYSTEM DEMO")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("tracking_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize tracker
    tracker = IterationTracker(output_dir)
    
    # Create sample data
    eval_data_before, eval_data_after, exclusions, fp_data, eval_results, metrics = create_sample_data()
    
    print(f"\n📊 Simulating 2 iterations of the pipeline...")
    
    # Iteration 1
    print(f"\n🔄 ITERATION 1:")
    tracker.export_evaluation_data_before(1, eval_data_before)
    tracker.export_evaluation_data_after(1, eval_data_after, len(eval_data_before) - len(eval_data_after))
    tracker.export_exclusions_added(1, exclusions)
    tracker.export_false_positives_extracted(1, fp_data, eval_results)
    tracker.export_iteration_summary(1, metrics, len(exclusions), 90)
    
    # Iteration 2 (with different data)
    print(f"\n🔄 ITERATION 2:")
    # Modify data for iteration 2
    eval_data_after_2 = eval_data_after.iloc[:70].copy()  # Even more filtering
    exclusions_2 = [
        {'video': 'Scene_004', 'frame': 45, 'class': 'not_Sylvie_Armor', 'original_wrong_prediction': 'Sylvie_Armor', 'reason': 'false_positive_negative'},
        {'video': 'Scene_005', 'frame': 55, 'class': 'not_TimeSpear', 'original_wrong_prediction': 'TimeSpear', 'reason': 'false_positive_negative'},
    ]
    fp_data_2 = pd.DataFrame({
        'class': ['not_Sylvie_Armor', 'not_TimeSpear'],
        'finetuned_embedding': [np.random.randn(512), np.random.randn(512)]
    })
    eval_results_2 = pd.DataFrame({
        'video': ['Scene_004', 'Scene_005'],
        'class': ['Sylvie_Armor', 'TimeSpear'],
        'actual': [0, 0],
        'predicted': [1, 1],
        'confidence': [0.82, 0.89],
        'frame': [45, 55],
        'classification': ['FP', 'FP']
    })
    metrics_2 = {
        'f1': 0.7123,
        'precision': 0.6845,
        'recall': 0.7421,
        'TP': 52,
        'FP': 8,
        'FN': 18,
        'TN': 122
    }
    
    tracker.export_evaluation_data_before(2, eval_data_before)  # Same input data
    tracker.export_evaluation_data_after(2, eval_data_after_2, len(eval_data_before) - len(eval_data_after_2))
    tracker.export_exclusions_added(2, exclusions_2)
    tracker.export_false_positives_extracted(2, fp_data_2, eval_results_2)
    tracker.export_iteration_summary(2, metrics_2, len(exclusions_2), 95)
    
    # Generate cumulative analysis
    print(f"\n📊 Generating cumulative analysis...")
    tracker.export_cumulative_analysis()
    
    print(f"\n✅ Demo complete! Check the output directory: {output_dir.resolve()}")
    print(f"\n📁 Files created:")
    
    # List all files created
    for file_path in sorted(output_dir.rglob("*")):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            print(f"   📄 {file_path.name} ({file_size:,} bytes)")
    
    return output_dir

if __name__ == "__main__":
    demo_dir = demo_tracking_system()
    
    print(f"\n🔍 WHAT YOU CAN DO NOW:")
    print(f"1. Explore the tracking_exports/ directory in {demo_dir}")
    print(f"2. Open the CSV files in Excel/Numbers for detailed analysis")
    print(f"3. Review the JSON summaries for iteration-by-iteration tracking")
    print(f"4. Check the cumulative analysis files for overall patterns")
    print(f"\n📊 KEY FILES TO EXAMINE:")
    print(f"   - cumulative_exclusions_all.csv: All exclusions across iterations")
    print(f"   - exclusion_impact_summary.csv: How exclusions affect data sizes")
    print(f"   - class_exclusion_summary.csv: Which classes get excluded most")
    print(f"   - iteration_N_*.csv: Per-iteration detailed data") 