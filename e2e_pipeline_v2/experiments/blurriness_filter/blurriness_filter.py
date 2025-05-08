import pandas as pd
import json
import numpy as np
import os
import argparse

def blurriness_filter(json_path, csv_path, output_path=None,
                     metric='laplacian', 
                     method='percentile', 
                     threshold=75):
    """
    Filter object detections based on image blurriness metrics, potentially on a per-scene basis.
    
    Parameters:
    json_path (str): Path to the combined_search_results_resnet50.json file
    csv_path (str): Path to the agg_results.csv file
    output_path (str, optional): Path for the filtered JSON output. If None, will generate a name based on parameters.
    metric (str): The blurriness metric to filter by (e.g., 'laplacian', 'tenengrad', 'fft', 'is_blurry_fixed')
    method (str): Method to use - 'percentile', 'std_dev', or 'threshold'
    threshold (float): The threshold value appropriate to the selected method
    
    Returns:
    str: Description of the filtering applied and results
    """
    # Generate output path if not provided
    if output_path is None:
        if isinstance(threshold, bool):
            threshold_str = str(threshold).lower()
        else:
            threshold_str = str(threshold).replace('.', 'p')
        output_path = f"filtered_{metric}_{method}_{threshold_str}_scene_specific.json"
        print(f"No output path provided, using auto-generated filename: {output_path}")
    
    metrics_df = pd.read_csv(csv_path)
    
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in CSV. Available metrics: {list(metrics_df.columns)}")
    
    description = f"Filtering based on {metric} using {method} method (scene-specific thresholds)."
    
    unique_scenes = metrics_df['scene'].unique()
    scene_specific_conditions = {}
    detailed_description_parts = [description]

    for scene_name in unique_scenes:
        scene_metrics_df = metrics_df[metrics_df['scene'] == scene_name]
        scene_desc_part = f"\n  Scene '{scene_name}':"

        current_metric_condition = {}
        if method == 'percentile':
            keeping_higher = not metric.startswith('is_blurry_')
            if not scene_metrics_df[metric].empty:
                if keeping_higher:
                    percentile_value = np.percentile(scene_metrics_df[metric].dropna(), threshold)
                    scene_desc_part += f" keeping frames above {threshold}th percentile ({percentile_value:.2f})"
                    current_metric_condition = {'min': percentile_value}
                else:
                    percentile_value = np.percentile(scene_metrics_df[metric].dropna(), 100 - threshold)
                    scene_desc_part += f" keeping frames below {100-threshold}th percentile ({percentile_value:.2f})"
                    current_metric_condition = {'max': percentile_value}
            else:
                scene_desc_part += f" no data for metric '{metric}', skipping percentile calculation."
                # Decide how to handle scenes with no data for the metric, e.g., keep all or none
                # For now, let's assume we keep none if condition can't be calculated (empty current_metric_condition)

        elif method == 'std_dev':
            if not scene_metrics_df[metric].empty and len(scene_metrics_df[metric].dropna()) > 1: # std needs at least 2 points
                mean_value = scene_metrics_df[metric].mean()
                std_value = scene_metrics_df[metric].std()
                if pd.isna(std_value): # Handle case where std is NaN (e.g. all values are the same)
                     lower_bound = mean_value
                     upper_bound = mean_value
                     scene_desc_part += f" all values are identical ({mean_value:.2f}), keeping if equal."
                else:
                    lower_bound = mean_value - (threshold * std_value)
                    upper_bound = mean_value + (threshold * std_value)
                    scene_desc_part += f" keeping frames within {threshold} std devs from mean [{lower_bound:.2f}, {upper_bound:.2f}]"
                current_metric_condition = {'min': lower_bound, 'max': upper_bound}
            elif not scene_metrics_df[metric].empty and len(scene_metrics_df[metric].dropna()) == 1:
                value = scene_metrics_df[metric].dropna().iloc[0]
                scene_desc_part += f" only one data point ({value:.2f}), keeping if equal."
                current_metric_condition = {'min': value, 'max': value}
            else:
                scene_desc_part += f" not enough data for metric '{metric}' for std_dev calculation, skipping."
        
        elif method == 'threshold':
            if isinstance(threshold, dict): # min/max dict provided directly
                current_metric_condition = threshold 
                threshold_desc_parts = []
                if 'min' in threshold: threshold_desc_parts.append(f"min={threshold['min']}")
                if 'max' in threshold: threshold_desc_parts.append(f"max={threshold['max']}")
                scene_desc_part += f" keeping frames with {', '.join(threshold_desc_parts)}"
            elif isinstance(threshold, bool) and metric.startswith('is_blurry_'):
                current_metric_condition = threshold
                scene_desc_part += f" keeping {'blurry' if threshold else 'non-blurry'} frames"
            else: # single numeric threshold, assume it's a minimum
                current_metric_condition = {'min': threshold}
                scene_desc_part += f" keeping frames with {metric} >= {threshold}"
        
        else:
            raise ValueError("Invalid method. Choose 'percentile', 'std_dev', or 'threshold'")
        
        scene_specific_conditions[scene_name] = current_metric_condition
        detailed_description_parts.append(scene_desc_part)

    description = "".join(detailed_description_parts)
    
    frame_quality = {}
    for _, row in metrics_df.iterrows():
        scene = row['scene']
        frame = str(row['processed_index'])
        
        # Get the condition for this specific scene
        current_scene_condition = scene_specific_conditions.get(scene, {}) # Default to empty if scene somehow missed
        
        meets_condition = True # Assume true unless a check fails or no condition exists
        if not current_scene_condition: # If no condition was set for the scene (e.g. no data)
            meets_condition = False # Default to not meeting condition
        elif isinstance(current_scene_condition, dict):
            if 'min' in current_scene_condition and (pd.isna(row[metric]) or row[metric] < current_scene_condition['min']):
                meets_condition = False
            if 'max' in current_scene_condition and (pd.isna(row[metric]) or row[metric] > current_scene_condition['max']):
                meets_condition = False
        else: # Boolean condition for is_blurry_*
            if pd.isna(row[metric]) or row[metric] != current_scene_condition:
                meets_condition = False
        
        if scene not in frame_quality:
            frame_quality[scene] = {}
        frame_quality[scene][frame] = meets_condition
    
    with open(json_path, 'r') as f:
        detections = json.load(f)
    
    filtered_detections = []
    for detection in detections:
        video = detection['video'] # Assuming 'video' column in JSON corresponds to 'scene' in CSV
        frame = detection['frame']
        
        if video in frame_quality and frame in frame_quality[video] and frame_quality[video][frame]:
            filtered_detections.append(detection)
            
    with open(output_path, 'w') as f:
        json.dump(filtered_detections, f, indent=2)
    
    final_description = description
    final_description += f"\n\nOriginal detections: {len(detections)}"
    final_description += f"\nFiltered detections: {len(filtered_detections)} ({len(filtered_detections)/len(detections)*100:.1f}% if detections > 0 else 0.0)%"
    final_description += f"\nSaved filtered results to {os.path.basename(output_path)}"
    
    print(final_description)
    return final_description

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter object detections based on image blurriness metrics, with scene-specific thresholds.')
    
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to the combined_search_results_resnet50.json file')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to the agg_results.csv file')
    parser.add_argument('--output_path', type=str, required=False, default=None,
                       help='Path for the filtered JSON output (optional, will auto-generate if not provided)')
    parser.add_argument('--metric', type=str, default='laplacian',
                       help='The blurriness metric to filter by (e.g., laplacian, tenengrad, fft, is_blurry_fixed)')
    parser.add_argument('--method', type=str, default='percentile', choices=['percentile', 'std_dev', 'threshold'],
                       help="Method to use - 'percentile', 'std_dev', or 'threshold'")
    parser.add_argument('--threshold', type=float, default=75,
                       help="The threshold value appropriate to the selected method (e.g., 75 for 75th percentile, 1.0 for 1 std dev)")
    parser.add_argument('--boolean_threshold', action='store_true',
                       help="Set threshold to True (for keeping blurry) instead of False (for keeping non-blurry) when using 'is_blurry_fixed' metric with 'threshold' method. Default is False (keep non-blurry).")
    
    args = parser.parse_args()
    
    actual_threshold = args.threshold
    if args.metric.startswith('is_blurry_') and args.method == 'threshold':
        actual_threshold = args.boolean_threshold # If flag is present, it's True, else False by default action
    
    blurriness_filter(
        args.json_path,
        args.csv_path,
        args.output_path,
        metric=args.metric,
        method=args.method,
        threshold=actual_threshold
    )

# Example usage (commented out, use CLI)
# ... (rest of the example usage comments remain the same) 