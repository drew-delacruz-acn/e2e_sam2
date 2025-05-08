import pandas as pd
import json
import numpy as np
import os

def blurriness_filter(json_path, csv_path, output_path,
                     metric='laplacian', 
                     method='percentile', 
                     threshold=75):
    """
    Filter object detections based on image blurriness metrics.
    
    Parameters:
    json_path (str): Path to the combined_search_results_resnet50.json file
    csv_path (str): Path to the agg_results.csv file
    output_path (str): Path for the filtered JSON output
    metric (str): The blurriness metric to filter by (e.g., 'laplacian', 'tenengrad', 'fft', 'is_blurry_fixed')
    method (str): Method to use - 'percentile', 'std_dev', or 'threshold'
    threshold (float): The threshold value appropriate to the selected method
    
    Returns:
    str: Description of the filtering applied and results
    """
    # Read the CSV metrics
    metrics_df = pd.read_csv(csv_path)
    
    # Validate metric exists
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in CSV. Available metrics: {list(metrics_df.columns)}")
    
    # Initialize the description string
    description = f"Filtering based on {metric} using {method} method"
    
    # Calculate thresholds dynamically based on specified method
    if method == 'percentile':
        # For metrics like laplacian, higher = sharper, so we keep above the percentile
        # For metrics like is_blurry_*, we would invert the logic
        keeping_higher = not metric.startswith('is_blurry_')
        
        if keeping_higher:
            percentile_value = np.percentile(metrics_df[metric], threshold)
            description += f" (keeping frames above {threshold}th percentile: {percentile_value:.2f})"
            metric_condition = {'min': percentile_value}
        else:
            percentile_value = np.percentile(metrics_df[metric], 100 - threshold)
            description += f" (keeping frames below {100-threshold}th percentile: {percentile_value:.2f})"
            metric_condition = {'max': percentile_value}
    
    elif method == 'std_dev':
        mean_value = metrics_df[metric].mean()
        std_value = metrics_df[metric].std()
        lower_bound = mean_value - (threshold * std_value)
        upper_bound = mean_value + (threshold * std_value)
        description += f" (keeping frames within {threshold} std devs from mean: [{lower_bound:.2f}, {upper_bound:.2f}])"
        metric_condition = {'min': lower_bound, 'max': upper_bound}
    
    elif method == 'threshold':
        if isinstance(threshold, dict):
            metric_condition = threshold
            threshold_desc = []
            if 'min' in threshold:
                threshold_desc.append(f"min={threshold['min']}")
            if 'max' in threshold:
                threshold_desc.append(f"max={threshold['max']}")
            description += f" (keeping frames with {', '.join(threshold_desc)})"
        elif isinstance(threshold, bool) and metric.startswith('is_blurry_'):
            # Special handling for boolean is_blurry fields
            metric_condition = threshold
            description += f" (keeping {'blurry' if threshold else 'non-blurry'} frames)"
        else:
            metric_condition = {'min': threshold}
            description += f" (keeping frames with {metric} >= {threshold})"
    
    else:
        raise ValueError("Invalid method. Choose 'percentile', 'std_dev', or 'threshold'")
    
    # Create lookup dictionary for frame quality based on dynamic thresholds
    frame_quality = {}
    for _, row in metrics_df.iterrows():
        scene = row['scene']
        frame = str(row['processed_index'])  # Convert to string to match JSON format
        
        # Check if frame meets condition based on metric and method
        meets_condition = True
        if isinstance(metric_condition, dict):
            if 'min' in metric_condition and row[metric] < metric_condition['min']:
                meets_condition = False
            if 'max' in metric_condition and row[metric] > metric_condition['max']:
                meets_condition = False
        else:
            if row[metric] != metric_condition:
                meets_condition = False
        
        if scene not in frame_quality:
            frame_quality[scene] = {}
        frame_quality[scene][frame] = meets_condition
    
    # Read the JSON detections
    with open(json_path, 'r') as f:
        detections = json.load(f)
    
    # Filter the detections
    filtered_detections = []
    for detection in detections:
        video = detection['video']
        frame = detection['frame']
        
        # Check if this frame meets our quality conditions
        if video in frame_quality and frame in frame_quality[video] and frame_quality[video][frame]:
            filtered_detections.append(detection)
    
    # Save the filtered results
    with open(output_path, 'w') as f:
        json.dump(filtered_detections, f, indent=2)
    
    # Add results to the description
    description += f"\nOriginal detections: {len(detections)}"
    description += f"\nFiltered detections: {len(filtered_detections)} ({len(filtered_detections)/len(detections)*100:.1f}%)"
    description += f"\nSaved filtered results to {os.path.basename(output_path)}"
    
    print(description)
    return description


# Example usage
if __name__ == "__main__":
    # Example 1: Filter using percentile method (top 25% sharpest frames)
    """
    blurriness_filter(
        'combined_search_results_resnet50.json',
        'e2e_pipeline_v2/experiments/agg_results.csv',
        'filtered_laplacian_top25.json',
        metric='laplacian',
        method='percentile',
        threshold=75
    )
    """
    
    # Example 2: Filter using standard deviation method
    """
    blurriness_filter(
        'combined_search_results_resnet50.json',
        'e2e_pipeline_v2/experiments/agg_results.csv',
        'filtered_tenengrad_std.json',
        metric='tenengrad',
        method='std_dev',
        threshold=1.0
    )
    """
    
    # Example 3: Keep only non-blurry frames
    """
    blurriness_filter(
        'combined_search_results_resnet50.json',
        'e2e_pipeline_v2/experiments/agg_results.csv',
        'filtered_nonblurry.json',
        metric='is_blurry_fixed',
        method='threshold',
        threshold=False
    )
    """ 