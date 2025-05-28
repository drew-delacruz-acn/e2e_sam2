#!/usr/bin/env python3
"""
Extract hard negatives from false positives analysis results.

This script processes false positives CSV files to create hard negative samples
for enhanced contrastive learning training.
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from typing import Dict, List, Any
import torch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract hard negatives from false positives')
    parser.add_argument('--false_positives_csv', type=str, required=True,
                       help='Path to false positives CSV file')
    parser.add_argument('--resnet_data', type=str, required=True,
                       help='Path to ResNet embeddings pickle file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for hard negatives pickle file')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                       help='Minimum confidence threshold for hard negatives')
    parser.add_argument('--max_per_class', type=int, default=50,
                       help='Maximum hard negatives per class')
    parser.add_argument('--exclude_training_data', action='store_true',
                       help='Exclude samples that might be in training data')
    
    return parser.parse_args()

def load_false_positives(csv_path: str) -> pd.DataFrame:
    """Load false positives from CSV file."""
    print(f"📥 Loading false positives from: {csv_path}")
    fp_df = pd.read_csv(csv_path)
    print(f"   Found {len(fp_df)} false positive samples")
    return fp_df

def load_resnet_embeddings(pickle_path: str) -> pd.DataFrame:
    """Load ResNet embeddings from pickle file."""
    print(f"📥 Loading ResNet embeddings from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        resnet_df = pickle.load(f)
    print(f"   Found {len(resnet_df)} embedding samples")
    return resnet_df

def extract_hard_negatives(fp_df: pd.DataFrame, resnet_df: pd.DataFrame, 
                          min_confidence: float, max_per_class: int,
                          exclude_training: bool = False) -> Dict[str, List[Dict]]:
    """
    Extract hard negatives from false positives.
    
    Args:
        fp_df: False positives DataFrame
        resnet_df: ResNet embeddings DataFrame
        min_confidence: Minimum confidence threshold
        max_per_class: Maximum samples per class
        exclude_training: Whether to exclude potential training samples
        
    Returns:
        Dictionary mapping ground_truth_class -> list of hard negative embeddings
    """
    print(f"🎯 Extracting hard negatives...")
    print(f"   Min confidence: {min_confidence}")
    print(f"   Max per class: {max_per_class}")
    print(f"   Exclude training: {exclude_training}")
    
    # Filter by confidence
    high_conf_fp = fp_df[fp_df['confidence'] >= min_confidence].copy()
    print(f"   High confidence FPs: {len(high_conf_fp)}")
    
    # Create lookup for ResNet embeddings
    # Assuming ResNet data has 'video', 'frame', 'owl_label', 'finetuned_embedding'
    resnet_lookup = {}
    for _, row in resnet_df.iterrows():
        key = (row['video'], row['frame'], row['owl_label'])
        resnet_lookup[key] = row['finetuned_embedding']
    
    hard_negatives = {}
    
    # Group by ground truth class
    for gt_class in high_conf_fp['ground_truth_class'].unique():
        class_fps = high_conf_fp[high_conf_fp['ground_truth_class'] == gt_class]
        
        # Sort by confidence (highest first - these are the hardest negatives)
        class_fps = class_fps.sort_values('confidence', ascending=False)
        
        # Limit number per class
        class_fps = class_fps.head(max_per_class)
        
        hard_negatives[gt_class] = []
        
        for _, fp_row in class_fps.iterrows():
            # Find corresponding embedding in ResNet data
            # Match by video, frame, and predicted class (owl_label)
            lookup_key = (fp_row['video'], fp_row['frame'], fp_row['predicted_class'])
            
            if lookup_key in resnet_lookup:
                embedding = resnet_lookup[lookup_key]
                
                # Convert to tensor if it's numpy array
                if isinstance(embedding, np.ndarray):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                elif isinstance(embedding, list):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                
                hard_negatives[gt_class].append({
                    'embedding': embedding,
                    'video': fp_row['video'],
                    'frame': fp_row['frame'],
                    'predicted_class': fp_row['predicted_class'],
                    'confidence': fp_row['confidence']
                })
        
        print(f"   {gt_class}: {len(hard_negatives[gt_class])} hard negatives")
    
    # Remove empty classes
    hard_negatives = {k: v for k, v in hard_negatives.items() if len(v) > 0}
    
    total_negatives = sum(len(negs) for negs in hard_negatives.values())
    print(f"✅ Extracted {total_negatives} hard negatives across {len(hard_negatives)} classes")
    
    return hard_negatives

def save_hard_negatives(hard_negatives: Dict[str, List[Dict]], output_path: str):
    """Save hard negatives to pickle file."""
    print(f"💾 Saving hard negatives to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(hard_negatives, f)
    
    print("✅ Hard negatives saved successfully")

def print_summary(hard_negatives: Dict[str, List[Dict]]):
    """Print summary of extracted hard negatives."""
    print("\n" + "="*60)
    print("🎯 HARD NEGATIVES SUMMARY")
    print("="*60)
    
    total_negatives = sum(len(negs) for negs in hard_negatives.values())
    print(f"Total hard negatives: {total_negatives}")
    print(f"Classes with hard negatives: {len(hard_negatives)}")
    print()
    
    print("Per-class breakdown:")
    for class_name, negatives in sorted(hard_negatives.items()):
        avg_conf = np.mean([neg['confidence'] for neg in negatives])
        print(f"  {class_name}: {len(negatives)} samples (avg conf: {avg_conf:.3f})")
    
    print("\nTop 5 hardest negatives overall:")
    all_negatives = []
    for class_name, negatives in hard_negatives.items():
        for neg in negatives:
            all_negatives.append((class_name, neg))
    
    # Sort by confidence (highest = hardest)
    all_negatives.sort(key=lambda x: x[1]['confidence'], reverse=True)
    
    for i, (class_name, neg) in enumerate(all_negatives[:5]):
        print(f"  {i+1}. {class_name} -> {neg['predicted_class']} "
              f"(conf: {neg['confidence']:.3f}, {neg['video']}:{neg['frame']})")

def main():
    args = parse_args()
    
    print("🚀 Starting hard negatives extraction...")
    print(f"   False positives: {args.false_positives_csv}")
    print(f"   ResNet data: {args.resnet_data}")
    print(f"   Output: {args.output_path}")
    print()
    
    # Load data
    fp_df = load_false_positives(args.false_positives_csv)
    resnet_df = load_resnet_embeddings(args.resnet_data)
    
    # Extract hard negatives
    hard_negatives = extract_hard_negatives(
        fp_df, resnet_df, 
        args.min_confidence, 
        args.max_per_class,
        args.exclude_training_data
    )
    
    # Save results
    save_hard_negatives(hard_negatives, args.output_path)
    
    # Print summary
    print_summary(hard_negatives)
    
    print("\n🎉 Hard negatives extraction completed!")

if __name__ == "__main__":
    main() 