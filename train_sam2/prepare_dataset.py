#!/usr/bin/env python
"""
SAM2 Dataset Preparation Script
This script prepares training and validation splits for SAM2 model fine-tuning.
"""
import os
import argparse
import random
import json
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare dataset for SAM2 fine-tuning")
    parser.add_argument("--data_dir", type=str, default="../../../data/davis-2017/DAVIS/",
                      help="Path to DAVIS dataset")
    parser.add_argument("--sequence", type=str, default="bear",
                      help="DAVIS sequence to use")
    parser.add_argument("--val_split", type=float, default=0.2,
                      help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./dataset_splits",
                      help="Output directory for dataset splits")
    return parser.parse_args()

def load_dataset_paths(data_dir, sequence):
    """Load dataset file paths"""
    data = []
    img_dir = os.path.join(data_dir, f"JPEGImages/480p/{sequence}/")
    ann_dir = os.path.join(data_dir, f"Annotations/480p/{sequence}/")
    
    # Check if directories exist
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    
    # Go over all files in the dataset
    for name in os.listdir(img_dir):
        if not name.endswith('.jpg'):
            continue
            
        ann_name = name[:-4] + ".png"
        if not os.path.exists(os.path.join(ann_dir, ann_name)):
            print(f"Warning: No annotation found for {name}")
            continue
            
        data.append({
            "image": os.path.join("JPEGImages/480p", sequence, name),
            "annotation": os.path.join("Annotations/480p", sequence, ann_name)
        })
    
    print(f"Found {len(data)} image-annotation pairs for sequence '{sequence}'")
    return data

def create_dataset_splits(data, val_split, seed):
    """Split dataset into training and validation sets"""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split index
    split_idx = int(len(data) * (1 - val_split))
    
    # Split data
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Created splits: {len(train_data)} training, {len(val_data)} validation samples")
    
    return {
        "train": train_data,
        "val": val_data
    }

def save_splits(splits, output_dir):
    """Save dataset splits to disk"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits to JSON files
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name} split to {output_file}")

def main():
    """Main function"""
    args = parse_args()
    
    # Load dataset paths
    data = load_dataset_paths(args.data_dir, args.sequence)
    
    # Create dataset splits
    splits = create_dataset_splits(data, args.val_split, args.seed)
    
    # Save splits to disk
    save_splits(splits, args.output_dir)
    
    print("Dataset preparation completed successfully!")

if __name__ == "__main__":
    main() 