#!/usr/bin/env python
"""
SAM2 Dataset Preparation Script
This script prepares training and validation splits for SAM2 model fine-tuning.
"""
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_dataset_paths, create_dataset_splits, save_splits

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare dataset for SAM2 fine-tuning")
    parser.add_argument("--data_dir", type=str, default="../data/davis-2017/DAVIS/",
                      help="Path to DAVIS dataset")
    parser.add_argument("--sequence", type=str, default="bear",
                      help="DAVIS sequence to use")
    parser.add_argument("--val_split", type=float, default=0.2,
                      help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="../dataset_splits",
                      help="Output directory for dataset splits")
    return parser.parse_args()

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