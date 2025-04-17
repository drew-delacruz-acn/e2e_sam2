#!/usr/bin/env python
"""
SAM2 Training Pipeline Script
This script runs the entire SAM2 training pipeline:
1. Prepare dataset splits
2. Train the model
"""
import os
import argparse
import subprocess
import sys

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SAM2 training pipeline")
    parser.add_argument("--data_dir", type=str, default="../../../data/davis-2017/DAVIS/",
                      help="Path to DAVIS dataset")
    parser.add_argument("--sequence", type=str, default="bear",
                      help="DAVIS sequence to use")
    parser.add_argument("--val_split", type=float, default=0.2,
                      help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, default="./models",
                      help="Output directory for trained models")
    parser.add_argument("--splits_dir", type=str, default="./dataset_splits",
                      help="Output directory for dataset splits")
    parser.add_argument("--sam2_checkpoint", type=str, default="../checkpoints/sam2.1_hiera_large.pt",
                      help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                      help="Path to model config")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for optimizer")
    parser.add_argument("--max_iterations", type=int, default=100,
                      help="Number of training iterations")
    parser.add_argument("--val_interval", type=int, default=10,
                      help="Validate model every N iterations")
    parser.add_argument("--save_interval", type=int, default=50,
                      help="Save model every N iterations")
    return parser.parse_args()

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'-'*80}\n{description}\n{'-'*80}")
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Main function to run the pipeline"""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)
    
    # 1. Prepare dataset splits
    prepare_cmd = [
        sys.executable, "prepare_dataset.py",
        "--data_dir", args.data_dir,
        "--sequence", args.sequence,
        "--val_split", str(args.val_split),
        "--output_dir", args.splits_dir
    ]
    
    if not run_command(prepare_cmd, "Preparing dataset splits"):
        print("Failed to prepare dataset splits. Exiting.")
        return
    
    # 2. Train the model
    output_model = os.path.join(args.output_dir, f"sam2_finetuned_{args.sequence}.torch")
    train_cmd = [
        sys.executable, "train_sam.py",
        "--data_dir", args.data_dir,
        "--splits_dir", args.splits_dir,
        "--sam2_checkpoint", args.sam2_checkpoint,
        "--model_cfg", args.model_cfg,
        "--learning_rate", str(args.learning_rate),
        "--max_iterations", str(args.max_iterations),
        "--val_interval", str(args.val_interval),
        "--save_interval", str(args.save_interval),
        "--output_model", output_model
    ]
    
    if not run_command(train_cmd, "Training SAM2 model"):
        print("Failed to train the model. Exiting.")
        return
    
    print("\n" + "="*80)
    print(f"🎉 SAM2 training pipeline completed successfully!")
    print(f"Trained model saved to: {output_model}")
    print(f"Best model saved to: {output_model.replace('.torch', '_best.torch')}")
    print("="*80)

if __name__ == "__main__":
    main() 