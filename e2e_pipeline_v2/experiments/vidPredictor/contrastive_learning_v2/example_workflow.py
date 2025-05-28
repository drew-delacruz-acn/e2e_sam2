#!/usr/bin/env python3
"""
Example workflow for hard negative training.

This script demonstrates the complete pipeline:
1. Extract hard negatives from false positives
2. Train with hard negatives
3. Validate the results
"""

import os
import subprocess
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run complete hard negative training workflow')
    parser.add_argument('--false_positives_csv', type=str, required=True,
                       help='Path to false positives CSV file')
    parser.add_argument('--resnet_data', type=str, required=True,
                       help='Path to ResNet embeddings pickle file')
    parser.add_argument('--representatives_data', type=str, required=True,
                       help='Path to initial representatives pickle file')
    parser.add_argument('--output_dir', type=str, default='results/hard_negative_workflow',
                       help='Output directory for all results')
    parser.add_argument('--min_confidence', type=float, default=0.7,
                       help='Minimum confidence for hard negatives')
    parser.add_argument('--max_per_class', type=int, default=30,
                       help='Maximum hard negatives per class')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lambda_hard', type=float, default=0.5,
                       help='Weight for hard negative loss')
    
    return parser.parse_args()

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    args = parse_args()
    
    print("🎯 HARD NEGATIVE TRAINING WORKFLOW")
    print("=" * 60)
    print(f"False positives: {args.false_positives_csv}")
    print(f"ResNet data: {args.resnet_data}")
    print(f"Representatives: {args.representatives_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Max per class: {args.max_per_class}")
    print(f"Epochs: {args.epochs}")
    print(f"Lambda hard: {args.lambda_hard}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    hard_negatives_path = os.path.join(args.output_dir, 'hard_negatives.pkl')
    training_output_dir = os.path.join(args.output_dir, 'training_results')
    validation_output_path = os.path.join(args.output_dir, 'validation_results.csv')
    
    # Step 1: Extract hard negatives
    extract_cmd = [
        'python', 'scripts/extract_hard_negatives.py',
        '--false_positives_csv', args.false_positives_csv,
        '--resnet_data', args.resnet_data,
        '--output_path', hard_negatives_path,
        '--min_confidence', str(args.min_confidence),
        '--max_per_class', str(args.max_per_class)
    ]
    
    if not run_command(extract_cmd, "Step 1: Extracting hard negatives"):
        print("❌ Failed to extract hard negatives. Stopping workflow.")
        return
    
    # Step 2: Train with hard negatives
    train_cmd = [
        'python', 'train_with_hard_negatives.py',
        '--representatives_data', args.representatives_data,
        '--hard_negatives', hard_negatives_path,
        '--output_dir', training_output_dir,
        '--epochs', str(args.epochs),
        '--lambda_hard', str(args.lambda_hard)
    ]
    
    if not run_command(train_cmd, "Step 2: Training with hard negatives"):
        print("❌ Failed to train with hard negatives. Stopping workflow.")
        return
    
    # Step 3: Validate results
    trained_representatives_path = os.path.join(training_output_dir, 'trained_representatives.pkl')
    
    validate_cmd = [
        'python', 'scripts/validate_hard_negatives.py',
        '--representatives_before', args.representatives_data,
        '--representatives_after', trained_representatives_path,
        '--hard_negatives', hard_negatives_path,
        '--output_path', validation_output_path
    ]
    
    if not run_command(validate_cmd, "Step 3: Validating hard negative effectiveness"):
        print("❌ Failed to validate results.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 All results saved to: {args.output_dir}")
    print(f"   Hard negatives: {hard_negatives_path}")
    print(f"   Training results: {training_output_dir}")
    print(f"   Validation results: {validation_output_path}")
    print()
    print("📊 Next steps:")
    print("   1. Review validation results to see if hard negatives were pushed away")
    print("   2. Use trained representatives for improved predictions")
    print("   3. Run results analysis again to see if false positives decreased")

if __name__ == "__main__":
    main() 