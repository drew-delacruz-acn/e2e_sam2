#!/usr/bin/env python3
"""
Example script showing how to use random initialization instead of class means.

This demonstrates the new initialization methods and compares their performance
on your actual dataset.
"""

import subprocess
import sys
from pathlib import Path


def run_experiment(data_path, init_method, epochs=60, use_auto_naming=True):
    """Run contrastive learning experiment with specified initialization method."""
    
    cmd = [
        sys.executable, 'train_representatives.py',
        '--data', str(data_path),
        '--init-method', init_method,
        '--epochs', str(epochs),
        '--lr', '0.01',
        '--margin', '0.22',
        '--lambda-push', '0.35'
    ]
    
    # Add manual output directory if not using auto-naming
    if not use_auto_naming:
        output_dir = f"results/manual_init_{init_method}"
        cmd.extend(['--output', output_dir, '--no-auto-name'])
    
    print(f"🚀 Running experiment with {init_method} initialization...")
    if use_auto_naming:
        print(f"📁 Output: Auto-generated based on parameters")
    else:
        print(f"📁 Output: {output_dir}")
    print(f"⚙️  Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Experiment completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Run comparison experiments with different initialization methods."""
    
    # Paths (adjust these to your actual data)
    data_path = Path("../../../data/representatives.pkl")  # Adjust path as needed
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please adjust the data_path in this script to point to your PKL file.")
        return
    
    # Initialization methods to test
    methods = [
        'class_means',      # Original method
        'random',           # Pure random
        'bounded_random',   # Random within data bounds
        'perturbed_means'   # Class means + noise
    ]
    
    print("🧪 Running Initialization Method Comparison")
    print("=" * 50)
    print(f"📊 Data: {data_path}")
    print(f"📁 Output: Auto-generated folders based on parameters")
    print(f"🔬 Methods: {', '.join(methods)}")
    print()
    
    # Run experiments
    results = {}
    for method in methods:
        success = run_experiment(data_path, method, epochs=60, use_auto_naming=True)
        results[method] = success
        print()
    
    # Summary
    print("=" * 50)
    print("📋 EXPERIMENT SUMMARY")
    print("=" * 50)
    
    for method, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{method:15} : {status}")
    
    print()
    print("📊 Results saved in auto-generated directories under results/")
    print("   Directory names include all parameters for easy identification:")
    print("   Format: init_{method}_lr_{lr}_margin_{margin}_lambda_{lambda}_epochs_{epochs}")
    
    print()
    print("💡 Tips for analysis:")
    print("  - Compare final F1 scores in results.json files")
    print("  - Look at loss curves in loss_curve.png files")
    print("  - Check t-SNE plots for representative quality")
    print("  - Random init may need more epochs to converge")
    print("  - Use 'ls results/' to see all generated experiment folders")


if __name__ == '__main__':
    main() 