#!/usr/bin/env python3
"""
Test script to demonstrate automatic experiment folder naming.

This script shows how the new automatic naming feature works and what
folder names are generated for different parameter combinations.
"""

import sys
from pathlib import Path

# Add the current directory to path to import the function
sys.path.append('.')
from train_representatives import generate_experiment_name


class MockArgs:
    """Mock arguments class for testing."""
    def __init__(self, **kwargs):
        # Set defaults
        self.init_method = kwargs.get('init_method', 'class_means')
        self.lr = kwargs.get('lr', 0.01)
        self.margin = kwargs.get('margin', 0.15)
        self.lambda_push = kwargs.get('lambda_push', 0.25)
        self.epochs = kwargs.get('epochs', 50)
        self.seed = kwargs.get('seed', 42)


def test_naming_examples():
    """Test various parameter combinations and show generated names."""
    
    print("🧪 Testing Automatic Experiment Folder Naming")
    print("=" * 60)
    print()
    
    # Test cases with different parameter combinations
    test_cases = [
        {
            'name': 'Default Parameters',
            'params': {}
        },
        {
            'name': 'Random Initialization',
            'params': {'init_method': 'random', 'epochs': 150}
        },
        {
            'name': 'Aggressive Parameters',
            'params': {'margin': 0.3, 'lambda_push': 0.6, 'epochs': 100}
        },
        {
            'name': 'Balanced Parameters',
            'params': {'margin': 0.22, 'lambda_push': 0.35, 'epochs': 60}
        },
        {
            'name': 'High Learning Rate',
            'params': {'lr': 0.015, 'init_method': 'bounded_random'}
        },
        {
            'name': 'Custom Seed',
            'params': {'seed': 123, 'init_method': 'perturbed_means'}
        },
        {
            'name': 'Long Training',
            'params': {'epochs': 200, 'lr': 0.005}
        }
    ]
    
    print("📋 Generated Folder Names:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        args = MockArgs(**test_case['params'])
        folder_name = generate_experiment_name(args)
        
        print(f"{i}. {test_case['name']}")
        print(f"   Parameters: {test_case['params']}")
        print(f"   Folder: {folder_name}")
        print()
    
    print("=" * 60)
    print("💡 NAMING CONVENTION")
    print("=" * 60)
    print("Format: init_{method}_lr_{lr}_margin_{margin}_lambda_{lambda}_epochs_{epochs}")
    print("- Only non-default seeds are included in the name")
    print("- All parameters are included for complete identification")
    print("- Underscores separate different parameter types")
    print("- Decimal points in floats are preserved")
    print()
    
    print("🎯 BENEFITS")
    print("-" * 20)
    print("✅ Easy to identify experiment parameters from folder name")
    print("✅ No accidental overwrites of different experiments")
    print("✅ Automatic organization of results")
    print("✅ Can still use custom output paths when needed")
    print()
    
    print("📁 EXAMPLE USAGE")
    print("-" * 20)
    print("# Auto-naming (default)")
    print("python train_representatives.py --data data.pkl")
    print("# → results/init_class_means_lr_0.01_margin_0.15_lambda_0.25_epochs_50/")
    print()
    print("# Auto-naming with different parameters")
    print("python train_representatives.py --data data.pkl --init-method random --epochs 150")
    print("# → results/init_random_lr_0.01_margin_0.15_lambda_0.25_epochs_150/")
    print()
    print("# Manual output path (disables auto-naming)")
    print("python train_representatives.py --data data.pkl --output my_experiment/")
    print("# → my_experiment/")
    print()
    print("# Disable auto-naming but use default results folder")
    print("python train_representatives.py --data data.pkl --no-auto-name")
    print("# → results/")


def test_path_creation():
    """Test that the paths would be valid."""
    print("\n🔍 TESTING PATH VALIDITY")
    print("-" * 30)
    
    # Test a complex case
    args = MockArgs(
        init_method='bounded_random',
        lr=0.0123,
        margin=0.456,
        lambda_push=0.789,
        epochs=999,
        seed=12345
    )
    
    folder_name = generate_experiment_name(args)
    full_path = Path('results') / folder_name
    
    print(f"Complex example:")
    print(f"  Folder name: {folder_name}")
    print(f"  Full path: {full_path}")
    print(f"  Path length: {len(str(full_path))} characters")
    print(f"  Valid path: {'✅' if len(str(full_path)) < 255 else '❌'}")


if __name__ == '__main__':
    test_naming_examples()
    test_path_creation() 