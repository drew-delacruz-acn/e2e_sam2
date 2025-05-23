#!/usr/bin/env python3
"""
Test script to verify parameter combinations are generated correctly.
"""

import itertools

# Parameter grid configuration (same as in parameter_sweep.py)
PARAMETER_GRID = {
    'init_method': ['class_means', 'random', 'bounded_random', 'perturbed_means'],
    'lr': [0.01, 0.001, 0.0001],
    'epochs': [50, 100, 150],
    'margin': [0.15, 0.22, 0.3],
    'lambda_push': [0.25, 0.5, 0.75]
}

def generate_experiment_combinations():
    """Generate all parameter combinations for experiments."""
    keys = list(PARAMETER_GRID.keys())
    values = list(PARAMETER_GRID.values())
    
    combinations = []
    for combo in itertools.product(*values):
        experiment = dict(zip(keys, combo))
        combinations.append(experiment)
    
    return combinations

def main():
    print("🧪 PARAMETER COMBINATION TEST")
    print("=" * 40)
    
    # Show parameter grid
    print("📋 Parameter Grid:")
    total_combinations = 1
    for param, values in PARAMETER_GRID.items():
        print(f"  {param:12}: {values} ({len(values)} options)")
        total_combinations *= len(values)
    
    print(f"\n🔢 Expected total combinations: {total_combinations}")
    
    # Generate combinations
    combinations = generate_experiment_combinations()
    print(f"🔢 Generated combinations: {len(combinations)}")
    
    # Show first few examples
    print(f"\n📝 First 5 combinations:")
    for i, combo in enumerate(combinations[:5], 1):
        print(f"  {i}. {combo}")
    
    print(f"\n📝 Last 5 combinations:")
    for i, combo in enumerate(combinations[-5:], len(combinations)-4):
        print(f"  {i}. {combo}")
    
    # Verify uniqueness
    unique_combinations = set()
    for combo in combinations:
        combo_tuple = tuple(sorted(combo.items()))
        unique_combinations.add(combo_tuple)
    
    print(f"\n✅ All combinations are unique: {len(unique_combinations) == len(combinations)}")
    
    # Show breakdown by parameter
    print(f"\n📊 Breakdown by parameter:")
    for param in PARAMETER_GRID.keys():
        values = [combo[param] for combo in combinations]
        unique_values = set(values)
        counts = {val: values.count(val) for val in unique_values}
        print(f"  {param:12}: {counts}")

if __name__ == '__main__':
    main() 