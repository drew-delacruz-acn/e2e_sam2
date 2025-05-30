#!/usr/bin/env python3
"""
Parameter Sweep Script for Contrastive Learning Experiments

This script runs systematic experiments across different parameter combinations:
- 4 initialization methods
- 3 learning rates  
- 3 epochs settings
- 3 margin values
- 3 lambda values

Total: 324 experiments

Usage:
    python parameter_sweep.py --data path/to/data.pkl --output sweep_results/
"""

import argparse
import subprocess
import sys
import json
import csv
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import itertools
import torch
import gc


# Parameter grid configuration
PARAMETER_GRID = {
    'init_method': ['class_means', 'random', 'bounded_random', 'perturbed_means'],
    'lr': [0.01, 0.001, 0.0001],
    'epochs': [50, 100, 150],
    'margin': [0.15, 0.22, 0.3],
    'lambda_push': [0.25, 0.5, 0.75]
}


def clear_gpu_memory():
    """Clear GPU memory to prevent bloat between experiments."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def generate_experiment_combinations():
    """Generate all parameter combinations for experiments."""
    keys = list(PARAMETER_GRID.keys())
    values = list(PARAMETER_GRID.values())
    
    combinations = []
    for combo in itertools.product(*values):
        experiment = dict(zip(keys, combo))
        combinations.append(experiment)
    
    return combinations


def get_experiment_folder_name(params):
    """Generate folder name for experiment based on parameters."""
    name_parts = [
        f"init_{params['init_method']}",
        f"lr_{params['lr']}",
        f"margin_{params['margin']}",
        f"lambda_{params['lambda_push']}",
        f"epochs_{params['epochs']}"
    ]
    return "_".join(name_parts)


def check_experiment_completed(output_dir, params):
    """Check if experiment has already been completed successfully."""
    folder_name = get_experiment_folder_name(params)
    experiment_dir = output_dir / 'experiments' / folder_name
    
    # Check if results.json exists and is valid
    results_file = experiment_dir / 'results.json'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            # Check if experiment completed successfully
            return 'final_f1' in results.get('metrics', {})
        except:
            return False
    return False


def run_single_experiment(data_path, output_dir, params, experiment_num, total_experiments):
    """Run a single experiment with given parameters."""
    
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT {experiment_num}/{total_experiments}")
    print(f"{'='*60}")
    print(f"Parameters: {params}")
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Build command
    cmd = [
        sys.executable, 'train_representatives.py',
        '--data', str(data_path),
        '--output', str(output_dir / 'experiments'),  # Will auto-generate subfolder
        '--init-method', params['init_method'],
        '--lr', str(params['lr']),
        '--margin', str(params['margin']),
        '--lambda-push', str(params['lambda_push']),
        '--epochs', str(params['epochs']),
        '--seed', '42'  # Fixed seed for reproducibility
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # Run experiment
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Experiment completed successfully in {duration:.1f}s")
        
        # Clear GPU memory after completion
        clear_gpu_memory()
        
        return {
            'success': True,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ Experiment failed after {duration:.1f}s")
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")
        
        # Clear GPU memory even after failure
        clear_gpu_memory()
        
        return {
            'success': False,
            'duration': duration,
            'error': str(e),
            'stdout': e.stdout if hasattr(e, 'stdout') else '',
            'stderr': e.stderr if hasattr(e, 'stderr') else ''
        }


def collect_results(output_dir):
    """Collect results from all completed experiments."""
    experiments_dir = output_dir / 'experiments'
    results = []
    
    if not experiments_dir.exists():
        return results
    
    for experiment_folder in experiments_dir.iterdir():
        if experiment_folder.is_dir():
            results_file = experiment_folder / 'results.json'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        result_data = json.load(f)
                    
                    # Extract key metrics
                    metrics = result_data.get('metrics', {})
                    config = result_data.get('config', {})
                    init_method = result_data.get('init_method', 'unknown')
                    
                    result_summary = {
                        'experiment_name': experiment_folder.name,
                        'init_method': init_method,
                        'lr': config.get('lr', 'unknown'),
                        'margin': config.get('margin', 'unknown'),
                        'lambda_push': config.get('lambda_push', 'unknown'),
                        'epochs': config.get('epochs', 'unknown'),
                        'final_f1': metrics.get('final_f1', None),
                        'baseline_f1': metrics.get('baseline_f1', None),
                        'improvement': metrics.get('improvement', None),
                        'final_loss': metrics.get('final_loss', None),
                        'initial_loss': metrics.get('initial_loss', None)
                    }
                    
                    results.append(result_summary)
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not read results from {experiment_folder}: {e}")
    
    return results


def save_results_summary(output_dir, results):
    """Save results summary to CSV and JSON files."""
    
    # Save to CSV
    csv_file = output_dir / 'sweep_results.csv'
    if results:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"📊 Results saved to {csv_file}")
    
    # Save to JSON
    json_file = output_dir / 'sweep_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📊 Results saved to {json_file}")
    
    # Generate summary statistics
    if results:
        # Filter out failed experiments
        successful_results = [r for r in results if r['final_f1'] is not None]
        
        if successful_results:
            # Find best results
            best_f1 = max(successful_results, key=lambda x: x['final_f1'])
            best_improvement = max(successful_results, key=lambda x: x['improvement'] or -999)
            
            # Calculate averages by init method
            init_method_stats = {}
            for result in successful_results:
                method = result['init_method']
                if method not in init_method_stats:
                    init_method_stats[method] = []
                init_method_stats[method].append(result['final_f1'])
            
            # Generate summary
            summary_file = output_dir / 'sweep_summary.txt'
            with open(summary_file, 'w') as f:
                f.write("PARAMETER SWEEP SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total experiments: {len(results)}\n")
                f.write(f"Successful experiments: {len(successful_results)}\n")
                f.write(f"Failed experiments: {len(results) - len(successful_results)}\n\n")
                
                f.write("BEST RESULTS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Best F1 Score: {best_f1['final_f1']:.4f}\n")
                f.write(f"  Experiment: {best_f1['experiment_name']}\n")
                f.write(f"  Parameters: init={best_f1['init_method']}, lr={best_f1['lr']}, "
                       f"margin={best_f1['margin']}, lambda={best_f1['lambda_push']}, "
                       f"epochs={best_f1['epochs']}\n\n")
                
                f.write(f"Best Improvement: {best_improvement['improvement']:.4f}\n")
                f.write(f"  Experiment: {best_improvement['experiment_name']}\n\n")
                
                f.write("AVERAGE F1 BY INITIALIZATION METHOD:\n")
                f.write("-" * 40 + "\n")
                for method, f1_scores in init_method_stats.items():
                    avg_f1 = sum(f1_scores) / len(f1_scores)
                    f.write(f"{method:15}: {avg_f1:.4f} (n={len(f1_scores)})\n")
            
            print(f"📋 Summary saved to {summary_file}")


def estimate_total_time(sample_duration, total_experiments, completed_experiments):
    """Estimate total time remaining based on sample experiment duration."""
    remaining_experiments = total_experiments - completed_experiments
    estimated_remaining = sample_duration * remaining_experiments
    
    return estimated_remaining


def main():
    """Main function to run parameter sweep."""
    parser = argparse.ArgumentParser(description='Run parameter sweep for contrastive learning')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to PKL file containing embeddings')
    parser.add_argument('--output', type=str, default='sweep_results/',
                       help='Output directory for sweep results')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run (skip completed experiments)')
    
    args = parser.parse_args()
    
    # Setup
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Generate all experiment combinations
    experiments = generate_experiment_combinations()
    total_experiments = len(experiments)
    
    print("🧪 PARAMETER SWEEP STARTING")
    print("=" * 50)
    print(f"📊 Data: {data_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🔬 Total experiments: {total_experiments}")
    print(f"📋 Parameter grid:")
    for param, values in PARAMETER_GRID.items():
        print(f"  {param}: {values}")
    print()
    
    # Filter out completed experiments if resuming
    if args.resume:
        print("🔄 Checking for completed experiments...")
        remaining_experiments = []
        completed_count = 0
        
        for exp in experiments:
            if check_experiment_completed(output_dir, exp):
                completed_count += 1
            else:
                remaining_experiments.append(exp)
        
        experiments = remaining_experiments
        print(f"✅ Found {completed_count} completed experiments")
        print(f"🔄 Remaining experiments: {len(experiments)}")
        print()
    
    if not experiments:
        print("✅ All experiments already completed!")
        print("📊 Collecting results...")
        results = collect_results(output_dir)
        save_results_summary(output_dir, results)
        return
    
    # Run experiments
    start_time = time.time()
    completed_experiments = total_experiments - len(experiments)
    
    for i, experiment_params in enumerate(experiments, 1):
        current_exp_num = completed_experiments + i
        
        # Run experiment
        result = run_single_experiment(
            data_path, output_dir, experiment_params, 
            current_exp_num, total_experiments
        )
        
        # Update progress and time estimation
        if i == 1:  # After first experiment, estimate total time
            sample_duration = result['duration']
            estimated_total = estimate_total_time(sample_duration, len(experiments), 1)
            estimated_completion = datetime.now() + timedelta(seconds=estimated_total)
            print(f"⏱️  Estimated completion time: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Estimated total duration: {estimated_total/3600:.1f} hours")
        
        # Show progress
        progress = (current_exp_num / total_experiments) * 100
        print(f"📈 Overall progress: {current_exp_num}/{total_experiments} ({progress:.1f}%)")
        
        # Brief pause between experiments
        time.sleep(1)
    
    # Final results collection and summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🎉 PARAMETER SWEEP COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Total duration: {total_duration/3600:.1f} hours")
    print(f"📊 Collecting and analyzing results...")
    
    # Collect all results
    results = collect_results(output_dir)
    save_results_summary(output_dir, results)
    
    print(f"\n✅ Sweep completed successfully!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"📊 Check sweep_results.csv for detailed comparison")
    print(f"📋 Check sweep_summary.txt for key findings")


if __name__ == '__main__':
    main() 