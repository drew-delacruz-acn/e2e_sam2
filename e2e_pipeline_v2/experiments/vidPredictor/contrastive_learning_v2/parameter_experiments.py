#!/usr/bin/env python3
"""
Parameter experiment script for contrastive learning optimization.

This script runs multiple experiments with different parameter combinations
to find optimal settings for your dataset.
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time

def run_experiment(data_path, output_dir, params, experiment_name):
    """Run a single experiment with given parameters."""
    
    print(f"\n🧪 Running experiment: {experiment_name}")
    print(f"📋 Parameters: {params}")
    
    # Create output directory for this experiment
    exp_output = Path(output_dir) / experiment_name
    exp_output.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "train_representatives.py",
        "--data", str(data_path),
        "--output", str(exp_output),
        "--lr", str(params['lr']),
        "--margin", str(params['margin']),
        "--lambda-push", str(params['lambda_push']),
        "--epochs", str(params['epochs']),
        "--val-frac", str(params.get('val_frac', 0.3)),
        "--seed", str(params.get('seed', 42))
    ]
    
    try:
        # Run the experiment
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        end_time = time.time()
        
        if result.returncode == 0:
            # Load results
            results_file = exp_output / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract key metrics
                metrics = {
                    'experiment': experiment_name,
                    'parameters': params,
                    'final_f1': results['metrics'].get('final_f1'),
                    'baseline_f1': results['metrics'].get('baseline_f1'),
                    'improvement': results['metrics'].get('improvement'),
                    'final_loss': results['metrics'].get('final_loss'),
                    'initial_loss': results['metrics'].get('initial_loss'),
                    'runtime_seconds': end_time - start_time,
                    'success': True
                }
                
                print(f"✅ Success! F1: {metrics['final_f1']:.4f}, Improvement: {metrics['improvement']:+.4f}")
                return metrics
            else:
                print(f"❌ Results file not found")
                return {'experiment': experiment_name, 'success': False, 'error': 'No results file'}
        else:
            print(f"❌ Command failed: {result.stderr}")
            return {'experiment': experiment_name, 'success': False, 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out")
        return {'experiment': experiment_name, 'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'experiment': experiment_name, 'success': False, 'error': str(e)}

def define_experiments():
    """Define parameter combinations to test."""
    
    experiments = {
        # Baseline (current settings)
        "baseline": {
            'lr': 0.01, 'margin': 0.15, 'lambda_push': 0.25, 'epochs': 50
        },
        
        # Stricter separation experiments
        "strict_separation_1": {
            'lr': 0.01, 'margin': 0.25, 'lambda_push': 0.4, 'epochs': 50
        },
        "strict_separation_2": {
            'lr': 0.01, 'margin': 0.3, 'lambda_push': 0.5, 'epochs': 75
        },
        
        # Higher learning rate experiments
        "faster_learning_1": {
            'lr': 0.02, 'margin': 0.2, 'lambda_push': 0.3, 'epochs': 50
        },
        "faster_learning_2": {
            'lr': 0.03, 'margin': 0.15, 'lambda_push': 0.25, 'epochs': 40
        },
        
        # Lower learning rate (more stable)
        "stable_learning_1": {
            'lr': 0.005, 'margin': 0.2, 'lambda_push': 0.3, 'epochs': 100
        },
        "stable_learning_2": {
            'lr': 0.003, 'margin': 0.25, 'lambda_push': 0.4, 'epochs': 150
        },
        
        # Focus on push term
        "strong_push_1": {
            'lr': 0.01, 'margin': 0.2, 'lambda_push': 0.6, 'epochs': 60
        },
        "strong_push_2": {
            'lr': 0.015, 'margin': 0.3, 'lambda_push': 0.8, 'epochs': 80
        },
        
        # Balanced approach
        "balanced_1": {
            'lr': 0.012, 'margin': 0.18, 'lambda_push': 0.35, 'epochs': 60
        },
        "balanced_2": {
            'lr': 0.008, 'margin': 0.22, 'lambda_push': 0.32, 'epochs': 80
        },
        
        # Conservative approach
        "conservative": {
            'lr': 0.005, 'margin': 0.12, 'lambda_push': 0.15, 'epochs': 100
        },
        
        # Aggressive approach
        "aggressive": {
            'lr': 0.025, 'margin': 0.35, 'lambda_push': 0.7, 'epochs': 100
        }
    }
    
    return experiments

def analyze_results(results_list):
    """Analyze and rank experiment results."""
    
    # Filter successful experiments
    successful = [r for r in results_list if r.get('success', False)]
    
    if not successful:
        print("❌ No successful experiments!")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(successful)
    
    # Sort by improvement (or F1 if improvement is None)
    df['sort_metric'] = df['improvement'].fillna(df['final_f1'])
    df_sorted = df.sort_values('sort_metric', ascending=False)
    
    print("\n" + "="*80)
    print("📊 EXPERIMENT RESULTS ANALYSIS")
    print("="*80)
    
    print(f"\n🏆 TOP 5 EXPERIMENTS:")
    for i, (_, row) in enumerate(df_sorted.head().iterrows()):
        print(f"\n{i+1}. {row['experiment']}")
        print(f"   F1 Score: {row['final_f1']:.4f}")
        print(f"   Improvement: {row['improvement']:+.4f}")
        print(f"   Final Loss: {row['final_loss']:.4f}")
        print(f"   Parameters: lr={row['parameters']['lr']}, margin={row['parameters']['margin']}, λ={row['parameters']['lambda_push']}")
        print(f"   Runtime: {row['runtime_seconds']:.1f}s")
    
    # Parameter analysis
    print(f"\n📈 PARAMETER ANALYSIS:")
    
    # Best parameters
    best_exp = df_sorted.iloc[0]
    best_params = best_exp['parameters']
    print(f"\n🎯 Best Parameter Combination:")
    print(f"   Learning Rate: {best_params['lr']}")
    print(f"   Margin: {best_params['margin']}")
    print(f"   Lambda Push: {best_params['lambda_push']}")
    print(f"   Epochs: {best_params['epochs']}")
    
    # Parameter correlations
    param_df = pd.DataFrame([r['parameters'] for r in successful])
    param_df['improvement'] = [r['improvement'] for r in successful]
    
    print(f"\n📊 Parameter Correlations with Improvement:")
    for param in ['lr', 'margin', 'lambda_push']:
        corr = param_df[param].corr(param_df['improvement'])
        print(f"   {param}: {corr:+.3f}")
    
    # Save detailed results
    results_file = "experiment_results.csv"
    df_sorted.to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return df_sorted

def main():
    """Main experiment runner."""
    
    print("🧪 Contrastive Learning Parameter Optimization")
    print("=" * 60)
    
    # Configuration
    data_path = "definitiveObjects_jeremiah.pkl"  # Update this path
    output_dir = "parameter_experiments"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path in the script.")
        return
    
    # Get experiments
    experiments = define_experiments()
    
    print(f"📋 Running {len(experiments)} experiments...")
    print(f"📁 Data: {data_path}")
    print(f"📊 Output: {output_dir}")
    
    # Run experiments
    results = []
    for exp_name, params in experiments.items():
        result = run_experiment(data_path, output_dir, params, exp_name)
        results.append(result)
        
        # Brief pause between experiments
        time.sleep(2)
    
    # Analyze results
    analyze_results(results)
    
    print(f"\n🎉 Parameter optimization completed!")
    print(f"📁 All results saved in: {output_dir}/")

if __name__ == '__main__':
    main() 