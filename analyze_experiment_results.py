import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_experiment_results():
    """Load results from all three experiments"""
    experiments = {
        'quick_test_agg': 'gitignore_exception/org/quick_test_agg/results.json',
        'quick_test_balanced': 'gitignore_exception/org/quick_test_balanced/results.json', 
        'very_agg': 'gitignore_exception/org/very_agg/results.json'
    }
    
    results = {}
    for name, path in experiments.items():
        with open(path, 'r') as f:
            results[name] = json.load(f)
    
    return results

def analyze_convergence(results):
    """Analyze convergence characteristics of each experiment"""
    analysis = {}
    
    for name, data in results.items():
        losses = data['history']['losses']
        config = data['config']
        
        # Calculate convergence metrics
        initial_loss = losses[0]
        final_loss = losses[-1]
        total_improvement = final_loss - initial_loss
        
        # Calculate convergence rate (improvement per epoch)
        convergence_rate = total_improvement / len(losses)
        
        # Calculate stability (variance in last 10 epochs)
        last_10_losses = losses[-10:]
        stability = np.std(last_10_losses)
        
        # Calculate efficiency (improvement per epoch)
        efficiency = abs(total_improvement) / config['epochs']
        
        analysis[name] = {
            'config': config,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'total_improvement': total_improvement,
            'convergence_rate': convergence_rate,
            'stability': stability,
            'efficiency': efficiency,
            'epochs': config['epochs'],
            'final_f1': data['metrics']['final_f1']
        }
    
    return analysis

def create_comparison_table(analysis):
    """Create a comparison table of all experiments"""
    df_data = []
    
    for name, metrics in analysis.items():
        df_data.append({
            'Experiment': name,
            'Margin': metrics['config']['margin'],
            'Lambda': metrics['config']['lambda_push'],
            'LR': metrics['config']['lr'],
            'Epochs': metrics['epochs'],
            'Initial Loss': f"{metrics['initial_loss']:.3f}",
            'Final Loss': f"{metrics['final_loss']:.3f}",
            'Total Improvement': f"{metrics['total_improvement']:.3f}",
            'Convergence Rate': f"{metrics['convergence_rate']:.4f}",
            'Stability (σ)': f"{metrics['stability']:.4f}",
            'Efficiency': f"{metrics['efficiency']:.4f}",
            'Final F1': metrics['final_f1']
        })
    
    return pd.DataFrame(df_data)

def plot_loss_curves(results):
    """Plot loss curves for all experiments"""
    plt.figure(figsize=(12, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (name, data) in enumerate(results.items()):
        losses = data['history']['losses']
        config = data['config']
        
        label = f"{name}\n(margin={config['margin']}, λ={config['lambda_push']})"
        plt.plot(losses, color=colors[i], linewidth=2, label=label)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def determine_best_experiment(analysis):
    """Determine which experiment performed best based on multiple criteria"""
    
    print("🎯 EXPERIMENT PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Rank by different criteria
    rankings = {}
    
    # 1. Final Loss (lower is better)
    final_losses = {name: metrics['final_loss'] for name, metrics in analysis.items()}
    rankings['final_loss'] = sorted(final_losses.items(), key=lambda x: x[1])
    
    # 2. Total Improvement (more negative is better for loss)
    improvements = {name: metrics['total_improvement'] for name, metrics in analysis.items()}
    rankings['improvement'] = sorted(improvements.items(), key=lambda x: x[1])
    
    # 3. Efficiency (improvement per epoch)
    efficiencies = {name: metrics['efficiency'] for name, metrics in analysis.items()}
    rankings['efficiency'] = sorted(efficiencies.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Stability (lower variance is better)
    stabilities = {name: metrics['stability'] for name, metrics in analysis.items()}
    rankings['stability'] = sorted(stabilities.items(), key=lambda x: x[1])
    
    print("\n📊 RANKINGS BY CRITERIA:")
    print("-" * 30)
    
    for criterion, ranking in rankings.items():
        print(f"\n{criterion.upper()}:")
        for i, (name, value) in enumerate(ranking, 1):
            print(f"  {i}. {name}: {value:.4f}")
    
    # Calculate overall score (lower rank number is better)
    scores = {}
    for name in analysis.keys():
        score = 0
        for criterion, ranking in rankings.items():
            for i, (exp_name, _) in enumerate(ranking, 1):
                if exp_name == name:
                    score += i
                    break
        scores[name] = score
    
    # Best overall (lowest total rank score)
    best_overall = min(scores.items(), key=lambda x: x[1])
    
    print(f"\n🏆 OVERALL WINNER: {best_overall[0]}")
    print(f"   Total Rank Score: {best_overall[1]} (lower is better)")
    
    return best_overall[0], rankings, scores

def main():
    print("Loading experiment results...")
    results = load_experiment_results()
    
    print("Analyzing convergence characteristics...")
    analysis = analyze_convergence(results)
    
    print("Creating comparison table...")
    comparison_df = create_comparison_table(analysis)
    print("\n📋 DETAILED COMPARISON TABLE:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    print("\nPlotting loss curves...")
    plot_loss_curves(results)
    
    print("\nDetermining best experiment...")
    best_experiment, rankings, scores = determine_best_experiment(analysis)
    
    # Detailed analysis of the winner
    winner_data = analysis[best_experiment]
    print(f"\n🎯 DETAILED ANALYSIS OF WINNER: {best_experiment}")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  - Margin: {winner_data['config']['margin']}")
    print(f"  - Lambda: {winner_data['config']['lambda_push']}")
    print(f"  - Learning Rate: {winner_data['config']['lr']}")
    print(f"  - Epochs: {winner_data['epochs']}")
    print(f"\nPerformance:")
    print(f"  - Final Loss: {winner_data['final_loss']:.4f}")
    print(f"  - Total Improvement: {winner_data['total_improvement']:.4f}")
    print(f"  - Convergence Rate: {winner_data['convergence_rate']:.4f}")
    print(f"  - Stability: {winner_data['stability']:.4f}")
    print(f"  - Efficiency: {winner_data['efficiency']:.4f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    if best_experiment == 'very_agg':
        print("✅ The very aggressive approach worked best!")
        print("   - High margin (0.4) and lambda (0.8) achieved strongest separation")
        print("   - More epochs (100) allowed for better convergence")
        print("   - Consider this configuration for final training")
    elif best_experiment == 'quick_test_agg':
        print("✅ The aggressive approach was most efficient!")
        print("   - Good balance of performance and training time")
        print("   - Consider extending epochs for even better results")
    else:
        print("✅ The balanced approach provided stable convergence!")
        print("   - More conservative but reliable optimization")
        print("   - Good starting point for further tuning")
    
    return best_experiment, analysis

if __name__ == "__main__":
    best_exp, analysis = main() 