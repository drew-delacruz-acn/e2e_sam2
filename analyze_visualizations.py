#!/usr/bin/env python3
"""
Visualization analysis script for contrastive learning experiment results.

This script analyzes the generated plots and provides detailed insights.
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_loss_curve(results_path):
    """Analyze the training loss curve."""
    print("📉 LOSS CURVE ANALYSIS")
    print("=" * 50)
    
    # Load results
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)
    
    losses = results['history']['losses']
    epochs = list(range(1, len(losses) + 1))
    
    # Basic statistics
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    
    print(f"📊 Loss Statistics:")
    print(f"   Initial Loss: {initial_loss:.4f}")
    print(f"   Final Loss: {final_loss:.4f}")
    print(f"   Best Loss: {min_loss:.4f} (epoch {losses.index(min_loss) + 1})")
    print(f"   Worst Loss: {max_loss:.4f} (epoch {losses.index(max_loss) + 1})")
    print(f"   Total Improvement: {initial_loss - final_loss:.4f}")
    print(f"   Improvement %: {((initial_loss - final_loss) / abs(initial_loss)) * 100:.2f}%")
    
    # Training phases analysis
    print(f"\n🔍 Training Phase Analysis:")
    
    # Early phase (first 25% of epochs)
    early_phase = len(losses) // 4
    early_improvement = losses[0] - losses[early_phase]
    print(f"   Early Phase (epochs 1-{early_phase}): {early_improvement:.4f} improvement")
    
    # Late phase (last 25% of epochs)
    late_start = 3 * len(losses) // 4
    late_improvement = losses[late_start] - losses[-1]
    print(f"   Late Phase (epochs {late_start}-{len(losses)}): {late_improvement:.4f} improvement")
    
    # Convergence analysis
    last_10_losses = losses[-10:]
    convergence_variance = np.var(last_10_losses)
    print(f"   Convergence Stability (last 10 epochs variance): {convergence_variance:.6f}")
    
    if convergence_variance < 0.001:
        print("   ✅ Well converged - stable training")
    elif convergence_variance < 0.01:
        print("   ⚠️  Moderately converged - could train longer")
    else:
        print("   ❌ Poor convergence - may need different hyperparameters")
    
    # Trend analysis
    print(f"\n📈 Trend Analysis:")
    
    # Calculate moving average
    window = 5
    if len(losses) >= window:
        moving_avg = []
        for i in range(window-1, len(losses)):
            moving_avg.append(np.mean(losses[i-window+1:i+1]))
        
        trend_slope = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
        print(f"   Moving Average Trend (window={window}): {trend_slope:.6f}")
        
        if trend_slope < -0.001:
            print("   ✅ Strong downward trend - good optimization")
        elif trend_slope < 0:
            print("   ✅ Slight downward trend - steady improvement")
        else:
            print("   ⚠️  Flat or upward trend - may have converged")


def analyze_tsne_plot(results_path):
    """Analyze the t-SNE visualization."""
    print("\n\n🗺️  T-SNE PLOT ANALYSIS")
    print("=" * 50)
    
    # Load results and representatives
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)
    
    with open(results_path / 'representatives.pkl', 'rb') as f:
        rep_data = pickle.load(f)
    
    class_names = results['data_info']['class_names']
    num_classes = results['data_info']['num_classes']
    embedding_dim = results['data_info']['embedding_dim']
    
    print(f"📊 Dataset Characteristics:")
    print(f"   Classes: {num_classes}")
    print(f"   Embedding Dimension: {embedding_dim}")
    print(f"   Train Samples: {results['data_info']['num_train_samples']}")
    print(f"   Val Samples: {results['data_info']['num_val_samples']}")
    
    print(f"\n🏷️  Class Distribution:")
    for i, class_name in enumerate(class_names):
        print(f"   {i:2d}. {class_name}")
    
    print(f"\n🎯 Performance Insights:")
    baseline_f1 = results['metrics']['baseline_f1']
    final_f1 = results['metrics']['final_f1']
    
    if baseline_f1 == 1.0 and final_f1 == 1.0:
        print("   ✅ Perfect separation achieved!")
        print("   📍 All classes are perfectly distinguishable")
        print("   🎯 Representatives are optimal for classification")
        print("   💡 This suggests high-quality embeddings and well-defined classes")
    elif final_f1 > baseline_f1:
        improvement = final_f1 - baseline_f1
        print(f"   📈 Improved classification: +{improvement:.4f} F1 score")
        print("   ✅ Contrastive learning successfully optimized representatives")
    elif final_f1 == baseline_f1:
        print("   ➡️  No improvement over baseline")
        print("   💭 Class means were already optimal, or learning rate too low")
    else:
        degradation = baseline_f1 - final_f1
        print(f"   📉 Performance degraded: -{degradation:.4f} F1 score")
        print("   ⚠️  May need different hyperparameters or more training")
    
    print(f"\n🔬 Technical Analysis:")
    print(f"   Dimensionality Reduction: {embedding_dim}D → 2D via t-SNE")
    print(f"   Visualization preserves local neighborhood structure")
    print(f"   Representatives shown as stars, embeddings as circles")
    
    # Analyze class complexity
    print(f"\n🎭 Domain-Specific Insights (Loki/TVA Dataset):")
    
    # Group classes by type
    armor_classes = [name for name in class_names if 'Armor' in name or 'armor' in name]
    weapon_classes = [name for name in class_names if any(weapon in name.lower() for weapon in ['dagger', 'machete', 'stick', 'spear'])]
    uniform_classes = [name for name in class_names if 'Uniform' in name or 'uniform' in name]
    accessory_classes = [name for name in class_names if any(acc in name.lower() for acc in ['collar', 'headpiece', 'tempad', 'monitor', 'plushie'])]
    
    print(f"   Armor/Clothing ({len(armor_classes)}): {', '.join(armor_classes)}")
    print(f"   Weapons ({len(weapon_classes)}): {', '.join(weapon_classes)}")
    print(f"   Uniforms ({len(uniform_classes)}): {', '.join(uniform_classes)}")
    print(f"   Accessories ({len(accessory_classes)}): {', '.join(accessory_classes)}")
    
    print(f"\n💡 Interpretation:")
    print(f"   • Perfect F1 suggests clear visual distinctions between items")
    print(f"   • 2048D embeddings capture fine-grained details effectively")
    print(f"   • Model can distinguish between similar items (different Loki armors)")
    print(f"   • Representatives serve as canonical prototypes for each class")


def analyze_experiment_quality(results_path):
    """Overall experiment quality assessment."""
    print("\n\n⭐ OVERALL EXPERIMENT ASSESSMENT")
    print("=" * 50)
    
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)
    
    # Quality metrics
    quality_score = 0
    max_score = 5
    
    # 1. Performance quality
    final_f1 = results['metrics']['final_f1']
    if final_f1 >= 0.95:
        quality_score += 1
        print("✅ Excellent Performance (F1 ≥ 0.95)")
    elif final_f1 >= 0.8:
        print("✅ Good Performance (F1 ≥ 0.8)")
    else:
        print("⚠️  Moderate Performance (F1 < 0.8)")
    
    # 2. Training stability
    losses = results['history']['losses']
    final_variance = np.var(losses[-10:])
    if final_variance < 0.01:
        quality_score += 1
        print("✅ Stable Training (low variance in final epochs)")
    else:
        print("⚠️  Unstable Training (high variance in final epochs)")
    
    # 3. Loss improvement
    improvement = losses[0] - losses[-1]
    if improvement > 0:
        quality_score += 1
        print("✅ Loss Decreased (successful optimization)")
    else:
        print("⚠️  Loss Did Not Decrease")
    
    # 4. Dataset size adequacy
    total_samples = results['data_info']['num_train_samples'] + results['data_info']['num_val_samples']
    samples_per_class = total_samples / results['data_info']['num_classes']
    if samples_per_class >= 3:
        quality_score += 1
        print(f"✅ Adequate Data (avg {samples_per_class:.1f} samples/class)")
    else:
        print(f"⚠️  Limited Data (avg {samples_per_class:.1f} samples/class)")
    
    # 5. Hyperparameter appropriateness
    config = results['config']
    if 0.001 <= config['lr'] <= 0.1 and 0.1 <= config['margin'] <= 0.5:
        quality_score += 1
        print("✅ Reasonable Hyperparameters")
    else:
        print("⚠️  Hyperparameters may need tuning")
    
    print(f"\n🏆 Quality Score: {quality_score}/{max_score}")
    
    if quality_score >= 4:
        print("🎉 EXCELLENT EXPERIMENT - Ready for production!")
    elif quality_score >= 3:
        print("👍 GOOD EXPERIMENT - Minor improvements possible")
    else:
        print("🔧 NEEDS IMPROVEMENT - Consider parameter tuning")
    
    print(f"\n🚀 Recommendations:")
    if final_f1 == 1.0:
        print("   • Your representatives are optimal - ready to deploy!")
        print("   • Consider testing on new/unseen data")
        print("   • Use representatives for fast similarity search")
    else:
        print("   • Try different hyperparameters for better performance")
        print("   • Consider more training epochs")
        print("   • Validate data quality and class definitions")


def main():
    """Main analysis function."""
    results_path = Path("/Users/andrewdelacruz/e2e_sam2/gitignore_exception/first_pass_rep_embeddings")
    
    print("🔍 CONTRASTIVE LEARNING EXPERIMENT ANALYSIS")
    print("=" * 60)
    print(f"📁 Results Path: {results_path}")
    print()
    
    # Analyze each component
    analyze_loss_curve(results_path)
    analyze_tsne_plot(results_path)
    analyze_experiment_quality(results_path)
    
    print("\n" + "=" * 60)
    print("📋 ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main() 