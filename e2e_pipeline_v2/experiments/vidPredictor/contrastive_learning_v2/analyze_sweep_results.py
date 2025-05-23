#!/usr/bin/env python3
"""
Analysis script for parameter sweep results.

This script provides additional analysis and visualization of the parameter sweep results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import json


def load_results(results_dir):
    """Load results from CSV file."""
    csv_file = results_dir / 'sweep_results.csv'
    if not csv_file.exists():
        print(f"❌ Results file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} experiment results")
    return df


def analyze_parameter_importance(df):
    """Analyze which parameters have the most impact on performance."""
    print("\n📈 PARAMETER IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    # Filter successful experiments
    successful_df = df[df['final_f1'].notna()].copy()
    
    if len(successful_df) == 0:
        print("❌ No successful experiments found")
        return
    
    print(f"Analyzing {len(successful_df)} successful experiments")
    
    # Analyze each parameter
    parameters = ['init_method', 'lr', 'margin', 'lambda_push', 'epochs']
    
    for param in parameters:
        print(f"\n🔍 {param.upper()}:")
        param_stats = successful_df.groupby(param)['final_f1'].agg(['mean', 'std', 'count'])
        param_stats = param_stats.sort_values('mean', ascending=False)
        
        for value, stats in param_stats.iterrows():
            print(f"  {value:15}: F1={stats['mean']:.4f} ±{stats['std']:.4f} (n={stats['count']})")


def find_best_combinations(df, top_n=10):
    """Find the best parameter combinations."""
    print(f"\n🏆 TOP {top_n} BEST RESULTS")
    print("=" * 50)
    
    # Filter and sort by F1 score
    successful_df = df[df['final_f1'].notna()].copy()
    top_results = successful_df.nlargest(top_n, 'final_f1')
    
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i:2d}. F1={row['final_f1']:.4f} | "
              f"init={row['init_method']:15} | lr={row['lr']:7} | "
              f"margin={row['margin']:4} | lambda={row['lambda_push']:4} | "
              f"epochs={row['epochs']:3}")


def analyze_learning_rates(df):
    """Specific analysis of learning rate effects."""
    print("\n📊 LEARNING RATE ANALYSIS")
    print("=" * 30)
    
    successful_df = df[df['final_f1'].notna()].copy()
    
    lr_analysis = successful_df.groupby(['lr', 'init_method'])['final_f1'].agg(['mean', 'count'])
    
    for lr in sorted(successful_df['lr'].unique()):
        print(f"\nLearning Rate: {lr}")
        lr_data = lr_analysis.loc[lr]
        lr_data = lr_data.sort_values('mean', ascending=False)
        
        for init_method, stats in lr_data.iterrows():
            print(f"  {init_method:15}: F1={stats['mean']:.4f} (n={stats['count']})")


def create_visualizations(df, output_dir):
    """Create visualization plots."""
    print("\n📊 Creating visualizations...")
    
    successful_df = df[df['final_f1'].notna()].copy()
    
    if len(successful_df) == 0:
        print("❌ No successful experiments to visualize")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. F1 scores by initialization method
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=successful_df, x='init_method', y='final_f1')
    plt.title('F1 Scores by Initialization Method')
    plt.xticks(rotation=45)
    
    # 2. F1 scores by learning rate
    plt.subplot(2, 2, 2)
    sns.boxplot(data=successful_df, x='lr', y='final_f1')
    plt.title('F1 Scores by Learning Rate')
    plt.xticks(rotation=45)
    
    # 3. F1 scores by margin
    plt.subplot(2, 2, 3)
    sns.boxplot(data=successful_df, x='margin', y='final_f1')
    plt.title('F1 Scores by Margin')
    
    # 4. F1 scores by lambda
    plt.subplot(2, 2, 4)
    sns.boxplot(data=successful_df, x='lambda_push', y='final_f1')
    plt.title('F1 Scores by Lambda Push')
    
    plt.tight_layout()
    plot_file = output_dir / 'parameter_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Parameter analysis plot saved to {plot_file}")
    
    # 2. Heatmap of init_method vs learning rate
    plt.figure(figsize=(10, 6))
    
    pivot_data = successful_df.pivot_table(
        values='final_f1', 
        index='init_method', 
        columns='lr', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis')
    plt.title('Average F1 Scores: Initialization Method vs Learning Rate')
    plt.tight_layout()
    
    heatmap_file = output_dir / 'init_lr_heatmap.png'
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Heatmap saved to {heatmap_file}")


def generate_recommendations(df):
    """Generate recommendations based on results."""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 30)
    
    successful_df = df[df['final_f1'].notna()].copy()
    
    if len(successful_df) == 0:
        print("❌ No successful experiments to analyze")
        return
    
    # Best overall
    best_result = successful_df.loc[successful_df['final_f1'].idxmax()]
    print(f"🏆 Best overall configuration:")
    print(f"   F1 Score: {best_result['final_f1']:.4f}")
    print(f"   Parameters: init={best_result['init_method']}, lr={best_result['lr']}, "
          f"margin={best_result['margin']}, lambda={best_result['lambda_push']}, "
          f"epochs={best_result['epochs']}")
    
    # Best by init method
    print(f"\n🎯 Best configuration by initialization method:")
    for init_method in successful_df['init_method'].unique():
        method_df = successful_df[successful_df['init_method'] == init_method]
        best_method = method_df.loc[method_df['final_f1'].idxmax()]
        print(f"   {init_method:15}: F1={best_method['final_f1']:.4f} "
              f"(lr={best_method['lr']}, margin={best_method['margin']}, "
              f"lambda={best_method['lambda_push']}, epochs={best_method['epochs']})")
    
    # Learning rate recommendations
    lr_means = successful_df.groupby('lr')['final_f1'].mean().sort_values(ascending=False)
    print(f"\n📈 Learning rate ranking:")
    for lr, mean_f1 in lr_means.items():
        print(f"   {lr:7}: {mean_f1:.4f}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze parameter sweep results')
    parser.add_argument('--results-dir', type=str, default='sweep_results/',
                       help='Directory containing sweep results')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top results to show')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print("📊 PARAMETER SWEEP RESULTS ANALYSIS")
    print("=" * 50)
    print(f"📁 Results directory: {results_dir}")
    
    # Load results
    df = load_results(results_dir)
    if df is None:
        return
    
    # Run analyses
    analyze_parameter_importance(df)
    find_best_combinations(df, args.top_n)
    analyze_learning_rates(df)
    generate_recommendations(df)
    
    # Create visualizations
    if not args.no_plots:
        try:
            create_visualizations(df, results_dir)
        except Exception as e:
            print(f"⚠️  Warning: Could not create plots: {e}")
    
    print(f"\n✅ Analysis completed!")


if __name__ == '__main__':
    main() 