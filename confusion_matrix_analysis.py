#!/usr/bin/env python3
"""
Confusion Matrix Analysis from Tracking Data
Shows confusion matrices for each iteration and compares them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_confusion_matrix_from_metrics(tp, fp, fn, tn):
    """Create a 2x2 confusion matrix from TP, FP, FN, TN values."""
    # Standard confusion matrix layout:
    # [[TN, FP],
    #  [FN, TP]]
    return np.array([[tn, fp], [fn, tp]])

def plot_confusion_matrix(cm, title, iteration, save_path=None):
    """Plot a confusion matrix with labels."""
    
    plt.figure(figsize=(8, 6))
    
    # Create labels
    labels = ['Negative', 'Positive']
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'{title}\nIteration {iteration}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Saved confusion matrix: {save_path}")
    
    plt.show()
    return plt.gcf()

def calculate_metrics_from_cm(cm):
    """Calculate performance metrics from confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity
    }

def analyze_confusion_matrices():
    """Analyze confusion matrices from tracking data."""
    
    tracking_dir = Path("/Users/andrewdelacruz/e2e_sam2/gitignore_exception/tracking_exports")
    
    print("📊 CONFUSION MATRIX ANALYSIS")
    print("=" * 50)
    
    # Load iteration summaries
    summaries = pd.read_csv(tracking_dir / "iteration_summaries_all.csv")
    
    print(f"\n🔍 Found data for {len(summaries)} iterations")
    
    # Create output directory for plots
    output_dir = Path("confusion_matrices")
    output_dir.mkdir(exist_ok=True)
    
    confusion_matrices = []
    metrics_comparison = []
    
    for idx, row in summaries.iterrows():
        iteration = int(row['iteration'])
        tp = int(row['tp'])
        fp = int(row['fp'])
        fn = int(row['fn'])
        tn = int(row['tn'])
        
        print(f"\n🎯 ITERATION {iteration}")
        print("-" * 30)
        print(f"   TP: {tp:4d}  |  FP: {fp:4d}")
        print(f"   FN: {fn:4d}  |  TN: {tn:4d}")
        
        # Create confusion matrix
        cm = create_confusion_matrix_from_metrics(tp, fp, fn, tn)
        confusion_matrices.append(cm)
        
        # Calculate metrics
        metrics = calculate_metrics_from_cm(cm)
        metrics['iteration'] = iteration
        metrics_comparison.append(metrics)
        
        print(f"   📊 F1: {metrics['f1']:.4f}")
        print(f"   📊 Precision: {metrics['precision']:.4f}")
        print(f"   📊 Recall: {metrics['recall']:.4f}")
        print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"   📊 Specificity: {metrics['specificity']:.4f}")
        
        # Plot confusion matrix
        save_path = output_dir / f"confusion_matrix_iteration_{iteration}.png"
        plot_confusion_matrix(cm, f"Confusion Matrix", iteration, save_path)
    
    # Compare iterations
    if len(confusion_matrices) >= 2:
        print(f"\n🔄 ITERATION COMPARISON")
        print("-" * 50)
        
        cm1, cm2 = confusion_matrices[0], confusion_matrices[1]
        
        print(f"📊 Changes from Iteration 1 to 2:")
        print(f"   TP: {cm1[1,1]} → {cm2[1,1]} ({cm2[1,1] - cm1[1,1]:+d})")
        print(f"   FP: {cm1[0,1]} → {cm2[0,1]} ({cm2[0,1] - cm1[0,1]:+d})")
        print(f"   FN: {cm1[1,0]} → {cm2[1,0]} ({cm2[1,0] - cm1[1,0]:+d})")
        print(f"   TN: {cm1[0,0]} → {cm2[0,0]} ({cm2[0,0] - cm1[0,0]:+d})")
        
        # Plot side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        labels = ['Negative', 'Positive']
        
        for i, (cm, iteration) in enumerate(zip(confusion_matrices, [1, 2])):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[i], cbar_kws={'label': 'Count'})
            axes[i].set_title(f'Iteration {iteration}', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
            # Add percentages
            total = cm.sum()
            for row in range(2):
                for col in range(2):
                    percentage = (cm[row, col] / total) * 100
                    axes[i].text(col + 0.5, row + 0.7, f'({percentage:.1f}%)', 
                               ha='center', va='center', fontsize=9, color='red')
        
        plt.suptitle('Confusion Matrix Comparison: Iteration 1 vs 2', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        comparison_path = output_dir / "confusion_matrix_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Saved comparison: {comparison_path}")
        plt.show()
    
    # Create metrics comparison table
    print(f"\n📈 METRICS COMPARISON TABLE")
    print("-" * 50)
    
    metrics_df = pd.DataFrame(metrics_comparison)
    
    print(f"{'Iteration':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<9} {'Specificity':<12}")
    print("-" * 70)
    
    for _, row in metrics_df.iterrows():
        print(f"{row['iteration']:<10} {row['f1']:<8.4f} {row['precision']:<10.4f} "
              f"{row['recall']:<8.4f} {row['accuracy']:<9.4f} {row['specificity']:<12.4f}")
    
    # Calculate improvements
    if len(metrics_df) >= 2:
        print(f"\n🎯 IMPROVEMENTS FROM ITERATION 1 TO 2")
        print("-" * 50)
        
        iter1 = metrics_df.iloc[0]
        iter2 = metrics_df.iloc[1]
        
        improvements = {
            'F1': (iter2['f1'] - iter1['f1'], (iter2['f1'] - iter1['f1']) / iter1['f1'] * 100),
            'Precision': (iter2['precision'] - iter1['precision'], (iter2['precision'] - iter1['precision']) / iter1['precision'] * 100),
            'Recall': (iter2['recall'] - iter1['recall'], (iter2['recall'] - iter1['recall']) / iter1['recall'] * 100),
            'Accuracy': (iter2['accuracy'] - iter1['accuracy'], (iter2['accuracy'] - iter1['accuracy']) / iter1['accuracy'] * 100),
            'Specificity': (iter2['specificity'] - iter1['specificity'], (iter2['specificity'] - iter1['specificity']) / iter1['specificity'] * 100)
        }
        
        for metric, (abs_change, pct_change) in improvements.items():
            direction = "📈" if abs_change > 0 else "📉" if abs_change < 0 else "➡️"
            print(f"   {direction} {metric}: {abs_change:+.4f} ({pct_change:+.1f}%)")
    
    print(f"\n🎯 CONFUSION MATRIX INSIGHTS")
    print("-" * 50)
    
    if len(confusion_matrices) >= 2:
        cm1, cm2 = confusion_matrices[0], confusion_matrices[1]
        
        # Analyze the changes
        tp_change = cm2[1,1] - cm1[1,1]
        fp_change = cm2[0,1] - cm1[0,1]
        fn_change = cm2[1,0] - cm1[1,0]
        tn_change = cm2[0,0] - cm1[0,0]
        
        print(f"✅ What improved:")
        if fp_change < 0:
            print(f"   📉 False Positives reduced by {abs(fp_change)} ({abs(fp_change/cm1[0,1]*100):.1f}%)")
        if fn_change < 0:
            print(f"   📉 False Negatives reduced by {abs(fn_change)} ({abs(fn_change/cm1[1,0]*100):.1f}%)")
        if tn_change > 0:
            print(f"   📈 True Negatives increased by {tn_change} ({tn_change/cm1[0,0]*100:.1f}%)")
        if tp_change > 0:
            print(f"   📈 True Positives increased by {tp_change} ({tp_change/cm1[1,1]*100:.1f}%)")
        
        print(f"\n🎯 Key takeaway:")
        if fp_change < 0 and tn_change > 0:
            print(f"   ✅ Filtering successfully reduced false positives!")
            print(f"   ✅ Model became better at correctly identifying negatives!")
        
        print(f"\n📂 All confusion matrices saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    analyze_confusion_matrices() 