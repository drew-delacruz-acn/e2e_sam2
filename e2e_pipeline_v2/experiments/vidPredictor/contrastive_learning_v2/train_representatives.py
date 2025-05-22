#!/usr/bin/env python3
"""
Main script for contrastive learning experiment.

This script runs the complete pipeline:
1. Load and validate PKL data
2. Split into train/validation sets
3. Train contrastive representatives
4. Evaluate performance
5. Generate visualizations
6. Save results

Usage:
    python train_representatives.py --data path/to/data.pkl --output results/
"""

import argparse
import json
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.data_loader import load_and_split_data
from src.trainer import ContrastiveTrainer
from src.visualizer import create_plots


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train contrastive representatives')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to PKL file containing embeddings')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory for results')
    parser.add_argument('--val-frac', type=float, default=0.3,
                       help='Fraction of data for validation')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.15,
                       help='Cosine similarity margin for negatives')
    parser.add_argument('--lambda-push', type=float, default=0.25,
                       help='Weight for push term')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    """Main experiment function."""
    args = parse_args()
    
    # Set up
    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Contrastive Learning Experiment")
    print(f"📁 Data: {args.data}")
    print(f"📊 Output: {output_dir}")
    print(f"⚙️  Config: lr={args.lr}, margin={args.margin}, λ={args.lambda_push}")
    print(f"🔄 Epochs: {args.epochs}")
    print()
    
    # 1. Load and split data
    print("📥 Loading and splitting data...")
    try:
        train_df, val_df, class_names = load_and_split_data(
            args.data, 
            val_frac=args.val_frac, 
            random_state=args.seed
        )
        print(f"✅ Loaded {len(train_df)} training samples, {len(val_df)} validation samples")
        print(f"🏷️  Classes: {class_names}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 2. Convert to tensors
    print("🔄 Converting to tensors...")
    train_embeddings = torch.tensor(
        np.stack(train_df['finetuned_embedding'].tolist()), 
        dtype=torch.float32
    )
    val_embeddings = torch.tensor(
        np.stack(val_df['finetuned_embedding'].tolist()), 
        dtype=torch.float32
    ) if len(val_df) > 0 else torch.empty(0, train_embeddings.shape[1])
    
    # Create label mappings
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    train_labels = torch.tensor([class_to_idx[cls] for cls in train_df['class']])
    val_labels = torch.tensor([class_to_idx[cls] for cls in val_df['class']]) if len(val_df) > 0 else torch.empty(0, dtype=torch.long)
    
    print(f"📐 Embedding dimension: {train_embeddings.shape[1]}")
    print(f"🔢 Number of classes: {len(class_names)}")
    print()
    
    # 3. Initialize trainer
    config = {
        'lr': args.lr,
        'margin': args.margin,
        'lambda_push': args.lambda_push,
        'epochs': args.epochs
    }
    
    trainer = ContrastiveTrainer(config)
    device = trainer.device
    print(f"🖥️  Using device: {device}")
    
    # 4. Initialize representatives and get baseline
    print("🎯 Initializing representatives as class means...")
    trainer.initialize_representatives(train_embeddings, train_labels)
    
    # Evaluate baseline performance
    if len(val_df) > 0:
        baseline_f1 = trainer.evaluate(val_embeddings, val_labels, trainer.representatives.detach())
        print(f"📊 Baseline F1 (class means): {baseline_f1:.4f}")
    else:
        baseline_f1 = None
        print("⚠️  No validation data - skipping baseline evaluation")
    print()
    
    # 5. Train representatives
    print("🏋️  Training representatives...")
    history = trainer.train(train_embeddings, train_labels, epochs=args.epochs)
    
    # Show training progress
    initial_loss = history['losses'][0]
    final_loss = history['losses'][-1]
    print(f"📉 Loss: {initial_loss:.4f} → {final_loss:.4f} (Δ: {initial_loss - final_loss:.4f})")
    
    # 6. Evaluate final performance
    if len(val_df) > 0:
        final_f1 = trainer.evaluate(val_embeddings, val_labels, trainer.representatives.detach())
        print(f"🎯 Final F1: {final_f1:.4f}")
        
        if baseline_f1 is not None:
            improvement = final_f1 - baseline_f1
            print(f"📈 Improvement: {improvement:+.4f}")
    else:
        final_f1 = None
        improvement = None
    print()
    
    # 7. Save results
    print("💾 Saving results...")
    
    # Save configuration and metrics
    results = {
        'config': config,
        'data_info': {
            'num_train_samples': len(train_df),
            'num_val_samples': len(val_df),
            'num_classes': len(class_names),
            'class_names': class_names,
            'embedding_dim': train_embeddings.shape[1]
        },
        'metrics': {
            'baseline_f1': baseline_f1,
            'final_f1': final_f1,
            'improvement': improvement,
            'initial_loss': initial_loss,
            'final_loss': final_loss
        },
        'history': history,
        'seed': args.seed
    }
    
    # Save results JSON
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {results_file}")
    
    # Save learned representatives as DataFrame
    representatives_file = output_dir / 'representatives.pkl'
    
    # Create DataFrame with finetuned_embedding and class columns
    representatives_data = []
    representatives_numpy = trainer.representatives.detach().cpu().numpy()
    
    for i, class_name in enumerate(class_names):
        representatives_data.append({
            'finetuned_embedding': representatives_numpy[i],
            'class': class_name
        })
    
    representatives_df = pd.DataFrame(representatives_data)
    representatives_df.to_pickle(representatives_file)
    print(f"✅ Representatives DataFrame saved to {representatives_file}")
    print(f"📊 DataFrame shape: {representatives_df.shape}")
    print(f"📋 Columns: {list(representatives_df.columns)}")
    
    # 8. Generate visualizations
    print("📊 Generating visualizations...")
    try:
        create_plots(
            history=history,
            representatives=trainer.representatives.detach().cpu(),
            embeddings=train_embeddings,
            labels=train_labels,
            output_dir=output_dir,
            class_names=class_names
        )
        print(f"✅ Plots saved to {output_dir}")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")
    
    print()
    print("🎉 Experiment completed successfully!")
    print(f"📁 All results saved to: {output_dir}")
    
    # Summary
    print("\n" + "="*50)
    print("📋 EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Data: {args.data}")
    print(f"Classes: {len(class_names)} ({', '.join(class_names)})")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Embedding dimension: {train_embeddings.shape[1]}")
    print(f"Training epochs: {args.epochs}")
    print(f"Final loss: {final_loss:.4f}")
    if final_f1 is not None:
        print(f"Final F1 score: {final_f1:.4f}")
        if improvement is not None:
            print(f"Improvement over baseline: {improvement:+.4f}")
    print(f"Results directory: {output_dir}")
    print("="*50)


if __name__ == '__main__':
    main() 