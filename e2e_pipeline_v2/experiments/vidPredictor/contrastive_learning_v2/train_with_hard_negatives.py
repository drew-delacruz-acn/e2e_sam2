#!/usr/bin/env python3
"""
Enhanced contrastive learning training with hard negatives.

This script trains representatives using both standard contrastive loss
and hard negative mining from false positives.
"""

import os
import sys
import yaml
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from trainer import ContrastiveTrainer
from data_loader import ContrastiveDataset
from sklearn.preprocessing import LabelEncoder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train with hard negatives')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--representatives_data', type=str, required=True,
                       help='Path to representatives pickle file')
    parser.add_argument('--hard_negatives', type=str, required=True,
                       help='Path to hard negatives pickle file')
    parser.add_argument('--output_dir', type=str, default='results/hard_negative_training',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lambda_hard', type=float, default=0.5,
                       help='Weight for hard negative loss')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_representatives_data(data_path: str) -> pd.DataFrame:
    """Load representatives data from pickle file."""
    print(f"📥 Loading representatives data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"   Found {len(data)} representative samples")
    return data

def load_hard_negatives(negatives_path: str) -> Dict[str, List[Dict]]:
    """Load hard negatives from pickle file."""
    print(f"📥 Loading hard negatives from: {negatives_path}")
    with open(negatives_path, 'rb') as f:
        hard_negatives = pickle.load(f)
    
    total_negatives = sum(len(negs) for negs in hard_negatives.values())
    print(f"   Found {total_negatives} hard negatives across {len(hard_negatives)} classes")
    return hard_negatives

def prepare_training_data(representatives_df: pd.DataFrame, validation_split: float = 0.2):
    """
    Prepare training and validation data from representatives.
    
    Args:
        representatives_df: DataFrame with representatives data
        validation_split: Fraction to use for validation
        
    Returns:
        Tuple of (train_embeddings, train_labels, val_embeddings, val_labels, class_names, label_encoder)
    """
    print(f"🔄 Preparing training data...")
    
    # Extract embeddings and labels
    embeddings = np.stack(representatives_df['finetuned_embedding'].values)
    class_names = representatives_df['class'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(class_names)
    
    # Convert to tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Split into train/validation
    n_samples = len(embeddings_tensor)
    n_val = int(n_samples * validation_split)
    
    # Random split
    indices = torch.randperm(n_samples)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_embeddings = embeddings_tensor[train_indices]
    train_labels = labels_tensor[train_indices]
    val_embeddings = embeddings_tensor[val_indices]
    val_labels = labels_tensor[val_indices]
    
    print(f"   Training samples: {len(train_embeddings)}")
    print(f"   Validation samples: {len(val_embeddings)}")
    print(f"   Classes: {len(label_encoder.classes_)}")
    
    return train_embeddings, train_labels, val_embeddings, val_labels, label_encoder.classes_, label_encoder

def train_with_hard_negatives(trainer: ContrastiveTrainer, 
                            train_embeddings: torch.Tensor,
                            train_labels: torch.Tensor,
                            val_embeddings: torch.Tensor,
                            val_labels: torch.Tensor,
                            epochs: int,
                            lambda_hard: float) -> Dict[str, List[float]]:
    """
    Train the model with hard negatives.
    
    Args:
        trainer: ContrastiveTrainer instance
        train_embeddings: Training embeddings
        train_labels: Training labels
        val_embeddings: Validation embeddings
        val_labels: Validation labels
        epochs: Number of epochs
        lambda_hard: Weight for hard negative loss
        
    Returns:
        Training history dictionary
    """
    print(f"🚀 Starting training with hard negatives...")
    print(f"   Epochs: {epochs}")
    print(f"   Lambda hard: {lambda_hard}")
    
    history = {
        'total_losses': [],
        'standard_losses': [],
        'hard_losses': [],
        'val_f1_scores': []
    }
    
    for epoch in range(epochs):
        # Training step with hard negatives
        total_loss, standard_loss, hard_loss = trainer.train_step_with_hard_negatives(
            train_embeddings, train_labels, lambda_hard
        )
        
        # Validation
        val_f1 = trainer.evaluate(val_embeddings, val_labels)
        
        # Record history
        history['total_losses'].append(total_loss.item())
        history['standard_losses'].append(standard_loss.item())
        history['hard_losses'].append(hard_loss.item())
        history['val_f1_scores'].append(val_f1)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Total Loss: {total_loss:.4f}, "
                  f"Standard: {standard_loss:.4f}, "
                  f"Hard: {hard_loss:.4f}, "
                  f"Val F1: {val_f1:.4f}")
    
    print("✅ Training completed!")
    return history

def save_results(trainer: ContrastiveTrainer, 
                history: Dict[str, List[float]], 
                class_names: List[str],
                output_dir: str):
    """Save training results."""
    print(f"💾 Saving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trained representatives
    representatives_df = trainer.save_representatives_to_dataframe(class_names)
    representatives_path = os.path.join(output_dir, 'trained_representatives.pkl')
    with open(representatives_path, 'wb') as f:
        pickle.dump(representatives_df, f)
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Save representatives as CSV for easy inspection
    csv_path = os.path.join(output_dir, 'trained_representatives.csv')
    representatives_df.to_csv(csv_path, index=False)
    
    print("✅ Results saved successfully!")

def print_training_summary(history: Dict[str, List[float]]):
    """Print summary of training results."""
    print("\n" + "="*60)
    print("🎯 TRAINING SUMMARY")
    print("="*60)
    
    final_total_loss = history['total_losses'][-1]
    final_standard_loss = history['standard_losses'][-1]
    final_hard_loss = history['hard_losses'][-1]
    final_val_f1 = history['val_f1_scores'][-1]
    
    best_val_f1 = max(history['val_f1_scores'])
    best_epoch = history['val_f1_scores'].index(best_val_f1) + 1
    
    print(f"Final losses:")
    print(f"  Total Loss: {final_total_loss:.4f}")
    print(f"  Standard Loss: {final_standard_loss:.4f}")
    print(f"  Hard Negative Loss: {final_hard_loss:.4f}")
    print(f"  Validation F1: {final_val_f1:.4f}")
    print()
    print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
    
    # Loss reduction
    initial_total_loss = history['total_losses'][0]
    loss_reduction = (initial_total_loss - final_total_loss) / initial_total_loss * 100
    print(f"Total loss reduction: {loss_reduction:.1f}%")

def main():
    args = parse_args()
    
    print("🚀 Starting enhanced contrastive learning with hard negatives...")
    print(f"   Config: {args.config}")
    print(f"   Representatives: {args.representatives_data}")
    print(f"   Hard negatives: {args.hard_negatives}")
    print(f"   Output: {args.output_dir}")
    print()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    representatives_df = load_representatives_data(args.representatives_data)
    hard_negatives = load_hard_negatives(args.hard_negatives)
    
    # Prepare training data
    train_embeddings, train_labels, val_embeddings, val_labels, class_names, label_encoder = prepare_training_data(
        representatives_df, args.validation_split
    )
    
    # Initialize trainer
    trainer = ContrastiveTrainer(config)
    
    # Load initial representatives
    trainer.load_representatives_from_dataframe(representatives_df, class_names)
    
    # Set hard negatives
    trainer.set_hard_negatives(hard_negatives)
    
    # Train with hard negatives
    history = train_with_hard_negatives(
        trainer, train_embeddings, train_labels, val_embeddings, val_labels,
        args.epochs, args.lambda_hard
    )
    
    # Save results
    save_results(trainer, history, class_names, args.output_dir)
    
    # Print summary
    print_training_summary(history)
    
    print("\n🎉 Enhanced training completed!")

if __name__ == "__main__":
    main() 