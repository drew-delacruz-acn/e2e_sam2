#!/usr/bin/env python3
"""
Test script for contrastive learning experiment using toy data.

This script creates synthetic data and runs the complete pipeline to verify
that all components work correctly together.
"""

import numpy as np
import pandas as pd
import pickle
import tempfile
from pathlib import Path

from src.data_loader import load_and_split_data
from src.trainer import ContrastiveTrainer
from src.visualizer import create_plots


def create_toy_data():
    """Create synthetic toy data for testing."""
    np.random.seed(42)
    
    # Create 3 classes with well-separated 2D embeddings
    data = []
    
    # Class A: around (2, 2)
    for i in range(15):
        embedding = np.array([2.0, 2.0]) + 0.3 * np.random.randn(2)
        data.append({'class': 'mirror', 'fine_tuned_embeddings': embedding.astype(np.float32)})
    
    # Class B: around (-2, -2)
    for i in range(15):
        embedding = np.array([-2.0, -2.0]) + 0.3 * np.random.randn(2)
        data.append({'class': 'lamp', 'fine_tuned_embeddings': embedding.astype(np.float32)})
    
    # Class C: around (2, -2)
    for i in range(15):
        embedding = np.array([2.0, -2.0]) + 0.3 * np.random.randn(2)
        data.append({'class': 'wall', 'fine_tuned_embeddings': embedding.astype(np.float32)})
    
    return pd.DataFrame(data)


def test_experiment():
    """Run the complete experiment with toy data."""
    print("🧪 Testing Contrastive Learning Experiment")
    print("=" * 50)
    
    # 1. Create toy data
    print("📊 Creating toy data...")
    toy_df = create_toy_data()
    print(f"✅ Created {len(toy_df)} samples with {len(toy_df['class'].unique())} classes")
    
    # Save to temporary PKL file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(toy_df, f)
        pkl_path = f.name
    
    try:
        # 2. Load and split data
        print("\n📥 Loading and splitting data...")
        train_df, val_df, class_names = load_and_split_data(pkl_path, val_frac=0.3, random_state=42)
        print(f"✅ Split: {len(train_df)} train, {len(val_df)} val")
        print(f"🏷️  Classes: {class_names}")
        
        # 3. Convert to tensors
        print("\n🔄 Converting to tensors...")
        import torch
        
        train_embeddings = torch.tensor(
            np.stack(train_df['fine_tuned_embeddings'].tolist()), 
            dtype=torch.float32
        )
        val_embeddings = torch.tensor(
            np.stack(val_df['fine_tuned_embeddings'].tolist()), 
            dtype=torch.float32
        )
        
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        train_labels = torch.tensor([class_to_idx[cls] for cls in train_df['class']])
        val_labels = torch.tensor([class_to_idx[cls] for cls in val_df['class']])
        
        print(f"📐 Embedding shape: {train_embeddings.shape}")
        
        # 4. Initialize trainer
        print("\n🎯 Initializing trainer...")
        config = {
            'lr': 0.05,      # Higher LR for faster convergence on toy data
            'margin': 0.15,
            'lambda_push': 0.5,  # Higher push for better separation
            'epochs': 20     # Fewer epochs for testing
        }
        
        trainer = ContrastiveTrainer(config)
        trainer.initialize_representatives(train_embeddings, train_labels)
        
        # 5. Evaluate baseline
        print("\n📊 Evaluating baseline...")
        baseline_f1 = trainer.evaluate(val_embeddings, val_labels, trainer.representatives.detach())
        print(f"📊 Baseline F1: {baseline_f1:.4f}")
        
        # 6. Train
        print("\n🏋️  Training...")
        history = trainer.train(train_embeddings, train_labels, epochs=config['epochs'])
        
        initial_loss = history['losses'][0]
        final_loss = history['losses'][-1]
        print(f"📉 Loss: {initial_loss:.4f} → {final_loss:.4f}")
        
        # 7. Evaluate final
        print("\n🎯 Final evaluation...")
        final_f1 = trainer.evaluate(val_embeddings, val_labels, trainer.representatives.detach())
        improvement = final_f1 - baseline_f1
        print(f"🎯 Final F1: {final_f1:.4f}")
        print(f"📈 Improvement: {improvement:+.4f}")
        
        # 8. Generate plots
        print("\n📊 Generating plots...")
        output_dir = Path('test_results')
        output_dir.mkdir(exist_ok=True)
        
        create_plots(
            history=history,
            representatives=trainer.representatives.detach().cpu(),
            embeddings=train_embeddings,
            labels=train_labels,
            output_dir=output_dir,
            class_names=class_names
        )
        print(f"✅ Plots saved to {output_dir}")
        
        # 9. Verify results
        print("\n✅ VERIFICATION")
        print("=" * 30)
        
        # Check that loss decreased
        assert final_loss < initial_loss, "Loss should decrease during training"
        print("✅ Loss decreased during training")
        
        # Check that F1 is reasonable (should be high for separable data)
        assert final_f1 > 0.7, f"F1 should be high for separable data, got {final_f1:.4f}"
        print("✅ F1 score is reasonable")
        
        # Check that improvement is non-negative
        assert improvement >= -0.1, f"Should not degrade significantly, got {improvement:.4f}"
        print("✅ Performance maintained or improved")
        
        print("\n🎉 ALL TESTS PASSED!")
        print(f"📁 Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        Path(pkl_path).unlink(missing_ok=True)


if __name__ == '__main__':
    success = test_experiment()
    exit(0 if success else 1) 