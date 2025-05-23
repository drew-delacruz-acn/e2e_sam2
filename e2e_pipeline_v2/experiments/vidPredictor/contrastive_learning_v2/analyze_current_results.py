#!/usr/bin/env python3
"""
Analyze current contrastive learning results and suggest improvements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_current_representatives():
    """Analyze the current representatives and suggest improvements."""
    
    print("🔍 Analyzing Current Contrastive Learning Results")
    print("=" * 60)
    
    # Load current representatives
    rep_path = Path("first_pass_rep_embeddings/representatives.pkl")
    
    if not rep_path.exists():
        print(f"❌ Representatives file not found: {rep_path}")
        return
    
    try:
        df = pd.read_pickle(rep_path)
        print(f"✅ Loaded {len(df)} representatives")
        
        # Extract embeddings and compute similarities
        embeddings = np.stack(df['finetuned_embedding'].values)
        class_names = df['class'].values
        
        print(f"📊 Dataset Info:")
        print(f"   Classes: {len(class_names)}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        
        # Normalize for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        # Analyze similarities
        np.fill_diagonal(similarity_matrix, -1)  # Exclude self-similarity
        
        # Find most similar pairs
        print(f"\n🔍 Most Similar Class Pairs (Potential Issues):")
        flat_indices = np.argsort(similarity_matrix.flatten())[::-1]
        
        for i in range(5):  # Top 5 most similar pairs
            flat_idx = flat_indices[i]
            row, col = np.unravel_index(flat_idx, similarity_matrix.shape)
            similarity = similarity_matrix[row, col]
            
            if similarity > 0:  # Only show positive similarities
                print(f"   {i+1}. {class_names[row]} ↔ {class_names[col]}")
                print(f"      Similarity: {similarity:.4f}")
        
        # Compute statistics
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        print(f"\n📈 Similarity Statistics:")
        print(f"   Mean similarity: {np.mean(upper_triangle):.4f}")
        print(f"   Std similarity: {np.std(upper_triangle):.4f}")
        print(f"   Max similarity: {np.max(upper_triangle):.4f}")
        print(f"   Min similarity: {np.min(upper_triangle):.4f}")
        
        # Identify problematic classes
        mean_similarities = []
        for i in range(len(class_names)):
            others = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
            mean_sim = np.mean(others)
            mean_similarities.append((class_names[i], mean_sim))
        
        # Sort by highest mean similarity (most confusing classes)
        mean_similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n⚠️  Most Confusing Classes (High avg similarity to others):")
        for i, (class_name, avg_sim) in enumerate(mean_similarities[:5]):
            print(f"   {i+1}. {class_name}: {avg_sim:.4f}")
        
        # Suggest parameter improvements
        suggest_improvements(np.max(upper_triangle), np.mean(upper_triangle), len(class_names))
        
    except Exception as e:
        print(f"❌ Error analyzing representatives: {e}")

def suggest_improvements(max_similarity, mean_similarity, num_classes):
    """Suggest parameter improvements based on current results."""
    
    print(f"\n🎯 PARAMETER IMPROVEMENT SUGGESTIONS")
    print("=" * 50)
    
    # Analyze current separation quality
    if max_similarity > 0.6:
        separation_quality = "Poor"
        urgency = "High"
    elif max_similarity > 0.4:
        separation_quality = "Moderate" 
        urgency = "Medium"
    else:
        separation_quality = "Good"
        urgency = "Low"
    
    print(f"📊 Current Separation Quality: {separation_quality}")
    print(f"🚨 Improvement Urgency: {urgency}")
    
    print(f"\n💡 Recommended Parameter Changes:")
    
    if max_similarity > 0.5:
        print(f"\n🔥 AGGRESSIVE SEPARATION (High Priority):")
        print(f"   --margin 0.3 --lambda-push 0.6 --lr 0.015 --epochs 100")
        print(f"   Rationale: High max similarity ({max_similarity:.3f}) needs strong separation")
        
        print(f"\n⚡ ALTERNATIVE - VERY AGGRESSIVE:")
        print(f"   --margin 0.4 --lambda-push 0.8 --lr 0.02 --epochs 150")
        print(f"   Rationale: For stubborn similar classes")
        
    elif max_similarity > 0.3:
        print(f"\n📈 MODERATE IMPROVEMENT:")
        print(f"   --margin 0.25 --lambda-push 0.4 --lr 0.012 --epochs 75")
        print(f"   Rationale: Moderate similarity needs balanced approach")
        
        print(f"\n🎯 STABLE ALTERNATIVE:")
        print(f"   --margin 0.2 --lambda-push 0.35 --lr 0.008 --epochs 100")
        print(f"   Rationale: Slower but more stable improvement")
        
    else:
        print(f"\n✨ FINE-TUNING:")
        print(f"   --margin 0.18 --lambda-push 0.3 --lr 0.01 --epochs 60")
        print(f"   Rationale: Good separation, minor adjustments needed")
    
    # Additional suggestions based on dataset size
    if num_classes > 15:
        print(f"\n🏗️  LARGE DATASET CONSIDERATIONS:")
        print(f"   - Consider longer training: --epochs 100-150")
        print(f"   - Use smaller learning rate: --lr 0.005-0.01")
        print(f"   - Increase validation: --val-frac 0.2")
    
    print(f"\n🧪 EXPERIMENTAL APPROACH:")
    print(f"   1. Try 'strict_separation_2' from parameter_experiments.py")
    print(f"   2. If that works, try 'aggressive' for maximum separation")
    print(f"   3. For stability, try 'stable_learning_2'")
    
    print(f"\n📋 QUICK TEST COMMANDS:")
    print(f"   # Quick aggressive test")
    print(f"   python train_representatives.py --data definitiveObjects_jeremiah.pkl \\")
    print(f"     --output quick_test_aggressive --margin 0.3 --lambda-push 0.6 --epochs 50")
    print(f"   ")
    print(f"   # Balanced improvement")
    print(f"   python train_representatives.py --data definitiveObjects_jeremiah.pkl \\")
    print(f"     --output quick_test_balanced --margin 0.22 --lambda-push 0.35 --epochs 60")

def main():
    """Main analysis function."""
    
    analyze_current_representatives()
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Run: python analyze_current_results.py")
    print(f"   2. Try suggested quick tests above")
    print(f"   3. Run: python parameter_experiments.py (for comprehensive testing)")
    print(f"   4. Compare results and pick best parameters")

if __name__ == '__main__':
    main() 