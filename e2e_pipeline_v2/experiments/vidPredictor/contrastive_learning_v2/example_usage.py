#!/usr/bin/env python3
"""
Example script showing how to load and use the representatives DataFrame.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_representatives(pkl_path):
    """Load representatives DataFrame from pkl file."""
    
    print(f"📥 Loading representatives from: {pkl_path}")
    
    try:
        df = pd.read_pickle(pkl_path)
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(df)}")
        
        expected_columns = {'finetuned_embedding', 'class'}
        if set(df.columns) != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {set(df.columns)}")
        
        print(f"✅ Successfully loaded {len(df)} representatives")
        return df
        
    except Exception as e:
        print(f"❌ Error loading representatives: {e}")
        return None

def demonstrate_usage():
    """Demonstrate how to use the representatives DataFrame."""
    
    # Load the representatives
    representatives_path = Path("first_pass_rep_embeddings/representatives.pkl")
    df = load_representatives(representatives_path)
    
    if df is None:
        return
    
    print(f"\n📊 Representatives DataFrame Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Classes: {len(df)} unique classes")
    
    # Show all classes
    print(f"\n🏷️  All Classes:")
    for i, class_name in enumerate(df['class']):
        embedding_shape = df.iloc[i]['finetuned_embedding'].shape
        print(f"  {i+1:2d}. {class_name} (embedding: {embedding_shape})")
    
    # Example 1: Get embedding for a specific class
    print(f"\n🔍 Example 1: Get embedding for a specific class")
    target_class = "Loki's dagger"
    
    if target_class in df['class'].values:
        class_row = df[df['class'] == target_class].iloc[0]
        embedding = class_row['finetuned_embedding']
        print(f"  Class: {target_class}")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding dtype: {embedding.dtype}")
        print(f"  First 5 values: {embedding[:5]}")
    else:
        print(f"  Class '{target_class}' not found")
    
    # Example 2: Convert to arrays for computation
    print(f"\n🔢 Example 2: Convert to arrays for computation")
    
    # Extract all embeddings as a numpy array
    embeddings_array = np.stack(df['finetuned_embedding'].values)
    class_names = df['class'].values
    
    print(f"  Embeddings array shape: {embeddings_array.shape}")
    print(f"  Class names array shape: {class_names.shape}")
    
    # Example 3: Compute pairwise similarities
    print(f"\n📐 Example 3: Compute pairwise similarities")
    
    # Normalize embeddings for cosine similarity
    embeddings_normalized = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)
    
    print(f"  Similarity matrix shape: {similarity_matrix.shape}")
    print(f"  Diagonal (self-similarity): {np.diag(similarity_matrix)[:3]} ...")
    
    # Find most similar pair (excluding self-similarity)
    np.fill_diagonal(similarity_matrix, -1)  # Exclude self-similarity
    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    max_similarity = similarity_matrix[max_sim_idx]
    
    class1 = class_names[max_sim_idx[0]]
    class2 = class_names[max_sim_idx[1]]
    
    print(f"  Most similar classes:")
    print(f"    {class1} <-> {class2}")
    print(f"    Similarity: {max_similarity:.4f}")
    
    # Example 4: Save in different formats
    print(f"\n💾 Example 4: Save in different formats")
    
    # Save as CSV (embeddings will be converted to string representation)
    csv_path = "representatives_export.csv"
    df_export = df.copy()
    df_export['embedding_str'] = df_export['finetuned_embedding'].apply(lambda x: ','.join(map(str, x)))
    df_export[['class', 'embedding_str']].to_csv(csv_path, index=False)
    print(f"  Exported to CSV: {csv_path}")
    
    # Save embeddings as separate numpy file
    np_path = "representatives_embeddings.npy"
    np.save(np_path, embeddings_array)
    print(f"  Embeddings saved to: {np_path}")
    
    # Save class names
    classes_path = "representatives_classes.txt"
    with open(classes_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"  Class names saved to: {classes_path}")
    
    print(f"\n✅ Usage demonstration completed!")

def main():
    """Main function."""
    
    print("📋 Representatives DataFrame Usage Example")
    print("=" * 50)
    
    demonstrate_usage()

if __name__ == '__main__':
    main() 