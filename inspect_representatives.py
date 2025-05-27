#!/usr/bin/env python3
"""
Script to inspect and analyze the representatives.pkl file.
"""

import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path

def load_representatives(pkl_path):
    """Load the representatives from pickle file."""
    print(f"📂 Loading representatives from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        rep_data = pickle.load(f)
    
    print(f"✅ Successfully loaded pickle file")
    print(f"📊 Data type: {type(rep_data)}")
    
    return rep_data

def analyze_representatives_structure(rep_data):
    """Analyze the structure of the representatives data."""
    print("\n🔍 REPRESENTATIVES STRUCTURE ANALYSIS")
    print("=" * 50)
    
    if isinstance(rep_data, dict):
        print("📋 Dictionary structure:")
        for key, value in rep_data.items():
            print(f"   '{key}': {type(value)} - {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
            
        # If it contains representatives array
        if 'representatives' in rep_data:
            reps = np.array(rep_data['representatives'])
            print(f"\n🎯 Representatives Array:")
            print(f"   Shape: {reps.shape}")
            print(f"   Data type: {reps.dtype}")
            print(f"   Min value: {reps.min():.6f}")
            print(f"   Max value: {reps.max():.6f}")
            print(f"   Mean: {reps.mean():.6f}")
            print(f"   Std: {reps.std():.6f}")
            
    elif isinstance(rep_data, np.ndarray):
        print("📊 NumPy Array:")
        print(f"   Shape: {rep_data.shape}")
        print(f"   Data type: {rep_data.dtype}")
        print(f"   Min value: {rep_data.min():.6f}")
        print(f"   Max value: {rep_data.max():.6f}")
        print(f"   Mean: {rep_data.mean():.6f}")
        print(f"   Std: {rep_data.std():.6f}")
        
    else:
        print(f"❓ Unknown data type: {type(rep_data)}")

def display_representatives_with_classes(rep_data, results_path):
    """Display representatives with their corresponding class names."""
    print("\n🏷️  CLASS REPRESENTATIVES")
    print("=" * 50)
    
    # Load class names from results.json
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)
    
    class_names = results['data_info']['class_names']
    
    # Extract representatives array
    if isinstance(rep_data, dict) and 'representatives' in rep_data:
        representatives = np.array(rep_data['representatives'])
    elif isinstance(rep_data, np.ndarray):
        representatives = rep_data
    else:
        print("❌ Could not find representatives array")
        return
    
    print(f"📊 Found {len(representatives)} representatives for {len(class_names)} classes")
    print()
    
    # Display each representative
    for i, (class_name, rep_vector) in enumerate(zip(class_names, representatives)):
        print(f"Class {i:2d}: {class_name}")
        print(f"   Representative shape: {rep_vector.shape}")
        print(f"   L2 norm: {np.linalg.norm(rep_vector):.6f}")
        print(f"   First 5 values: {rep_vector[:5]}")
        print(f"   Last 5 values: {rep_vector[-5:]}")
        print()

def compute_similarity_matrix(rep_data):
    """Compute and display similarity matrix between representatives."""
    print("\n🔗 SIMILARITY MATRIX")
    print("=" * 50)
    
    # Extract representatives
    if isinstance(rep_data, dict) and 'representatives' in rep_data:
        representatives = np.array(rep_data['representatives'])
    elif isinstance(rep_data, np.ndarray):
        representatives = rep_data
    else:
        print("❌ Could not find representatives array")
        return
    
    # Normalize representatives
    normalized_reps = representatives / np.linalg.norm(representatives, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(normalized_reps, normalized_reps.T)
    
    print(f"📊 Cosine Similarity Matrix ({similarity_matrix.shape}):")
    print(f"   Diagonal (self-similarity): {np.diag(similarity_matrix)}")
    print(f"   Off-diagonal min: {similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)].min():.6f}")
    print(f"   Off-diagonal max: {similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)].max():.6f}")
    print(f"   Off-diagonal mean: {similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)].mean():.6f}")
    
    return similarity_matrix

def save_representatives_as_csv(rep_data, results_path, class_names):
    """Save representatives as CSV for easy viewing."""
    print("\n💾 SAVING AS CSV")
    print("=" * 50)
    
    # Extract representatives
    if isinstance(rep_data, dict) and 'representatives' in rep_data:
        representatives = np.array(rep_data['representatives'])
    elif isinstance(rep_data, np.ndarray):
        representatives = rep_data
    else:
        print("❌ Could not find representatives array")
        return
    
    # Create DataFrame
    df = pd.DataFrame(representatives)
    df.index = class_names
    df.index.name = 'Class'
    
    # Save to CSV
    csv_path = results_path / 'representatives.csv'
    df.to_csv(csv_path)
    
    print(f"✅ Saved representatives to: {csv_path}")
    print(f"📊 CSV shape: {df.shape}")
    print(f"📋 First few columns:")
    print(df.iloc[:, :5])

def interactive_exploration(rep_data, results_path):
    """Provide interactive exploration options."""
    print("\n🔧 INTERACTIVE EXPLORATION")
    print("=" * 50)
    
    # Load class names
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)
    class_names = results['data_info']['class_names']
    
    # Extract representatives
    if isinstance(rep_data, dict) and 'representatives' in rep_data:
        representatives = np.array(rep_data['representatives'])
    elif isinstance(rep_data, np.ndarray):
        representatives = rep_data
    else:
        print("❌ Could not find representatives array")
        return
    
    print("Available commands:")
    print("  1. View specific class representative")
    print("  2. Compare two classes")
    print("  3. Find most similar classes")
    print("  4. Find most different classes")
    print("  5. Export specific classes")
    print()
    
    # Example: Find most similar and different pairs
    normalized_reps = representatives / np.linalg.norm(representatives, axis=1, keepdims=True)
    similarity_matrix = np.dot(normalized_reps, normalized_reps.T)
    
    # Mask diagonal
    masked_sim = similarity_matrix.copy()
    np.fill_diagonal(masked_sim, -2)  # Set diagonal to very low value
    
    # Most similar pair
    max_idx = np.unravel_index(np.argmax(masked_sim), masked_sim.shape)
    most_similar_score = masked_sim[max_idx]
    
    # Most different pair
    min_idx = np.unravel_index(np.argmin(masked_sim), masked_sim.shape)
    most_different_score = masked_sim[min_idx]
    
    print(f"🔗 Most similar classes:")
    print(f"   {class_names[max_idx[0]]} ↔ {class_names[max_idx[1]]}")
    print(f"   Cosine similarity: {most_similar_score:.6f}")
    print()
    
    print(f"🔀 Most different classes:")
    print(f"   {class_names[min_idx[0]]} ↔ {class_names[min_idx[1]]}")
    print(f"   Cosine similarity: {most_different_score:.6f}")

def main():
    """Main inspection function."""
    results_path = Path("/Users/andrewdelacruz/e2e_sam2/gitignore_exception/first_pass_rep_embeddings")
    pkl_path = results_path / 'representatives.pkl'
    
    print("🔍 REPRESENTATIVES PICKLE FILE INSPECTOR")
    print("=" * 60)
    print(f"📁 File: {pkl_path}")
    print()
    
    # Load the pickle file
    rep_data = load_representatives(pkl_path)
    
    # Analyze structure
    analyze_representatives_structure(rep_data)
    
    # Display with class names
    display_representatives_with_classes(rep_data, results_path)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(rep_data)
    
    # Load class names for CSV export
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)
    class_names = results['data_info']['class_names']
    
    # Save as CSV
    save_representatives_as_csv(rep_data, results_path, class_names)
    
    # Interactive exploration
    interactive_exploration(rep_data, results_path)
    
    print("\n" + "=" * 60)
    print("📋 INSPECTION COMPLETE")
    print("=" * 60)
    print("\n💡 Tips:")
    print("   • Use the CSV file for easy viewing in Excel/Google Sheets")
    print("   • Representatives are normalized vectors in 2048D space")
    print("   • Each row represents one class prototype")
    print("   • Use cosine similarity for comparing classes")

if __name__ == '__main__':
    main() 