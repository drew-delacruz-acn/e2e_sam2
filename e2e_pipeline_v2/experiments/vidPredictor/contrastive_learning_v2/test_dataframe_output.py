#!/usr/bin/env python3
"""
Test script to verify that representatives are saved as DataFrame with correct format.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_representatives_format(pkl_path):
    """Test that the representatives pkl file has the correct DataFrame format."""
    
    print(f"🔍 Testing representatives format from: {pkl_path}")
    
    try:
        # Load the pkl file
        df = pd.read_pickle(pkl_path)
        
        # Check if it's a DataFrame
        if not isinstance(df, pd.DataFrame):
            print(f"❌ Error: Expected DataFrame, got {type(df)}")
            return False
        
        # Check columns
        expected_columns = {'finetuned_embedding', 'class'}
        actual_columns = set(df.columns)
        
        if actual_columns != expected_columns:
            print(f"❌ Error: Expected columns {expected_columns}, got {actual_columns}")
            return False
        
        print(f"✅ DataFrame format is correct!")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"🏷️  Classes: {df['class'].tolist()}")
        
        # Check embedding format
        first_embedding = df['finetuned_embedding'].iloc[0]
        print(f"📐 Embedding shape: {first_embedding.shape}")
        print(f"📊 Embedding dtype: {first_embedding.dtype}")
        
        # Display first few rows (without showing full embeddings)
        print("\n📋 Sample data:")
        for idx, row in df.head().iterrows():
            embedding_shape = row['finetuned_embedding'].shape
            print(f"  Class: {row['class']}, Embedding shape: {embedding_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False

def test_existing_representatives():
    """Test the existing representatives file if it exists."""
    
    # Check the existing representatives file
    existing_path = Path("/Users/andrewdelacruz/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/contrastive_learning_v2/first_pass_rep_embeddings/representatives.pkl")
    
    if existing_path.exists():
        print("🔍 Testing existing representatives file...")
        
        try:
            # Try loading as the old format first
            import pickle
            with open(existing_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                print("📋 Old format detected (dictionary)")
                print(f"Keys: {list(data.keys())}")
                
                if 'representatives' in data and 'class_names' in data:
                    representatives = data['representatives']
                    class_names = data['class_names']
                    
                    print(f"📊 Representatives shape: {representatives.shape}")
                    print(f"🏷️  Classes: {class_names}")
                    
                    # Convert to new format
                    print("\n🔄 Converting to new DataFrame format...")
                    representatives_data = []
                    for i, class_name in enumerate(class_names):
                        representatives_data.append({
                            'finetuned_embedding': representatives[i],
                            'class': class_name
                        })
                    
                    new_df = pd.DataFrame(representatives_data)
                    print(f"✅ Converted DataFrame shape: {new_df.shape}")
                    print(f"📋 Columns: {list(new_df.columns)}")
                    
                    return new_df
                    
            else:
                print("🔍 Trying to load as DataFrame...")
                return test_representatives_format(existing_path)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    else:
        print(f"⚠️  File not found: {existing_path}")
        return None

if __name__ == '__main__':
    print("🧪 Testing Representatives DataFrame Format")
    print("=" * 50)
    
    # Test existing file
    result = test_existing_representatives()
    
    if result is not None and isinstance(result, pd.DataFrame):
        print("\n✅ Test completed successfully!")
        print("The representatives are now in the correct DataFrame format.")
    else:
        print("\n⚠️  No existing representatives found or conversion needed.")
        print("Run the training script to generate representatives in the new format.") 