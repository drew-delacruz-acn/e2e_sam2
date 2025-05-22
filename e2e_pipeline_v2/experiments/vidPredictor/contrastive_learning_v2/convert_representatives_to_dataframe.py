#!/usr/bin/env python3
"""
Script to convert existing representatives from dictionary format to DataFrame format.
"""

import pandas as pd
import pickle
from pathlib import Path
import shutil

def convert_representatives_to_dataframe(input_path, output_path=None, backup=True):
    """
    Convert representatives from dictionary format to DataFrame format.
    
    Args:
        input_path: Path to the existing representatives.pkl file
        output_path: Path to save the converted file (default: overwrites input)
        backup: Whether to create a backup of the original file
    """
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        return False
    
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    print(f"🔄 Converting representatives from {input_path}")
    
    try:
        # Load the existing file
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if it's already in DataFrame format
        if isinstance(data, pd.DataFrame):
            print("✅ File is already in DataFrame format!")
            
            # Verify columns
            expected_columns = {'finetuned_embedding', 'class'}
            actual_columns = set(data.columns)
            
            if actual_columns == expected_columns:
                print("📋 Columns are correct: finetuned_embedding, class")
                return True
            else:
                print(f"⚠️  Warning: Unexpected columns {actual_columns}")
                return False
        
        # Check if it's in dictionary format
        if not isinstance(data, dict):
            print(f"❌ Error: Unexpected data type: {type(data)}")
            return False
        
        # Verify required keys
        required_keys = {'representatives', 'class_names'}
        if not required_keys.issubset(data.keys()):
            print(f"❌ Error: Missing required keys. Expected {required_keys}, got {list(data.keys())}")
            return False
        
        representatives = data['representatives']
        class_names = data['class_names']
        
        print(f"📊 Original format:")
        print(f"  Representatives shape: {representatives.shape}")
        print(f"  Number of classes: {len(class_names)}")
        print(f"  Classes: {class_names}")
        
        # Create backup if requested
        if backup and output_path == input_path:
            backup_path = input_path.with_suffix('.pkl.backup')
            shutil.copy2(input_path, backup_path)
            print(f"💾 Backup created: {backup_path}")
        
        # Convert to DataFrame format
        representatives_data = []
        for i, class_name in enumerate(class_names):
            representatives_data.append({
                'finetuned_embedding': representatives[i],
                'class': class_name
            })
        
        representatives_df = pd.DataFrame(representatives_data)
        
        # Save the DataFrame
        representatives_df.to_pickle(output_path)
        
        print(f"✅ Conversion successful!")
        print(f"📊 New DataFrame format:")
        print(f"  Shape: {representatives_df.shape}")
        print(f"  Columns: {list(representatives_df.columns)}")
        print(f"  Saved to: {output_path}")
        
        # Verify the conversion
        print(f"\n🔍 Verifying conversion...")
        test_df = pd.read_pickle(output_path)
        
        if isinstance(test_df, pd.DataFrame) and set(test_df.columns) == {'finetuned_embedding', 'class'}:
            print("✅ Verification successful!")
            
            # Show sample data
            print(f"\n📋 Sample data:")
            for idx, row in test_df.head(3).iterrows():
                embedding_shape = row['finetuned_embedding'].shape
                print(f"  Class: {row['class']}, Embedding shape: {embedding_shape}")
            
            return True
        else:
            print("❌ Verification failed!")
            return False
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def main():
    """Main function to convert representatives."""
    
    print("🔄 Converting Representatives to DataFrame Format")
    print("=" * 50)
    
    # Convert the existing representatives file
    representatives_path = Path("first_pass_rep_embeddings/representatives.pkl")
    
    if representatives_path.exists():
        success = convert_representatives_to_dataframe(representatives_path)
        
        if success:
            print("\n🎉 Conversion completed successfully!")
            print("The representatives file now uses the DataFrame format with 'finetuned_embedding' and 'class' columns.")
        else:
            print("\n❌ Conversion failed!")
    else:
        print(f"⚠️  Representatives file not found: {representatives_path}")
        print("Run the training script first to generate representatives.")

if __name__ == '__main__':
    main() 