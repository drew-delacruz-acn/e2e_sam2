#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify embedding format in a pickle file")
    parser.add_argument("--input", type=str, required=True, help="Path to input pickle file with embeddings")
    args = parser.parse_args()
    
    print(f"Loading embedding file: {args.input}")
    
    try:
        df = pd.read_pickle(args.input)
        print(f"DataFrame loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for embedding column
        if "embedding" in df.columns:
            col_name = "embedding"
        elif "finetuned_embedding" in df.columns:
            col_name = "finetuned_embedding"
        else:
            print("ERROR: No embedding column found!")
            embedding_cols = [col for col in df.columns if "embedding" in col.lower()]
            if embedding_cols:
                print(f"Potential embedding columns: {embedding_cols}")
            return 1
        
        # Get first embedding sample
        first_embedding = df[col_name].iloc[0]
        print(f"First embedding type: {type(first_embedding)}")
        print(f"First embedding dimension: {len(first_embedding)}")
        
        # Check classes
        if "class" in df.columns:
            unique_classes = df["class"].unique()
            print(f"Number of unique classes: {len(unique_classes)}")
            print(f"First few classes: {unique_classes[:5]}")
        else:
            print("WARNING: No 'class' column found!")
        
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 