import os
import json
import time
import requests
import numpy as np
import twelvelabs
from dotenv import load_dotenv
from twelvelabs.models.embed import SegmentEmbedding
from typing import List
import requests
import io
from PIL import Image, ImageOps
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import argparse
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze single model results')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to representatives.pkl file')
    parser.add_argument('--resnet-data', type=str, required=True,
                       help='Path to ResNet embeddings pkl file')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth pkl file')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Confidence threshold for predictions')
    parser.add_argument('--output', type=str, default='analysis_results.pkl',
                       help='Output file for results')
    
    return parser.parse_args()

def cosine_scores(input_emb, class_emb_matrix):
    """Calculate cosine similarity scores."""
    input_tensor = F.normalize(torch.tensor(input_emb, dtype=torch.float32).unsqueeze(0), dim=1)
    class_tensor = F.normalize(torch.tensor(np.vstack(class_emb_matrix), dtype=torch.float32), dim=1)
    return F.cosine_similarity(input_tensor, class_tensor).numpy()

def visual_prediction_from_text_filtered(row, class_emb_matrix, objs):
    """Generate predictions for a single row."""
    visual_scores = cosine_scores(row['finetuned_embedding'], class_emb_matrix)
    best_idx = np.argmax(visual_scores)
    return pd.Series({
        'visual_predicted_object': objs[best_idx],
        'visual_max_score': visual_scores[best_idx],
    })

def classify_answer(answer, pred):
    """Classify prediction as TP, FP, TN, or FN."""
    if answer == 0 and pred == 0:
        return 'TN'
    elif answer == 1 and pred == 1:
        return 'TP'
    elif answer == 1 and pred == 0:
        return 'FN'
    return 'FP'

def main():
    args = parse_args()
    
    print(f"🔍 Analyzing model: {args.model_path}")
    print(f"📊 Threshold: {args.threshold}")
    print(f"📥 ResNet data: {args.resnet_data}")
    print(f"🎯 Ground truth: {args.ground_truth}")
    print()
    
    # Load ResNet embeddings
    print("📥 Loading ResNet embeddings...")
    with open(args.resnet_data, 'rb') as file:
        resnet_df = pickle.load(file)
    
    resnet_df = resnet_df[['video', 'frame', 'owl_label', 'finetuned_embedding']]
    print(f"✅ Loaded {len(resnet_df)} ResNet samples")
    
    # Load model representatives
    print("🤖 Loading model representatives...")
    with open(args.model_path, 'rb') as file:
        defObjects = pickle.load(file)
    
    class_emb_matrix = list(defObjects['finetuned_embedding'])
    objs = list(defObjects['class'])
    print(f"✅ Loaded representatives for {len(objs)} classes: {objs}")
    
    # Generate predictions
    print("🔮 Generating predictions...")
    visual_preds = resnet_df.apply(
        lambda row: visual_prediction_from_text_filtered(row, class_emb_matrix, objs), 
        axis=1
    )
    resnet_df[['visual_predicted_object', 'visual_max_score']] = visual_preds
    
    # Apply threshold filtering
    print(f"🎯 Applying threshold: {args.threshold}")
    resnet_df['prediction'] = 1
    resnet_df['frame'] = resnet_df['frame'].apply(lambda row: int(row))
    
    # Keep highest scoring prediction per video-object pair
    idx = resnet_df.groupby(['video', 'visual_predicted_object'])['visual_max_score'].idxmax()
    resnet_df = resnet_df.loc[idx].reset_index(drop=True)
    
    # Filter by threshold
    resnet_df['visual_predicted_object'] = resnet_df.apply(
        lambda row: row['visual_predicted_object'] if row['visual_max_score'] > args.threshold else 'No Class', 
        axis=1
    )
    resnet_df = resnet_df[resnet_df['visual_predicted_object'] != 'No Class']
    print(f"✅ {len(resnet_df)} predictions above threshold")
    
    # Load ground truth
    print("📋 Loading ground truth...")
    with open(args.ground_truth, 'rb') as file:
        final_SOT = pickle.load(file)
    
    final_SOT = final_SOT.rename(columns={'frame': 'second', 'second': 'frame'})
    final_SOT = final_SOT.groupby(['video', 'tag'])['actual'].max().reset_index()
    print(f"✅ Loaded ground truth for {len(final_SOT)} video-tag pairs")
    
    # Merge predictions with ground truth
    print("🔗 Merging predictions with ground truth...")
    merged = pd.merge(
        resnet_df, 
        final_SOT, 
        left_on=['video', 'visual_predicted_object'], 
        right_on=['video', 'tag'], 
        how='right'
    )
    
    # Classify predictions
    merged['prediction'] = merged['prediction'].fillna(0)
    merged['answerClass'] = merged.apply(
        lambda row: classify_answer(row['actual'], row['prediction']), 
        axis=1
    )
    
    # Calculate overall metrics
    counts = merged['answerClass'].value_counts()
    TP = counts.get('TP', 0)
    FP = counts.get('FP', 0)
    FN = counts.get('FN', 0)
    TN = counts.get('TN', 0)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 Overall Results:")
    print(f"   TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1_score:.4f}")
    
    # Extract false positives
    false_positives = merged[merged['answerClass'] == 'FP'].copy()
    print(f"\n🚨 Found {len(false_positives)} false positives")
    
    if len(false_positives) > 0:
        # Sort by confidence score (highest first)
        false_positives = false_positives.sort_values('visual_max_score', ascending=False)
        
        # Create clean output DataFrame
        fp_output = false_positives[[
            'visual_predicted_object',  # predicted class
            'tag',                     # actual class  
            'visual_max_score',        # confidence score
            'finetuned_embedding',     # embedding vector
            'video',                   # source video
            'frame'                    # source frame
        ]].copy()
        
        # Rename columns for clarity
        fp_output = fp_output.rename(columns={
            'visual_predicted_object': 'predicted_class',
            'tag': 'actual_class',
            'visual_max_score': 'confidence_score'
        })
        
        # Add metadata
        fp_output['model_path'] = args.model_path
        fp_output['threshold'] = args.threshold
        
        # Save results
        print(f"💾 Saving false positives to: {args.output}")
        fp_output.to_pickle(args.output)
        
        # Display summary
        print(f"\n📋 False Positive Summary:")
        print(f"   Total FPs: {len(fp_output)}")
        print(f"   Confidence range: {fp_output['confidence_score'].min():.3f} - {fp_output['confidence_score'].max():.3f}")
        print(f"   Classes predicted: {fp_output['predicted_class'].nunique()}")
        print(f"   Classes actual: {fp_output['actual_class'].nunique()}")
        
        # Show top false positives
        print(f"\n🔝 Top 5 highest confidence false positives:")
        top_fps = fp_output.head()
        for idx, row in top_fps.iterrows():
            print(f"   {row['confidence_score']:.3f}: Predicted '{row['predicted_class']}' but was '{row['actual_class']}' (video: {row['video']}, frame: {row['frame']})")
    
    else:
        print("✅ No false positives found!")
        # Still save empty DataFrame for consistency
        fp_output = pd.DataFrame(columns=[
            'predicted_class', 'actual_class', 'confidence_score', 
            'finetuned_embedding', 'video', 'frame', 'model_path', 'threshold'
        ])
        fp_output.to_pickle(args.output)

if __name__ == "__main__":
    main()


# python results_analysis.py \
#     --model-path contrastive_results/custom/model1/representatives.pkl \
#     --resnet-data /path/to/finetuned_may19.pkl \
#     --ground-truth /path/to/sourceTruth_jeremiah.pkl \
#     --threshold 0.65 \
#     --output high_confidence_fps.pkl