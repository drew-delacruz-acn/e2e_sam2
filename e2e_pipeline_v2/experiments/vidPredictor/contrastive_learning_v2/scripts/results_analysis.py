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
    
    # 🔍 DEBUG: Let's examine the data structures
    print("\n🔍 DEBUG INFO:")
    print("Sample predictions:")
    print(resnet_df[['video', 'visual_predicted_object', 'visual_max_score', 'owl_label']].head())
    print("\nSample ground truth:")
    print(final_SOT.head())
    
    print(f"\nUnique predicted classes: {sorted(resnet_df['visual_predicted_object'].unique())}")
    print(f"Unique GT classes: {sorted(final_SOT['tag'].unique())}")
    print(f"Classes in both: {set(resnet_df['visual_predicted_object'].unique()) & set(final_SOT['tag'].unique())}")
    
    # 🔍 Let's also check what owl_label contains - this might be the "actual" class!
    print(f"\nUnique owl_label values: {sorted(resnet_df['owl_label'].unique())}")
    
    # 🔍 NEW APPROACH: Use owl_label as the actual class for cross-class confusion
    print("\n🆕 Attempting cross-class confusion analysis using owl_label...")
    
    # Create true cross-class false positives
    cross_class_fps = []
    
    for idx, row in resnet_df.iterrows():
        predicted_class = row['visual_predicted_object']
        actual_class = row['owl_label']  # This should be the true class
        confidence = row['visual_max_score']
        
        if predicted_class != actual_class:
            # This is a true cross-class false positive!
            cross_class_fps.append({
                'predicted_class': predicted_class,
                'actual_class': actual_class,
                'confidence_score': confidence,
                'finetuned_embedding': row['finetuned_embedding'],
                'video': row['video'],
                'frame': row['frame'],
                'fp_type': 'CrossClass',
                'model_path': args.model_path,
                'threshold': args.threshold
            })
    
    if cross_class_fps:
        cross_class_df = pd.DataFrame(cross_class_fps)
        cross_class_df = cross_class_df.sort_values('confidence_score', ascending=False)
        
        print(f"\n🎯 Found {len(cross_class_df)} cross-class false positives!")
        print("Top 10 cross-class confusions:")
        for idx, row in cross_class_df.head(10).iterrows():
            print(f"   {row['confidence_score']:.3f}: Predicted '{row['predicted_class']}' but actually '{row['actual_class']}' (video: {row['video']})")
        
        # Save cross-class FPs
        output_file = args.output.replace('.pkl', '_crossclass.pkl')
        cross_class_df.to_pickle(output_file)
        print(f"\n💾 Saved cross-class false positives to: {output_file}")
        
    else:
        print("\n❌ No cross-class false positives found using owl_label")

    # Continue with original analysis for completeness...
    print("\n" + "="*50)
    print("ORIGINAL ANALYSIS (presence/absence):")
    
    # Merge predictions with ground truth
    print("🔗 Merging predictions with ground truth...")
    
    # First, let's get all predictions and mark which ones have ground truth
    all_predictions = resnet_df.copy()
    
    # Create a comprehensive ground truth lookup
    gt_lookup = final_SOT.set_index(['video', 'tag'])['actual'].to_dict()
    
    # For each prediction, find the ground truth
    def get_ground_truth_and_classify(row):
        video = row['video']
        predicted_class = row['visual_predicted_object']
        
        # Check if this video-class combination exists in ground truth
        gt_key = (video, predicted_class)
        if gt_key in gt_lookup:
            actual = gt_lookup[gt_key]
            prediction = 1  # Model made a prediction
            
            # Classify
            if actual == 1 and prediction == 1:
                return pd.Series({'actual': actual, 'prediction': prediction, 'answerClass': 'TP', 'has_gt': True})
            elif actual == 0 and prediction == 1:
                return pd.Series({'actual': actual, 'prediction': prediction, 'answerClass': 'FP', 'has_gt': True})
            else:
                return pd.Series({'actual': actual, 'prediction': prediction, 'answerClass': 'Other', 'has_gt': True})
        else:
            # Model predicted a class that doesn't exist in ground truth for this video
            # This is also a type of False Positive (hallucination)
            return pd.Series({'actual': 0, 'prediction': 1, 'answerClass': 'FP_NoGT', 'has_gt': False})
    
    classification_results = all_predictions.apply(get_ground_truth_and_classify, axis=1)
    merged = pd.concat([all_predictions, classification_results], axis=1)
    
    # Also need to find False Negatives (ground truth classes that were never predicted)
    # For now, let's focus on the predictions we have
    
    # Calculate overall metrics
    counts = merged['answerClass'].value_counts()
    TP = counts.get('TP', 0)
    FP = counts.get('FP', 0) + counts.get('FP_NoGT', 0)  # Combine both types of FP
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
    
    # Extract false positives (both types)
    false_positives = merged[merged['answerClass'].isin(['FP', 'FP_NoGT'])].copy()
    fp_with_gt = merged[merged['answerClass'] == 'FP']
    fp_no_gt = merged[merged['answerClass'] == 'FP_NoGT']
    
    print(f"\n🚨 Found {len(false_positives)} total false positives:")
    print(f"   - {len(fp_with_gt)} FPs: Model predicted class present but ground truth says absent")
    print(f"   - {len(fp_no_gt)} FP_NoGT: Model predicted class not in ground truth for this video")
    
    if len(false_positives) > 0:
        # Sort by confidence score (highest first)
        false_positives = false_positives.sort_values('visual_max_score', ascending=False)
        
        # Create clean output DataFrame
        fp_output = false_positives[[
            'visual_predicted_object',  # predicted class
            'visual_max_score',        # confidence score
            'finetuned_embedding',     # embedding vector
            'video',                   # source video
            'frame',                   # source frame
            'answerClass',             # type of FP
            'has_gt'                   # whether ground truth exists
        ]].copy()
        
        # Add actual class info
        fp_output['actual_class'] = fp_output.apply(
            lambda row: row['visual_predicted_object'] if row['answerClass'] == 'FP' else 'Not in GT',
            axis=1
        )
        
        # Rename columns for clarity
        fp_output = fp_output.rename(columns={
            'visual_predicted_object': 'predicted_class',
            'visual_max_score': 'confidence_score',
            'answerClass': 'fp_type'
        })
        
        # Add metadata
        fp_output['model_path'] = args.model_path
        fp_output['threshold'] = args.threshold
        
        # Save results
        print(f"💾 Saving presence/absence false positives to: {args.output}")
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
            if row['fp_type'] == 'FP':
                print(f"   {row['confidence_score']:.3f}: Predicted '{row['predicted_class']}' present but GT says absent (video: {row['video']}, frame: {row['frame']})")
            else:
                print(f"   {row['confidence_score']:.3f}: Predicted '{row['predicted_class']}' but class not in GT for this video (video: {row['video']}, frame: {row['frame']})")
    
    else:
        print("✅ No presence/absence false positives found!")
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