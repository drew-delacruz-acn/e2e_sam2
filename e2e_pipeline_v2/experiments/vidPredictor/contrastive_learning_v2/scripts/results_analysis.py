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
    
    # Handle column name variations - owl_label is the TRUE class
    if 'owl_label' in resnet_df.columns:
        true_class_col = 'owl_label'
    elif 'class' in resnet_df.columns:
        true_class_col = 'class'
    else:
        raise ValueError("No 'owl_label' or 'class' column found in ResNet data")
    
    # Select required columns
    required_cols = ['video', 'frame', true_class_col, 'finetuned_embedding']
    resnet_df = resnet_df[required_cols].copy()
    resnet_df = resnet_df.rename(columns={true_class_col: 'true_class'})
    
    print(f"✅ Loaded {len(resnet_df)} ResNet samples")
    print(f"📋 Using '{true_class_col}' as true class column")
    print(f"📊 Data structure:")
    print(f"   - video: Video filename")
    print(f"   - frame: Frame number") 
    print(f"   - true_class: Actual class (from {true_class_col})")
    print(f"   - finetuned_embedding: Feature vector")
    
    # Load model representatives
    print("\n🤖 Loading model representatives...")
    with open(args.model_path, 'rb') as file:
        defObjects = pickle.load(file)
    
    # Handle different formats for representatives
    if isinstance(defObjects, dict):
        if 'finetuned_embedding' in defObjects:
            class_emb_matrix = list(defObjects['finetuned_embedding'])
            objs = list(defObjects['class'])
        elif 'representative_embedding' in defObjects:
            # Handle prototypes.pkl format
            class_emb_matrix = list(defObjects['representative_embedding'])
            objs = list(defObjects['class'])
        else:
            # Assume defObjects is a dict of class -> embedding
            objs = list(defObjects.keys())
            class_emb_matrix = list(defObjects.values())
    else:
        # Assume defObjects is a DataFrame
        if 'finetuned_embedding' in defObjects.columns:
            class_emb_matrix = list(defObjects['finetuned_embedding'])
            objs = list(defObjects['class'])
        elif 'representative_embedding' in defObjects.columns:
            class_emb_matrix = list(defObjects['representative_embedding'])
            objs = list(defObjects['class'])
        else:
            raise ValueError("Could not find embedding column in representatives data")
    
    print(f"✅ Loaded representatives for {len(objs)} classes")
    print(f"🏷️  Model classes: {objs}")
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    visual_preds = resnet_df.apply(
        lambda row: visual_prediction_from_text_filtered(row, class_emb_matrix, objs), 
        axis=1
    )
    resnet_df[['visual_predicted_object', 'visual_max_score']] = visual_preds
    
    # Convert frame to int (handle any format issues)
    resnet_df['frame'] = resnet_df['frame'].astype(int)
    
    print(f"✅ Generated {len(resnet_df)} initial predictions")
    print(f"📊 Prediction structure:")
    print(f"   - visual_predicted_object: Model's predicted class")
    print(f"   - visual_max_score: Confidence score")
    
    # Keep highest scoring prediction per video-object pair (before threshold filtering)
    print("\n🔄 Keeping highest scoring prediction per video-class pair...")
    idx = resnet_df.groupby(['video', 'visual_predicted_object'])['visual_max_score'].idxmax()
    resnet_df_filtered = resnet_df.loc[idx].reset_index(drop=True)
    print(f"✅ Reduced to {len(resnet_df_filtered)} unique video-class predictions")
    
    # Apply threshold filtering
    print(f"\n🎯 Applying threshold: {args.threshold}")
    above_threshold = resnet_df_filtered['visual_max_score'] > args.threshold
    predictions_above_threshold = resnet_df_filtered[above_threshold].copy()
    print(f"✅ {len(predictions_above_threshold)} predictions above threshold")
    
    # Load ground truth
    print("\n📋 Loading ground truth...")
    with open(args.ground_truth, 'rb') as file:
        final_SOT = pickle.load(file)
    
    # Fix column naming issues based on the logs
    print("🔧 Fixing ground truth column structure...")
    print(f"📊 Ground truth structure:")
    print(f"   - video: Video filename")
    print(f"   - tag: Class name")
    print(f"   - actual: Binary presence (1=present, 0=absent)")
    
    if 'second' in final_SOT.columns and 'frame' in final_SOT.columns:
        # Keep the existing structure, just rename for clarity
        final_SOT = final_SOT.rename(columns={'frame': 'time_frame', 'second': 'frame_number'})
    
    # Group by video and tag to get max actual value
    final_SOT_grouped = final_SOT.groupby(['video', 'tag'])['actual'].max().reset_index()
    print(f"✅ Loaded ground truth for {len(final_SOT_grouped)} video-tag pairs")
    
    # 🔍 DEBUG: Let's examine the data structures
    print("\n" + "="*60)
    print("🔍 DATA STRUCTURE ANALYSIS:")
    print("="*60)
    
    print("\nSample predictions (above threshold):")
    if len(predictions_above_threshold) > 0:
        sample_preds = predictions_above_threshold[['video', 'true_class', 'visual_predicted_object', 'visual_max_score']].head()
        print(sample_preds.to_string(index=False))
    else:
        print("No predictions above threshold!")
    
    print("\nSample ground truth:")
    sample_gt = final_SOT_grouped[['video', 'tag', 'actual']].head()
    print(sample_gt.to_string(index=False))
    
    # Analyze class distributions
    true_classes = set(resnet_df_filtered['true_class'].unique())
    pred_classes = set(predictions_above_threshold['visual_predicted_object'].unique()) if len(predictions_above_threshold) > 0 else set()
    gt_classes = set(final_SOT_grouped['tag'].unique())
    model_classes = set(objs)
    
    print(f"\n📊 CLASS ANALYSIS:")
    print(f"   True classes in data ({len(true_classes)}): {sorted(list(true_classes)[:5])}{'...' if len(true_classes) > 5 else ''}")
    print(f"   Model classes ({len(model_classes)}): {sorted(list(model_classes)[:5])}{'...' if len(model_classes) > 5 else ''}")
    print(f"   Predicted classes above threshold ({len(pred_classes)}): {sorted(list(pred_classes)[:5])}{'...' if len(pred_classes) > 5 else ''}")
    print(f"   Ground truth classes ({len(gt_classes)}): {sorted(list(gt_classes)[:5])}{'...' if len(gt_classes) > 5 else ''}")
    
    print(f"\n🔄 CLASS OVERLAPS:")
    print(f"   True ∩ Model: {len(true_classes & model_classes)}/{len(true_classes)} true classes in model")
    print(f"   Pred ∩ GT: {len(pred_classes & gt_classes)}/{len(pred_classes)} predicted classes in GT")
    print(f"   Model ∩ GT: {len(model_classes & gt_classes)}/{len(model_classes)} model classes in GT")
    
    # 🆕 CROSS-CLASS CONFUSION ANALYSIS
    print("\n" + "="*60)
    print("🆕 CROSS-CLASS CONFUSION ANALYSIS:")
    print("="*60)
    print("Comparing: visual_predicted_object vs true_class (owl_label)")
    
    # Use all predictions (not just above threshold) for cross-class analysis
    cross_class_results = []
    correct_predictions = 0
    total_predictions = 0
    
    for idx, row in resnet_df_filtered.iterrows():
        predicted_class = row['visual_predicted_object']
        true_class = row['true_class']
        confidence = row['visual_max_score']
        
        total_predictions += 1
        
        if predicted_class == true_class:
            correct_predictions += 1
            result_type = 'Correct'
        else:
            result_type = 'CrossClass_Error'
            cross_class_results.append({
                'predicted_class': predicted_class,
                'actual_class': true_class,
                'confidence_score': confidence,
                'finetuned_embedding': row['finetuned_embedding'],
                'video': row['video'],
                'frame': row['frame'],
                'error_type': 'CrossClass',
                'above_threshold': confidence > args.threshold,
                'model_path': args.model_path,
                'threshold': args.threshold
            })
    
    cross_class_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"\n📊 Cross-Class Classification Results:")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Cross-class errors: {len(cross_class_results)}")
    print(f"   Cross-class accuracy: {cross_class_accuracy:.4f}")
    
    if cross_class_results:
        cross_class_df = pd.DataFrame(cross_class_results)
        cross_class_df = cross_class_df.sort_values('confidence_score', ascending=False)
        
        # Analyze by threshold
        above_thresh_errors = cross_class_df[cross_class_df['above_threshold']]
        print(f"   Cross-class errors above threshold: {len(above_thresh_errors)}")
        
        print(f"\n🔝 Top 10 cross-class confusions:")
        for idx, row in cross_class_df.head(10).iterrows():
            thresh_marker = "✓" if row['above_threshold'] else "✗"
            print(f"   {thresh_marker} {row['confidence_score']:.3f}: '{row['predicted_class']}' ← '{row['actual_class']}' (video: {row['video']})")
        
        # Save cross-class errors
        output_file = args.output.replace('.pkl', '_crossclass.pkl')
        cross_class_df.to_pickle(output_file)
        print(f"\n💾 Saved cross-class errors to: {output_file}")
        
        # Analyze confusion patterns
        confusion_counts = cross_class_df.groupby(['actual_class', 'predicted_class']).size().reset_index(name='count')
        confusion_counts = confusion_counts.sort_values('count', ascending=False)
        
        print(f"\n🔄 Top confusion patterns:")
        for idx, row in confusion_counts.head(5).iterrows():
            print(f"   {row['count']}x: '{row['actual_class']}' → '{row['predicted_class']}'")
    
    # 🎯 PRESENCE/ABSENCE ANALYSIS
    print("\n" + "="*60)
    print("🎯 PRESENCE/ABSENCE ANALYSIS:")
    print("="*60)
    print("Comparing: model predictions vs ground truth presence/absence")
    
    # Create comprehensive evaluation
    gt_lookup = final_SOT_grouped.set_index(['video', 'tag'])['actual'].to_dict()
    
    # Get all unique video-class combinations that should be evaluated
    all_videos = set(final_SOT_grouped['video'].unique())
    all_classes = set(final_SOT_grouped['tag'].unique())
    
    print(f"\n📊 Evaluation scope:")
    print(f"   Videos: {len(all_videos)}")
    print(f"   Classes: {len(all_classes)}")
    print(f"   Total combinations: {len(all_videos) * len(all_classes)}")
    
    # Create comprehensive results
    results = []
    
    # Get predictions lookup
    pred_lookup = {}
    for idx, row in predictions_above_threshold.iterrows():
        key = (row['video'], row['visual_predicted_object'])
        if key not in pred_lookup or row['visual_max_score'] > pred_lookup[key]['score']:
            pred_lookup[key] = {
                'score': row['visual_max_score'],
                'frame': row['frame']
            }
    
    # Evaluate all video-class combinations
    for video in all_videos:
        for class_name in all_classes:
            gt_key = (video, class_name)
            pred_key = (video, class_name)
            
            # Get ground truth
            actual = gt_lookup.get(gt_key, 0)
            
            # Get prediction
            prediction = 1 if pred_key in pred_lookup else 0
            confidence = pred_lookup[pred_key]['score'] if pred_key in pred_lookup else 0.0
            
            # Classify
            if actual == 1 and prediction == 1:
                result_type = 'TP'
            elif actual == 0 and prediction == 1:
                result_type = 'FP'
            elif actual == 1 and prediction == 0:
                result_type = 'FN'
            else:
                result_type = 'TN'
            
            results.append({
                'video': video,
                'class': class_name,
                'actual': actual,
                'prediction': prediction,
                'confidence': confidence,
                'result_type': result_type
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    counts = results_df['result_type'].value_counts()
    TP = counts.get('TP', 0)
    FP = counts.get('FP', 0)
    FN = counts.get('FN', 0)
    TN = counts.get('TN', 0)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    
    print(f"\n📊 Presence/Absence Results:")
    print(f"   TP: {TP}")
    print(f"   FP: {FP}")
    print(f"   FN: {FN}")
    print(f"   TN: {TN}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1_score:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    
    # Extract and save detailed results
    false_positives = results_df[results_df['result_type'] == 'FP']
    false_negatives = results_df[results_df['result_type'] == 'FN']
    
    print(f"\n🚨 Error Analysis:")
    print(f"   False Positives: {len(false_positives)}")
    print(f"   False Negatives: {len(false_negatives)}")
    
    if len(false_positives) > 0:
        print(f"\n🔝 Top False Positives (by confidence):")
        fp_sorted = false_positives.sort_values('confidence', ascending=False)
        for idx, row in fp_sorted.head(5).iterrows():
            print(f"   {row['confidence']:.3f}: '{row['class']}' in {row['video']}")
    
    if len(false_negatives) > 0:
        print(f"\n❌ Sample False Negatives:")
        for idx, row in false_negatives.head(5).iterrows():
            print(f"   Missed: '{row['class']}' in {row['video']}")
    
    # Save individual dataframes as CSV
    base_output = args.output.replace('.pkl', '')
    
    if len(false_positives) > 0:
        fp_file = f"{base_output}_false_positives.csv"
        false_positives.to_csv(fp_file, index=False)
        print(f"\n💾 Saved False Positives CSV to: {fp_file}")
        print(f"   Columns: {list(false_positives.columns)}")
    
    if len(false_negatives) > 0:
        fn_file = f"{base_output}_false_negatives.csv"
        false_negatives.to_csv(fn_file, index=False)
        print(f"💾 Saved False Negatives CSV to: {fn_file}")
        print(f"   Columns: {list(false_negatives.columns)}")
    
    # Also save all results as CSV for easy inspection
    all_results_file = f"{base_output}_all_results.csv"
    results_df.to_csv(all_results_file, index=False)
    print(f"💾 Saved All Results CSV to: {all_results_file}")
    
    # Save comprehensive results (keep the pickle for programmatic access)
    output_data = {
        'metrics': {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'precision': precision, 'recall': recall, 
            'f1_score': f1_score, 'accuracy': accuracy,
            'cross_class_accuracy': cross_class_accuracy
        },
        'false_positives': false_positives.to_dict('records'),
        'false_negatives': false_negatives.to_dict('records'),
        'class_analysis': {
            'true_classes': sorted(list(true_classes)),
            'model_classes': sorted(list(model_classes)),
            'predicted_classes': sorted(list(pred_classes)),
            'gt_classes': sorted(list(gt_classes)),
            'overlaps': {
                'true_model': len(true_classes & model_classes),
                'pred_gt': len(pred_classes & gt_classes),
                'model_gt': len(model_classes & gt_classes)
            }
        },
        'config': {
            'model_path': args.model_path,
            'threshold': args.threshold,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'true_class_column': true_class_col
        }
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"💾 Saved comprehensive analysis (pickle) to: {args.output}")
    
    # Summary
    print(f"\n" + "="*60)
    print("📋 SUMMARY:")
    print("="*60)
    print(f"Cross-class accuracy: {cross_class_accuracy:.4f}")
    print(f"Presence/absence F1: {f1_score:.4f}")
    print(f"Total errors: {len(cross_class_results) + FP + FN}")
    print(f"Data columns used:")
    print(f"  - True class: {true_class_col}")
    print(f"  - Predicted class: visual_predicted_object")
    print(f"  - Ground truth: final_SOT['tag'] and ['actual']")
    print(f"Analysis saved to: {args.output}")

if __name__ == "__main__":
    main()


# python results_analysis.py \
#     --model-path contrastive_results/custom/model1/representatives.pkl \
#     --resnet-data /path/to/finetuned_may19.pkl \
#     --ground-truth /path/to/sourceTruth_jeremiah.pkl \
#     --threshold 0.65 \
#     --output high_confidence_fps.pkl