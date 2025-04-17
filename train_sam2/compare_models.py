#!/usr/bin/env python
"""
SAM2 Model Comparison Script
Compares performance between original pre-trained SAM2 and fine-tuned model.
"""
import os
import argparse
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare SAM2 models")
    parser.add_argument("--data_dir", type=str, default="../data/davis-2017/DAVIS/",
                      help="Path to dataset")
    parser.add_argument("--splits_dir", type=str, default="./dataset_splits",
                      help="Path to dataset splits")
    parser.add_argument("--pretrained_checkpoint", type=str, 
                      default="../checkpoints/sam2.1_hiera_large.pt",
                      help="Path to original pre-trained SAM2 checkpoint")
    parser.add_argument("--finetuned_checkpoint", type=str,
                      default="./models/sam2_finetuned_best.torch",
                      help="Path to fine-tuned model checkpoint")
    parser.add_argument("--model_cfg", type=str,
                      default="../configs/sam2.1/sam2.1_hiera_l.yaml",
                      help="Path to model config")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                      help="Directory to save comparison results")
    parser.add_argument("--num_samples", type=int, default=-1,
                      help="Number of samples to test. Use -1 for all samples.")
    return parser.parse_args()

def setup_device():
    """Set up computation device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def load_model(checkpoint_path, model_cfg, device):
    """Load SAM2 model from checkpoint"""
    model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor

def calculate_metrics(pred_mask, gt_mask):
    """Calculate various metrics for comparison"""
    # Convert to binary masks
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0

    # Calculate IoU
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0

    # Calculate Dice coefficient
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0

    # Calculate precision and recall
    true_positives = intersection
    false_positives = pred_mask.sum() - true_positives
    false_negatives = gt_mask.sum() - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall)
    }

def evaluate_model(predictor, data, device):
    """Evaluate model on dataset"""
    results = []
    
    for entry in tqdm(data, desc="Evaluating"):
        # Load image and annotation
        img = cv2.imread(entry["image"])[...,::-1]  # Read as RGB
        ann_map = cv2.imread(entry["annotation"])

        # Resize image
        r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                           interpolation=cv2.INTER_NEAREST)

        # Process annotation
        mat_map = ann_map[:,:,0]
        ves_map = ann_map[:,:,2]
        mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1)

        # Get ground truth masks and points
        inds = np.unique(mat_map)[1:]
        sample_results = []

        for ind in inds:
            gt_mask = (mat_map == ind).astype(np.uint8)
            coords = np.argwhere(gt_mask > 0)
            if len(coords) == 0:
                continue
            
            # Choose random point
            yx = coords[np.random.randint(len(coords))]
            point = np.array([[yx[1], yx[0]]])

            # Get prediction
            predictor.set_image(img)
            masks, scores, _ = predictor.predict(
                point_coords=point,
                point_labels=np.ones((1, 1)),
                multimask_output=False
            )

            # Calculate metrics
            metrics = calculate_metrics(masks[0], gt_mask)
            sample_results.append(metrics)

        # Average metrics for this sample
        if sample_results:
            avg_metrics = {
                metric: np.mean([r[metric] for r in sample_results])
                for metric in sample_results[0].keys()
            }
            results.append(avg_metrics)

    # Calculate overall averages
    overall_metrics = {
        metric: np.mean([r[metric] for r in results])
        for metric in results[0].keys()
    }
    
    # Calculate standard deviations
    metric_stds = {
        metric: np.std([r[metric] for r in results])
        for metric in results[0].keys()
    }

    return overall_metrics, metric_stds, results

def main():
    """Main function"""
    args = parse_args()
    device = setup_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    with open(os.path.join(args.splits_dir, "val.json"), 'r') as f:
        data = json.load(f)
    
    # Prepend data_dir to paths
    for item in data:
        item['image'] = os.path.join(args.data_dir, item['image'])
        item['annotation'] = os.path.join(args.data_dir, item['annotation'])

    # Limit number of samples if specified
    if args.num_samples > 0:
        data = data[:args.num_samples]

    # Evaluate pre-trained model
    print("\nEvaluating pre-trained model...")
    pretrained_predictor = load_model(args.pretrained_checkpoint, args.model_cfg, device)
    pretrained_metrics, pretrained_stds, pretrained_results = evaluate_model(
        pretrained_predictor, data, device
    )

    # Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    finetuned_predictor = load_model(args.finetuned_checkpoint, args.model_cfg, device)
    finetuned_metrics, finetuned_stds, finetuned_results = evaluate_model(
        finetuned_predictor, data, device
    )

    # Calculate improvements
    improvements = {
        metric: (finetuned_metrics[metric] - pretrained_metrics[metric]) / pretrained_metrics[metric] * 100
        for metric in pretrained_metrics.keys()
    }

    # Save detailed results
    results = {
        'pretrained': {
            'metrics': pretrained_metrics,
            'std_devs': pretrained_stds,
            'all_results': pretrained_results
        },
        'finetuned': {
            'metrics': finetuned_metrics,
            'std_devs': finetuned_stds,
            'all_results': finetuned_results
        },
        'improvements': improvements
    }

    output_file = os.path.join(args.output_dir, 'comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison
    print("\nModel Comparison Results:")
    print("=" * 80)
    print(f"{'Metric':15} {'Pre-trained':>12} {'Fine-tuned':>12} {'Improvement':>12}")
    print("-" * 80)
    
    for metric in pretrained_metrics.keys():
        print(f"{metric:15} {pretrained_metrics[metric]:>12.4f} {finetuned_metrics[metric]:>12.4f} {improvements[metric]:>11.1f}%")
    
    print("\nDetailed results saved to:", output_file)

if __name__ == "__main__":
    main() 