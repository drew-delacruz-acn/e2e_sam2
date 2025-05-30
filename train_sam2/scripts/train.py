#!/usr/bin/env python
"""
SAM2 Model Fine-tuning Script
This script fine-tunes a SAM2 model on a specific dataset.
"""
import os
import sys
import argparse
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_dataset_splits, read_batch
from utils.model_utils import setup_device, load_sam2_model, setup_optimizer
from utils.metrics import MetricsTracker, evaluate_model
from config.default_config import *

# If using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 model")
    parser.add_argument("--data_dir", type=str, default="../data/davis-2017/DAVIS/",
                      help="Path to dataset")
    parser.add_argument("--splits_dir", type=str, default="../dataset_splits",
                      help="Path to dataset splits")
    parser.add_argument("--sam2_checkpoint", type=str, default="../checkpoints/sam2.1_hiera_large.pt",
                      help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, default="../configs/sam2.1/sam2.1_hiera_l.yaml",
                      help="Path to model config")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=4e-5,
                      help="Weight decay for optimizer")
    parser.add_argument("--max_iterations", type=int, default=10000,
                      help="Number of training iterations")
    parser.add_argument("--save_interval", type=int, default=100,
                      help="Save model every N iterations")
    parser.add_argument("--val_interval", type=int, default=10,
                      help="Validate model every N iterations")
    parser.add_argument("--metrics_log", type=str, default="../outputs/logs/training_metrics.json",
                      help="Path to save training metrics")
    parser.add_argument("--gap_threshold", type=float, default=1.0,
                      help="Threshold for warning about train-val IoU gap")
    parser.add_argument("--gap_increase_threshold", type=float, default=0.05,
                      help="Threshold for warning about increasing train-val gap")
    parser.add_argument("--output_model", type=str, default="../outputs/models/model.torch",
                      help="Output model path")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                      help="Number of consecutive warnings before stopping")
    parser.add_argument("--max_gap", type=float, default=0.3,
                      help="Maximum allowed gap between training and validation IoU")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    device = setup_device()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_log), exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        args.metrics_log,
        args.gap_threshold,
        args.gap_increase_threshold,
        args.early_stopping_patience,
        args.max_gap
    )
    
    # Load dataset splits
    dataset = load_dataset_splits(args.data_dir, args.splits_dir)
    print(f"Loaded dataset with {len(dataset['train'])} training and {len(dataset['val'])} validation samples")

    # Load model
    print(f"Current working directory: {os.getcwd()}")
    print(f'Loading model from {args.sam2_checkpoint}')
    print(f'Loading model config from {args.model_cfg}')
    print(f'Using device: {device}')
    
    predictor = load_sam2_model(args.model_cfg, args.sam2_checkpoint, device)

    # Set up optimizer
    optimizer, scaler = setup_optimizer(
        predictor.model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    mean_iou = 0
    for itr in range(args.max_iterations):
        with torch.cuda.amp.autocast():  # Cast to mixed precision
            # Load data batch from training set
            image, mask, input_point, input_label = read_batch(dataset['train'])
            if mask.shape[0] == 0:
                print("Empty batch, skipping...")
                continue  # Ignore empty batches
            
            # Apply SAM image encoder to the image
            predictor.set_image(image)

            # Prompt encoding
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )

            # Mask decoder
            batched_mode = unnorm_coords.shape[0] > 1  # Multi-object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            
            # Upscale the masks to the original image resolution
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            # Segmentation Loss calculation
            gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
            prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
            
            # Cross entropy loss
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - 
                        (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            # Score loss calculation (intersection over union) IOU
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
            iou = inter / union
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            
            # Combine losses
            loss = seg_loss + score_loss * 0.05

            # Apply back propagation
            predictor.model.zero_grad()  # Empty gradient
            scaler.scale(loss).backward()  # Backpropagate
            scaler.step(optimizer)
            scaler.update()  # Mix precision

            # Calculate training IoU
            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            # Validation and metrics tracking
            if itr % args.val_interval == 0:
                predictor.model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    val_iou = evaluate_model(predictor, device, dataset['val'])
                predictor.model.train()  # Set model back to training mode
                
                # Track metrics and check for warnings/stopping conditions
                warnings, should_stop, stop_reason = metrics_tracker.update(
                    itr, 
                    mean_iou, 
                    val_iou,
                    predictor.model,  # Pass the model
                    args.output_model  # Pass the save path
                )
                
                # Print warnings and metrics
                for warning in warnings:
                    print(f"\n{'-'*80}\n{warning}\n{'-'*80}")
                
                print(f"\nIteration {itr}:")
                print(f"  Training IoU: {mean_iou:.4f}")
                print(f"  Validation IoU: {val_iou:.4f}")
                print(f"  Gap: {(mean_iou - val_iou):.4f}")
                
                # Check if training should stop
                if should_stop:
                    print(f"\n{'='*80}\n{stop_reason}\n{'='*80}")
                    print(f"Best validation IoU achieved: {metrics_tracker.best_val_iou:.4f}")
                    print(f"Best model saved at: {metrics_tracker.best_model_path}")
                    return  # Exit training
                
                # Save model periodically
                if itr % args.save_interval == 0:
                    torch.save(predictor.model.state_dict(), args.output_model)
                    print(f"Saved model at iteration {itr}")
    
    # Save final model
    torch.save(predictor.model.state_dict(), args.output_model)
    print(f"\nTraining completed:")
    print(f"  Final model saved to: {args.output_model}")
    print(f"  Best validation IoU: {metrics_tracker.best_val_iou:.4f}")
    print(f"  Training metrics saved to: {args.metrics_log}")

if __name__ == "__main__":
    main()
