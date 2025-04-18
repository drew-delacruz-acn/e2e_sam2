import os
import argparse
import torch
import numpy as np

# Import from our modules
from config import get_device, DATA_DIR, MODEL_CONFIG, SAM2_CHECKPOINT, SAVE_DIR, LEARNING_RATE, WEIGHT_DECAY, MAX_ITERATIONS, SAVE_INTERVAL
from models.model_utils import load_sam2_model, get_predictor, setup_optimizer, save_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM2 model")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to DAVIS dataset")
    parser.add_argument("--category", type=str, default="bear", help="Category to train on")
    parser.add_argument("--checkpoint", type=str, default=SAM2_CHECKPOINT, help="Path to SAM2 checkpoint")
    parser.add_argument("--model_config", type=str, default=MODEL_CONFIG, help="Path to model config")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Directory to save checkpoints")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--max_iters", type=int, default=MAX_ITERATIONS, help="Maximum iterations")
    parser.add_argument("--save_interval", type=int, default=SAVE_INTERVAL, help="Save interval")
    return parser.parse_args()

def train():
    # Parse arguments
    args = parse_args()
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}")
    data = get_dataset_files(args.data_dir, category=args.category)
    print(f"Found {len(data)} images in the dataset")
    
    # Load model
    print(f"Loading SAM2 model from {args.checkpoint}")

    print(args.model_config)
    print(args.checkpoint)
    sam2_model = load_sam2_model(args.model_config, args.checkpoint, device)
    predictor = get_predictor(sam2_model, device)
    
    # Setup optimizer
    optimizer, scaler = setup_optimizer(predictor.model, lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    mean_iou = 0
    for itr in range(args.max_iters):
        with torch.cuda.amp.autocast():  # mixed precision
            # Load data batch
            image, mask, input_point, input_label = read_batch(data)
            if mask.shape[0] == 0:
                print("Skipping empty batch")
                continue  # ignore empty batches
            
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
            batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
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
            
            # Upscale masks to original image resolution
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            
            # Convert ground truth mask to tensor
            gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
            
            # Apply sigmoid to get probability maps
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            
            # Compute segmentation loss
            seg_loss = compute_segmentation_loss(gt_mask, prd_mask)
            
            # Compute IoU and score loss
            prd_mask_binary = (prd_mask > 0.5)
            iou = compute_iou(gt_mask, prd_mask_binary)
            score_loss = compute_score_loss(prd_scores[:, 0], iou)
            
            # Compute total loss
            loss = compute_total_loss(seg_loss, score_loss)
        
        # Apply backpropagation
        predictor.model.zero_grad()  # empty gradient
        scaler.scale(loss).backward()  # Backpropagate
        scaler.step(optimizer)
        scaler.update()  # Mix precision
        
        # Save model periodically
        if itr % args.save_interval == 0:
            save_model(predictor.model, args.save_dir, itr)
        
        # Update and display metrics
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        print(f"Step {itr}, Loss: {loss.item():.4f}, Accuracy (IOU): {mean_iou:.4f}")
    
    # Save final model
    save_model(predictor.model, args.save_dir, args.max_iters)
    print("Training completed!")

if __name__ == "__main__":
    train() 