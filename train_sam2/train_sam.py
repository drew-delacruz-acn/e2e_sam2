#!/usr/bin/env python
"""
SAM2 Model Fine-tuning Script
This script fine-tunes a SAM2 model on a specific dataset.
"""
import os
import argparse
import json
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# If using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 model")
    parser.add_argument("--data_dir", type=str, default="../../../data/davis-2017/DAVIS/",
                      help="Path to dataset")
    parser.add_argument("--splits_dir", type=str, default="./dataset_splits",
                      help="Path to dataset splits")
    parser.add_argument("--sam2_checkpoint", type=str, default="../checkpoints/sam2.1_hiera_large.pt",
                      help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                      help="Path to model config")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=4e-5,
                      help="Weight decay for optimizer")
    parser.add_argument("--max_iterations", type=int, default=100,
                      help="Number of training iterations")
    parser.add_argument("--save_interval", type=int, default=100,
                      help="Save model every N iterations")
    parser.add_argument("--val_interval", type=int, default=10,
                      help="Validate model every N iterations")
    parser.add_argument("--output_model", type=str, default="model.torch",
                      help="Output model path")
    return parser.parse_args()

def setup_device():
    """Set up computation device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        # Use bfloat16 for the entire script
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # Turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    
    return device

def load_dataset_splits(data_dir, splits_dir):
    """Load dataset splits"""
    dataset = {}
    
    # Load train and validation splits
    for split in ['train', 'val']:
        split_file = os.path.join(splits_dir, f"{split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Prepend the data_dir to each path
        for item in split_data:
            item['image'] = os.path.join(data_dir, item['image'])
            item['annotation'] = os.path.join(data_dir, item['annotation'])
        
        dataset[split] = split_data
        print(f"Loaded {len(split_data)} samples for {split} split")
    
    return dataset

def read_batch(data):
    """Read random image and its annotation from the dataset"""
    # Select image
    entry = data[np.random.randint(len(data))]  # Choose random entry
    img = cv2.imread(entry["image"])[...,::-1]  # Read image as RGB
    ann_map = cv2.imread(entry["annotation"])  # Read annotation

    # Resize image
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # Scaling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), 
                        interpolation=cv2.INTER_NEAREST)

    # Merge vessels and materials annotations
    mat_map = ann_map[:,:,0]  # Material annotation map
    ves_map = ann_map[:,:,2]  # Vessel annotation map
    mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1)  # Merge maps

    # Get binary masks and points
    inds = np.unique(mat_map)[1:]  # Load all indices
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # Make binary mask for index ind
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # Get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))])  # Choose random point
        points.append([[yx[1], yx[0]]])
    
    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def evaluate(predictor, device, val_data, num_samples=5):
    """Evaluate model on validation set"""
    total_iou = 0.0
    total_masks = 0
    
    for _ in range(num_samples):
        # Load data batch
        image, gt_mask, input_point, input_label = read_batch(val_data)
        if gt_mask.shape[0] == 0:
            continue
        
        # Apply SAM image encoder to the image
        predictor.set_image(image)

        # Process each mask separately
        for i in range(gt_mask.shape[0]):
            # Get prediction for this mask
            masks, scores, _ = predictor.predict(
                point_coords=input_point[i:i+1],
                point_labels=input_label[i:i+1],
                multimask_output=False
            )
            
            # Calculate IoU
            pred_mask = masks[0]  # First mask prediction
            target_mask = gt_mask[i]
            
            # Convert to tensors
            pred_tensor = torch.tensor(pred_mask.astype(np.float32)).to(device)
            target_tensor = torch.tensor(target_mask.astype(np.float32)).to(device)
            
            # Calculate intersection and union
            intersection = (pred_tensor * target_tensor).sum()
            union = pred_tensor.sum() + target_tensor.sum() - intersection
            iou = (intersection / union).item() if union > 0 else 0.0
            
            total_iou += iou
            total_masks += 1
    
    # Calculate average IoU
    avg_iou = total_iou / total_masks if total_masks > 0 else 0.0
    return avg_iou

def main():
    """Main training function"""
    args = parse_args()
    device = setup_device()
    
    # Load dataset splits
    dataset = load_dataset_splits(args.data_dir, args.splits_dir)
    print(f"Loaded dataset with {len(dataset['train'])} training and {len(dataset['val'])} validation samples")

    # Load model
    print(f"Current working directory: {os.getcwd()}")
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Best validation score
    best_val_iou = 0.0
    
    # Training loop
    mean_iou = 0
    for itr in range(args.max_iterations):
        with torch.cuda.amp.autocast():  # Cast to mixed precision
            # Load data batch from training set
            image, mask, input_point, input_label = read_batch(dataset['train'])
            if mask.shape[0] == 0:
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

            # Validation
            if itr % args.val_interval == 0:
                predictor.model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    val_iou = evaluate(predictor, device, dataset['val'])
                predictor.model.train()  # Set model back to training mode
                
                print(f"Validation IoU at iteration {itr}: {val_iou:.4f}")
                
                # Save the best model
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    best_model_path = args.output_model.replace('.torch', '_best.torch')
                    torch.save(predictor.model.state_dict(), best_model_path)
                    print(f"Saved best model with IoU {best_val_iou:.4f} to {best_model_path}")

            # Save model periodically
            if itr % args.save_interval == 0:
                torch.save(predictor.model.state_dict(), args.output_model)
                print(f"Saved model at iteration {itr}")

            # Display results
            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print(f"Step {itr}, Training Accuracy(IOU)={mean_iou:.4f}")
    
    # Save final model
    torch.save(predictor.model.state_dict(), args.output_model)
    print(f"Training completed. Final model saved to {args.output_model}")
    print(f"Best validation IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    main()