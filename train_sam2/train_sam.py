#!/usr/bin/env python
"""
SAM2 Model Fine-tuning Script
This script fine-tunes a SAM2 model on a specific dataset.
"""
import os
import argparse
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

def load_dataset(data_dir):
    """Load dataset file paths"""
    data = []
    # Go over all files in the dataset
    for name in os.listdir(os.path.join(data_dir, "JPEGImages/480p/bear/")):
        data.append({
            "image": os.path.join(data_dir, "JPEGImages/480p/bear/", name),
            "annotation": os.path.join(data_dir, "Annotations/480p/bear/", name[:-4] + ".png")
        })
    return data

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

def main():
    """Main training function"""
    args = parse_args()
    device = setup_device()
    
    # Load dataset
    data = load_dataset(args.data_dir)
    print(f"Loaded {len(data)} images from dataset")

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
    
    # Training loop
    mean_iou = 0
    for itr in range(args.max_iterations):
        with torch.cuda.amp.autocast():  # Cast to mixed precision
            # Load data batch
            image, mask, input_point, input_label = read_batch(data)
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

            # Save model periodically
            if itr % args.save_interval == 0:
                torch.save(predictor.model.state_dict(), args.output_model)
                print(f"Saved model at iteration {itr}")

            # Display results
            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print(f"Step {itr}, Accuracy(IOU)={mean_iou:.4f}")

if __name__ == "__main__":
    main()
