#!/usr/bin/env python
"""
SAM2 Model Comparison Script
Compares performance between original pre-trained SAM2 and fine-tuned model.
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import cv2
import logging
import traceback
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def setup_logging(output_dir):
    """Set up logging configuration"""
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'comparison_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

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
                      default="./model.torch",
                      help="Path to fine-tuned model checkpoint")
    parser.add_argument("--model_cfg", type=str,
                      default="../configs/sam2.1/sam2.1_hiera_l.yaml",
                      help="Path to model config")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                      help="Directory to save comparison results")
    parser.add_argument("--num_samples", type=int, default=-1,
                      help="Number of samples to test. Use -1 for all samples.")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with additional logging")
    return parser.parse_args()

def setup_device():
    """Set up computation device (CUDA, MPS, or CPU)"""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
        return device
    except Exception as e:
        logging.error(f"Error setting up device: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def load_model(checkpoint_path, model_cfg, device):
    """Load SAM2 model from checkpoint"""
    try:
        logging.info(f"Loading model from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(model_cfg):
            raise FileNotFoundError(f"Model config not found: {model_cfg}")

        # Determine if it's likely a state dict based on file extension
        is_state_dict = checkpoint_path.endswith('.torch') or checkpoint_path.endswith('.pt')
        
        try:
            if is_state_dict:
                # Load from state dict (fine-tuned model)
                logging.info(f"Loading as state dict: {checkpoint_path}")
                # First load the base model with architecture
                model = build_sam2(model_cfg, None, device=device)
                # Then load the state dict
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        model.load_state_dict(checkpoint["model"])
                    elif "state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["state_dict"])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Load regular checkpoint
                logging.info(f"Loading as regular checkpoint: {checkpoint_path}")
                model = build_sam2(model_cfg, checkpoint_path, device=device)
        except Exception as first_error:
            logging.warning(f"First loading attempt failed: {str(first_error)}")
            logging.info("Trying alternative loading method...")
            
            # Try the alternative approach
            model = build_sam2(model_cfg, None, device=device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Try different ways the checkpoint might be structured
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            logging.info("Alternative loading method successful!")

        predictor = SAM2ImagePredictor(model)
        logging.info("Model loaded successfully")
        return predictor
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def calculate_metrics(pred_mask, gt_mask):
    """Calculate various metrics for comparison"""
    try:
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

        metrics = {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        if any(not 0 <= v <= 1 for v in metrics.values()):
            logging.warning(f"Unusual metric values detected: {metrics}")
        
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def evaluate_model(predictor, data, device, debug=False):
    """Evaluate model on dataset"""
    results = []
    errors = []
    
    for idx, entry in enumerate(tqdm(data, desc="Evaluating")):
        try:
            # Verify file existence
            if not os.path.exists(entry["image"]):
                raise FileNotFoundError(f"Image not found: {entry['image']}")
            if not os.path.exists(entry["annotation"]):
                raise FileNotFoundError(f"Annotation not found: {entry['annotation']}")

            # Load image and annotation
            img = cv2.imread(entry["image"])
            if img is None:
                raise ValueError(f"Failed to load image: {entry['image']}")
            img = img[...,::-1]  # Convert to RGB
            
            ann_map = cv2.imread(entry["annotation"])
            if ann_map is None:
                raise ValueError(f"Failed to load annotation: {entry['annotation']}")

            if debug:
                logging.debug(f"Processing sample {idx}:")
                logging.debug(f"  Image shape: {img.shape}")
                logging.debug(f"  Annotation shape: {ann_map.shape}")

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
            if len(inds) == 0:
                logging.warning(f"No objects found in annotation: {entry['annotation']}")
                continue

            sample_results = []
            for obj_idx, ind in enumerate(inds):
                try:
                    gt_mask = (mat_map == ind).astype(np.uint8)
                    coords = np.argwhere(gt_mask > 0)
                    if len(coords) == 0:
                        logging.warning(f"Empty mask found for object {obj_idx} in {entry['annotation']}")
                        continue
                    
                    # Choose random point
                    yx = coords[np.random.randint(len(coords))]
                    point = np.array([[yx[1], yx[0]]])

                    if debug:
                        logging.debug(f"  Object {obj_idx}: point={point}, mask_size={gt_mask.sum()}")

                    # Get prediction
                    predictor.set_image(img)
                    masks, scores, _ = predictor.predict(
                        point_coords=point,
                        point_labels=np.ones(1),
                        multimask_output=False
                    )

                    # Calculate metrics
                    metrics = calculate_metrics(masks[0], gt_mask)
                    sample_results.append(metrics)

                    if debug:
                        logging.debug(f"  Object {obj_idx} metrics: {metrics}")

                except Exception as e:
                    error_msg = f"Error processing object {obj_idx} in {entry['annotation']}: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    errors.append(error_msg)

            # Average metrics for this sample
            if sample_results:
                avg_metrics = {
                    metric: np.mean([r[metric] for r in sample_results])
                    for metric in sample_results[0].keys()
                }
                results.append(avg_metrics)
                
                if debug:
                    logging.debug(f"Sample {idx} average metrics: {avg_metrics}")

        except Exception as e:
            error_msg = f"Error processing sample {idx}: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            errors.append(error_msg)

    if not results:
        raise ValueError("No valid results obtained from evaluation")

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

    logging.info(f"Evaluation completed. Processed {len(results)} samples successfully.")
    if errors:
        logging.warning(f"Encountered {len(errors)} errors during evaluation.")

    return overall_metrics, metric_stds, results, errors

def main():
    """Main function"""
    try:
        args = parse_args()
        
        # Set up output directory and logging
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = setup_logging(args.output_dir)
        
        # Log system information
        logging.info(f"Python version: {sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"OpenCV version: {cv2.__version__}")
        logging.info(f"Arguments: {vars(args)}")
        
        device = setup_device()

        # Load dataset
        try:
            split_file = os.path.join(args.splits_dir, "val.json")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            
            with open(split_file, 'r') as f:
                data = json.load(f)
            logging.info(f"Loaded {len(data)} samples from {split_file}")
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            logging.error(traceback.format_exc())
            raise

        # Prepend data_dir to paths
        for item in data:
            item['image'] = os.path.join(args.data_dir, item['image'])
            item['annotation'] = os.path.join(args.data_dir, item['annotation'])
            print(item)
        # Limit number of samples if specified
        if args.num_samples > 0:
            data = data[:args.num_samples]
            logging.info(f"Using {args.num_samples} samples for evaluation")

        # Evaluate pre-trained model
        logging.info("\nEvaluating pre-trained model...")
        pretrained_predictor = load_model(args.pretrained_checkpoint, args.model_cfg, device)
        pretrained_metrics, pretrained_stds, pretrained_results, pretrained_errors = evaluate_model(
            pretrained_predictor, data, device, args.debug
        )

        # Evaluate fine-tuned model
        logging.info("\nEvaluating fine-tuned model...")
        finetuned_predictor = load_model(args.finetuned_checkpoint, args.model_cfg, device)
        finetuned_metrics, finetuned_stds, finetuned_results, finetuned_errors = evaluate_model(
            finetuned_predictor, data, device, args.debug
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
                'all_results': pretrained_results,
                'errors': pretrained_errors
            },
            'finetuned': {
                'metrics': finetuned_metrics,
                'std_devs': finetuned_stds,
                'all_results': finetuned_results,
                'errors': finetuned_errors
            },
            'improvements': improvements
        }

        output_file = os.path.join(args.output_dir, 'comparison_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Print comparison
        logging.info("\nModel Comparison Results:")
        logging.info("=" * 80)
        logging.info(f"{'Metric':15} {'Pre-trained':>12} {'Fine-tuned':>12} {'Improvement':>12}")
        logging.info("-" * 80)
        
        for metric in pretrained_metrics.keys():
            logging.info(
                f"{metric:15} {pretrained_metrics[metric]:>12.4f} "
                f"{finetuned_metrics[metric]:>12.4f} {improvements[metric]:>11.1f}%"
            )
        
        logging.info("\nDetailed results saved to: " + output_file)
        logging.info("Log file: " + log_file)

        # Report any errors
        total_errors = len(pretrained_errors) + len(finetuned_errors)
        if total_errors > 0:
            logging.warning(f"\nTotal errors encountered: {total_errors}")
            logging.warning("Check the log file for detailed error messages.")

    except Exception as e:
        logging.error("Fatal error in main execution:")
        logging.error(str(e))
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 