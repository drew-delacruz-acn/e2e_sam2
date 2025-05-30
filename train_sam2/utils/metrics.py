"""
Metrics and evaluation utilities for SAM2 training and inference.
"""
import json
import numpy as np
import torch
from tqdm import tqdm


class MetricsTracker:
    """Class to track and analyze training metrics"""
    def __init__(self, metrics_file, gap_threshold=0.1, gap_increase_threshold=0.05, 
                 patience=5, max_gap=0.3):
        self.metrics_file = metrics_file
        self.gap_threshold = gap_threshold
        self.gap_increase_threshold = gap_increase_threshold
        self.patience = patience  # Number of consecutive warnings before stopping
        self.max_gap = max_gap   # Maximum allowed gap before forced stopping
        self.metrics = {
            'train_iou': [],
            'val_iou': [],
            'gaps': [],
            'iterations': []
        }
        self.previous_gap = None
        self.consecutive_warnings = 0
        self.best_val_iou = 0
        self.best_model_path = None
        self.iterations_without_improvement = 0
    
    def update(self, iteration, train_iou, val_iou, model, save_path):
        """Update metrics and check for warnings"""
        self.metrics['iterations'].append(iteration)
        self.metrics['train_iou'].append(float(train_iou))
        self.metrics['val_iou'].append(float(val_iou))
        
        current_gap = train_iou - val_iou
        self.metrics['gaps'].append(float(current_gap))
        
        warnings = []
        should_stop = False
        stop_reason = None
        
        # Save model when validation IoU improves
        if val_iou > self.best_val_iou:
            self.best_val_iou = val_iou
            self.iterations_without_improvement = 0
            # Save best model with validation score in filename
            best_model_path = f"{save_path[:-6]}_best_val_{val_iou:.4f}.torch"
            torch.save({
                'model': model.state_dict(),
                'val_iou': val_iou,
                'iteration': iteration
            }, best_model_path)
            # Also save as the default path for backward compatibility
            torch.save({
                'model': model.state_dict(),
                'val_iou': val_iou,
                'iteration': iteration
            }, save_path)
            self.best_model_path = best_model_path
            print(f"\nNew best validation IoU: {val_iou:.4f}")
            print(f"Saved best model to: {best_model_path}")
        else:
            self.iterations_without_improvement += 1
        
        # Check absolute gap
        if current_gap > self.gap_threshold:
            warnings.append(
                f"Warning: Large gap between training and validation IoU "
                f"(train: {train_iou:.4f}, val: {val_iou:.4f}, gap: {current_gap:.4f})"
            )
            self.consecutive_warnings += 1
        else:
            self.consecutive_warnings = 0
        
        # Check gap increase
        if self.previous_gap is not None:
            gap_increase = current_gap - self.previous_gap
            if gap_increase > self.gap_increase_threshold:
                warnings.append(
                    f"Warning: Gap between training and validation IoU is increasing "
                    f"(previous gap: {self.previous_gap:.4f}, current gap: {current_gap:.4f})"
                )
                self.consecutive_warnings += 1
            
        self.previous_gap = current_gap
        
        # Save metrics to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return warnings, should_stop, stop_reason


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
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        raise


def evaluate_model(predictor, device, val_data, num_samples=5):
    """Evaluate model on validation set"""
    from .data_utils import read_batch
    
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


def compute_metrics(pred_mask, gt_mask):
    """Compute metrics between predicted mask and ground truth mask"""
    if gt_mask is None:
        return None
    
    # Convert to binary masks
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    
    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Compute IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0
    
    # Compute Dice coefficient
    dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 0
    
    # Compute precision and recall
    true_positives = intersection
    false_positives = pred_binary.sum() - true_positives
    false_negatives = gt_binary.sum() - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    
    return metrics 