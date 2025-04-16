import torch

def compute_segmentation_loss(gt_mask, prd_mask):
    """
    Compute cross entropy loss between ground truth mask and predicted mask
    
    Args:
        gt_mask: Ground truth binary mask
        prd_mask: Predicted probability mask (after sigmoid)
    
    Returns:
        Cross entropy loss
    """
    return (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()

def compute_iou(gt_mask, prd_mask_binary):
    """
    Compute Intersection over Union (IoU) between masks
    
    Args:
        gt_mask: Ground truth binary mask
        prd_mask_binary: Predicted binary mask (after thresholding)
    
    Returns:
        IoU score
    """
    inter = (gt_mask * prd_mask_binary).sum(1).sum(1)
    union = gt_mask.sum(1).sum(1) + prd_mask_binary.sum(1).sum(1) - inter
    iou = inter / (union + 1e-5)  # add epsilon to avoid division by zero
    return iou

def compute_score_loss(pred_scores, iou):
    """
    Compute score loss as absolute difference between predicted score and IoU
    
    Args:
        pred_scores: Predicted scores from model
        iou: Calculated IoU values
    
    Returns:
        Score loss
    """
    return torch.abs(pred_scores - iou).mean()

def compute_total_loss(seg_loss, score_loss, score_weight=0.05):
    """
    Compute total loss as weighted sum of segmentation and score losses
    
    Args:
        seg_loss: Segmentation loss
        score_loss: Score loss
        score_weight: Weight for score loss
    
    Returns:
        Total loss
    """
    return seg_loss + score_loss * score_weight 