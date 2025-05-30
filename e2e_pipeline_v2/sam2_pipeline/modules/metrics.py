import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import json
import os
from typing import Dict, List, Tuple, Any

def compute_cosine_similarity(query_embedding: torch.Tensor, gt_embedding: torch.Tensor) -> float:
    """
    Compute cosine similarity between two embeddings.
    """
    if isinstance(query_embedding, list):
        query_embedding = torch.tensor(query_embedding)
    if isinstance(gt_embedding, list):
        gt_embedding = torch.tensor(gt_embedding)
        
    query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
    gt_norm = torch.nn.functional.normalize(gt_embedding.unsqueeze(0), p=2, dim=1)
    
    return torch.mm(query_norm, gt_norm.transpose(0, 1)).item()

def calculate_metrics(comparison_results: Dict, ground_truth_mapping: Dict, threshold: float = 0.75) -> Dict:
    """
    Calculate precision, recall, F1 score for object detection and matching.
    """
    metrics = {
        "total_TP": 0,
        "total_FP": 0,
        "total_FN": 0,
        "by_class": {}
    }
    
    # Initialize per-class counters
    for obj_class in set(ground_truth_mapping.values()):
        metrics["by_class"][obj_class] = {
            "TP": 0, "FP": 0, "FN": 0
        }

    # Process each detection
    for detection_id, results in comparison_results.items():
        best_match = None
        best_score = 0
        
        # Find best match across all models
        for model_type, matches in results["matches"].items():
            for match in matches:
                if match["similarity"] > best_score:
                    best_score = match["similarity"]
                    best_match = match["object_name"]
        
        matched_class = ground_truth_mapping.get(best_match) if best_match else None
        
        # Update metrics
        if best_score >= threshold and matched_class:
            metrics["total_TP"] += 1
            metrics["by_class"][matched_class]["TP"] += 1
        else:
            metrics["total_FP"] += 1
            for class_metrics in metrics["by_class"].values():
                class_metrics["FP"] += 1

    # Calculate aggregate metrics
    metrics["precision"] = calculate_precision(metrics["total_TP"], metrics["total_FP"])
    metrics["recall"] = calculate_recall(metrics["total_TP"], metrics["total_FN"])
    metrics["f1_score"] = calculate_f1(metrics["precision"], metrics["recall"])

    # Calculate per-class metrics
    for class_name, class_metrics in metrics["by_class"].items():
        class_metrics["precision"] = calculate_precision(class_metrics["TP"], class_metrics["FP"])
        class_metrics["recall"] = calculate_recall(class_metrics["TP"], class_metrics["FN"])
        class_metrics["f1_score"] = calculate_f1(class_metrics["precision"], class_metrics["recall"])

    return metrics

def calculate_precision(tp: int, fp: int) -> float:
    """Calculate precision from true positives and false positives."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calculate_recall(tp: int, fn: int) -> float:
    """Calculate recall from true positives and false negatives."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def save_metrics_report(metrics: Dict, output_path: str) -> None:
    """Save detailed metrics report to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = {
        "overall_metrics": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "true_positives": metrics["total_TP"],
            "false_positives": metrics["total_FP"],
            "false_negatives": metrics["total_FN"]
        },
        "class_metrics": metrics["by_class"]
    }
    
    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save human-readable summary
    summary_path = output_path.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== Overall Metrics ===\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall: {metrics['recall']:.3f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.3f}\n")
        f.write(f"True Positives: {metrics['total_TP']}\n")
        f.write(f"False Positives: {metrics['total_FP']}\n")
        f.write(f"False Negatives: {metrics['total_FN']}\n\n")
        
        f.write("=== Per-Class Metrics ===\n")
        for class_name, class_metrics in metrics["by_class"].items():
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {class_metrics['precision']:.3f}\n")
            f.write(f"  Recall: {class_metrics['recall']:.3f}\n")
            f.write(f"  F1 Score: {class_metrics['f1_score']:.3f}\n") 