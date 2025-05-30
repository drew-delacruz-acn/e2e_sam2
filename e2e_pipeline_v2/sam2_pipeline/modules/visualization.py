import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict, List, Tuple, Any

def draw_detection_results(image_path: str, detections: List[Dict], output_path: str) -> None:
    """
    Draw bounding boxes and labels on the image.
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Draw each detection
    for det in detections:
        box = det["box"]
        label = det["best_match"] if "best_match" in det else "Unknown"
        score = det["best_score"] if "best_score" in det else 0.0
        
        # Draw box
        draw.rectangle(box, outline="red", width=2)
        
        # Draw label and score
        label_text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1]-25), label_text, fill="red", font=font)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image.save(output_path)

def plot_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str], output_path: str) -> None:
    """
    Create and save confusion matrix visualization.
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_metrics_summary(metrics: Dict, output_path: str) -> None:
    """
    Create and save metrics summary plots.
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall metrics bar plot
    overall_metrics = [
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"]
    ]
    ax1.bar(['Precision', 'Recall', 'F1 Score'], overall_metrics)
    ax1.set_title('Overall Metrics')
    ax1.set_ylim([0, 1])
    
    # Per-class metrics
    class_names = list(metrics["by_class"].keys())
    class_f1_scores = [metrics["by_class"][c]["f1_score"] for c in class_names]
    
    ax2.bar(class_names, class_f1_scores)
    ax2.set_title('F1 Score by Class')
    ax2.set_ylim([0, 1])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def create_results_visualization(
    image_path: str,
    detections: List[Dict],
    metrics: Dict,
    output_dir: str
) -> None:
    """
    Create comprehensive results visualization including detections and metrics.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Draw detections on image
    detection_path = os.path.join(output_dir, "detections.png")
    draw_detection_results(image_path, detections, detection_path)
    
    # Plot metrics
    metrics_path = os.path.join(output_dir, "metrics_summary.png")
    plot_metrics_summary(metrics, metrics_path)
    
    # If we have ground truth data, create confusion matrix
    if "confusion_matrix" in metrics:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            metrics["confusion_matrix"]["true"],
            metrics["confusion_matrix"]["pred"],
            metrics["confusion_matrix"]["labels"],
            cm_path
        ) 