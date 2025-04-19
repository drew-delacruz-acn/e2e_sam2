#!/usr/bin/env python3
"""
Script to process detection and segmentation results through the embedding module.
Compares detected objects with ground truth embeddings.
"""

import os
import sys
import json
import time
import logging
import traceback
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from typing import Dict, List, Any

from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator
from e2e_pipeline_v2.modules.metrics import compute_cosine_similarity, calculate_metrics, save_metrics_report
from e2e_pipeline_v2.modules.visualization import create_results_visualization
from e2e_pipeline_v2.pipeline import DetectionSegmentationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('embedding_process')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process images and compare with ground truth embeddings")
    
    parser.add_argument("--image", type=str, required=True,
                      help="Path to the input image")
    parser.add_argument("--gt_embeddings", type=str, required=True,
                      help="Path to ground truth embeddings directory")
    parser.add_argument("--gt_mapping", type=str, required=True,
                      help="Path to ground truth mapping file")
    parser.add_argument("--config", type=str, default="e2e_pipeline_v2/config/config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--models", type=str, nargs="+", default=["clip", "vit", "resnet50"],
                      choices=["clip", "vit", "resnet50"],
                      help="Embedding models to use")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to use (cuda or cpu)")
    parser.add_argument("--similarity_threshold", type=float, default=0.75,
                      help="Threshold for similarity matching")
    parser.add_argument("--debug", action="store_true",
                      help="Enable detailed debug logging")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_ground_truth_embeddings(gt_dir: str) -> Dict:
    """Load all ground truth embeddings from directory."""
    gt_embeddings = {}
    
    for file in os.listdir(gt_dir):
        if not file.endswith('_embeddings.json'):
            continue
            
        with open(os.path.join(gt_dir, file), 'r') as f:
            data = json.load(f)
            gt_embeddings[data['object_name']] = data['embeddings']
            
    return gt_embeddings

def load_ground_truth_mapping(mapping_path: str) -> Dict:
    """Load ground truth mapping file."""
    with open(mapping_path, 'r') as f:
        return json.load(f)

def get_detection_crop(image: np.ndarray, box: List[int]) -> np.ndarray:
    """Extract crop from image using detection box."""
    x1, y1, x2, y2 = [int(coord) for coord in box]
    return image[y1:y2, x1:x2]

def compare_with_ground_truth(
    detection_embedding: np.ndarray,
    gt_embeddings: Dict,
    model_type: str,
    threshold: float
) -> List[Dict]:
    """Compare detection embedding with all ground truth embeddings."""
    matches = []
    
    for obj_name, obj_embeddings in gt_embeddings.items():
        if model_type not in obj_embeddings:
            continue
            
        similarity = compute_cosine_similarity(
            detection_embedding,
            obj_embeddings[model_type]
        )
        
        if similarity >= threshold:
            matches.append({
                "object_name": obj_name,
                "similarity": similarity
            })
    
    return sorted(matches, key=lambda x: x['similarity'], reverse=True)

def process_image(
    image_path: str,
    pipeline: DetectionSegmentationPipeline,
    embedding_generator: EmbeddingGenerator,
    gt_embeddings: Dict,
    models: List[str],
    threshold: float
) -> Dict:
    """Process single image and compare with ground truth."""
    # Run detection pipeline
    results = pipeline.run(
        image_path=image_path,
        visualize=True,
        save_results=True,
        generate_embeddings=False
    )
    
    if not results or "detection" not in results:
        raise ValueError("Detection failed or no objects found")
    
    # Process each detection
    processed_results = {}
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for i, (box, score) in enumerate(zip(
        results["detection"]["boxes"],
        results["detection"]["scores"]
    )):
        detection_id = f"detection_{i}"
        
        # Get crop
        crop = get_detection_crop(image, box)
        
        # Generate embeddings and compare
        detection_results = {
            "box": box.tolist() if isinstance(box, np.ndarray) else box,
            "score": float(score),
            "matches": {}
        }
        
        for model_type in models:
            try:
                # Generate embedding
                embedding = embedding_generator.generate_embedding(crop, model_type)
                
                # Compare with ground truth
                matches = compare_with_ground_truth(
                    embedding,
                    gt_embeddings,
                    model_type,
                    threshold
                )
                
                detection_results["matches"][model_type] = matches
                
                # Update best match if found
                if matches:
                    best_match = matches[0]
                    if (
                        "best_score" not in detection_results
                        or best_match["similarity"] > detection_results["best_score"]
                    ):
                        detection_results["best_match"] = best_match["object_name"]
                        detection_results["best_score"] = best_match["similarity"]
                        
            except Exception as e:
                logger.error(f"Error processing {model_type} for detection {i}: {str(e)}")
                continue
        
        processed_results[detection_id] = detection_results
    
    return processed_results

def main():
    start_time = time.time()
    args = parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration and ground truth data
    config = load_config(args.config)
    gt_embeddings = load_ground_truth_embeddings(args.gt_embeddings)
    gt_mapping = load_ground_truth_mapping(args.gt_mapping)
    
    # Initialize components
    pipeline = DetectionSegmentationPipeline(args.config)
    embedding_generator = EmbeddingGenerator(
        model_types=args.models,
        device=args.device
    )
    
    try:
        # Process image
        logger.info(f"Processing image: {args.image}")
        results = process_image(
            args.image,
            pipeline,
            embedding_generator,
            gt_embeddings,
            args.models,
            args.similarity_threshold
        )
        
        # Calculate metrics
        metrics = calculate_metrics(
            results,
            gt_mapping["object_classes"],
            args.similarity_threshold
        )
        
        # Create output directory structure
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = os.path.join(args.output_dir, image_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_path = os.path.join(output_dir, "detection_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        save_metrics_report(metrics, metrics_path)
        
        # Create visualizations
        create_results_visualization(
            args.image,
            [det for det in results.values()],
            metrics,
            os.path.join(output_dir, "visualizations")
        )
        
        # Print summary
        end_time = time.time()
        logger.info("\n=== Processing Summary ===")
        logger.info(f"Total time: {end_time - start_time:.2f} seconds")
        logger.info(f"Processed detections: {len(results)}")
        logger.info(f"Overall F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 