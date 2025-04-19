#!/usr/bin/env python3
"""
Script to compare existing embeddings against ground truth embeddings.
Skips detection and segmentation, directly compares embeddings.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import clip
from tqdm import tqdm

from e2e_pipeline_v2.modules.metrics import compute_cosine_similarity, calculate_metrics, save_metrics_report
from e2e_pipeline_v2.modules.visualization import create_results_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('embedding_comparison')

class EmbeddingComparator:
    def __init__(self, ground_truth_dir: str, results_dir: str):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.results_dir = Path(results_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load ground truth mapping
        self.gt_mapping = self._load_ground_truth_mapping()
        
        # Load all ground truth embeddings
        self.gt_embeddings = self._load_all_ground_truth_embeddings()

    def _load_ground_truth_mapping(self) -> dict:
        mapping_path = self.ground_truth_dir / "ground_truth_embeddings/ground_truth_mapping.json"
        with open(mapping_path) as f:
            return json.load(f)

    def _load_all_ground_truth_embeddings(self) -> dict:
        embeddings = {}
        for obj_name, obj_info in self.gt_mapping["processed_objects"].items():
            embedding_path = self.ground_truth_dir / "ground_truth_embeddings" / f"{obj_name}_embeddings.json"
            with open(embedding_path) as f:
                embeddings[obj_name] = json.load(f)
        return embeddings

    def generate_embedding(self, image_path: str) -> torch.Tensor:
        """Generate CLIP embedding for a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize the features
                image_features = F.normalize(image_features, dim=-1)
            return image_features.cpu()
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            return None

    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        return F.cosine_similarity(embedding1, embedding2).item()

    def process_object_folder(self, object_name: str) -> list:
        """Process all detections for a single object."""
        results = []
        object_dir = self.results_dir / object_name
        
        # Skip if not a directory or embeddings_full
        if not object_dir.is_dir() or object_name == "embeddings_full":
            return results

        # Get ground truth info
        gt_name = object_name.split('-')[0] if '-' in object_name else object_name
        if gt_name not in self.gt_embeddings:
            logger.warning(f"No ground truth found for {object_name}")
            return results

        gt_embedding = torch.tensor(self.gt_embeddings[gt_name]["embeddings"]["clip"])
        threshold = self.gt_mapping["object_classes"][gt_name]["similarity_threshold"]

        # Process each timestamp directory
        for timestamp_dir in object_dir.glob("*-*"):
            if not timestamp_dir.is_dir():
                continue

            metadata_path = timestamp_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Process each detection
            for detection in metadata["detections"]:
                crop_path = self.results_dir.parent / detection["crop_path"]
                if not crop_path.exists():
                    continue

                # Generate embedding for detection
                detection_embedding = self.generate_embedding(str(crop_path))
                if detection_embedding is None:
                    continue

                # Compute similarity
                similarity = self.compute_similarity(detection_embedding, gt_embedding.unsqueeze(0))

                results.append({
                    "object_name": object_name,
                    "detection_id": detection["id"],
                    "label": detection["label"],
                    "score": detection["score"],
                    "similarity": similarity,
                    "exceeds_threshold": similarity >= threshold,
                    "crop_path": str(crop_path)
                })

        return results

    def process_all_objects(self) -> dict:
        """Process all object folders and generate a comprehensive report."""
        all_results = {}
        
        # Get list of object folders
        object_folders = [d for d in self.results_dir.iterdir() 
                        if d.is_dir() and d.name != "embeddings_full"]

        for object_folder in tqdm(object_folders, desc="Processing objects"):
            results = self.process_object_folder(object_folder.name)
            if results:
                all_results[object_folder.name] = results

        return all_results

    def save_results(self, results: dict, output_path: str):
        """Save comparison results to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare detections with ground truth embeddings")
    parser.add_argument("--ground_truth_dir", type=str, required=True,
                      help="Path to ground truth directory")
    parser.add_argument("--results_dir", type=str, required=True,
                      help="Path to results directory")
    parser.add_argument("--output", type=str, default="embedding_comparison_results.json",
                      help="Path to save comparison results")
    parser.add_argument("--object", type=str,
                      help="Process specific object only")
    return parser.parse_args()

def main():
    args = parse_args()
    
    comparator = EmbeddingComparator(args.ground_truth_dir, args.results_dir)
    
    if args.object:
        results = {args.object: comparator.process_object_folder(args.object)}
    else:
        results = comparator.process_all_objects()
    
    comparator.save_results(results, args.output)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    sys.exit(main()) 