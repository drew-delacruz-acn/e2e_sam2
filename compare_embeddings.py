#!/usr/bin/env python3
"""
Script to compare embeddings between ground truth objects and query objects.
"""

import os
import json
import numpy as np
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedding_compare')

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_embeddings(json_path):
    """Load embeddings from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data["embeddings"]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare embeddings between objects")
    parser.add_argument("--query", type=str, required=True,
                      help="Path to query object embeddings JSON")
    parser.add_argument("--ground_truth_dir", type=str, required=True,
                      help="Directory containing ground truth embeddings JSONs")
    parser.add_argument("--threshold", type=float, default=0.7,
                      help="Similarity threshold (0-1)")
    parser.add_argument("--output", type=str, default="results/comparison_results.json",
                      help="Path to save comparison results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load query embeddings
    logger.info(f"Loading query embeddings from {args.query}")
    query_emb = load_embeddings(args.query)
    
    # Load ground truth embeddings
    gt_dir = Path(args.ground_truth_dir)
    ground_truth = {}
    logger.info(f"Loading ground truth embeddings from {gt_dir}")
    
    for json_file in gt_dir.glob("*_embeddings.json"):
        name = json_file.stem.replace("_embeddings", "")
        ground_truth[name] = load_embeddings(json_file)
        logger.info(f"Loaded ground truth: {name}")
    
    if not ground_truth:
        logger.error(f"No ground truth embeddings found in {gt_dir}")
        return 1
    
    # Compare embeddings
    results = []
    logger.info("Comparing embeddings...")
    
    for gt_name, gt_emb in ground_truth.items():
        similarities = {}
        
        # Compare each model's embeddings
        for model in set(query_emb.keys()) & set(gt_emb.keys()):
            try:
                sim = cosine_similarity(
                    query_emb[model]["crop"],
                    gt_emb[model]["crop"]
                )
                similarities[model] = float(sim)
            except Exception as e:
                logger.warning(f"Could not compare {model} embeddings for {gt_name}: {e}")
                continue
        
        if similarities:
            avg_sim = sum(similarities.values()) / len(similarities)
            
            results.append({
                "ground_truth_object": gt_name,
                "similarities": similarities,
                "average_similarity": avg_sim
            })
    
    # Sort by average similarity
    results.sort(key=lambda x: x["average_similarity"], reverse=True)
    
    # Filter by threshold
    results = [r for r in results if r["average_similarity"] > args.threshold]
    
    # Print results
    print("\nResults:")
    for r in results:
        print(f"\nGround Truth Object: {r['ground_truth_object']}")
        print("Similarities:")
        for model, score in r["similarities"].items():
            print(f"  {model}: {score:.3f}")
        print(f"Average: {r['average_similarity']:.3f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "query_embeddings": args.query,
            "ground_truth_dir": str(args.ground_truth_dir),
            "threshold": args.threshold,
            "matches": results
        }, f, indent=2)
    
    logger.info(f"Saved results to {args.output}")
    
    if not results:
        print("\nNo matches found above threshold {args.threshold}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 