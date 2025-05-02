import os
import json
from pathlib import Path

def load_embeddings(results_dir, model_type="resnet50"):
    """
    Load embeddings from the results directory.
    
    Args:
        results_dir: Path to the results directory
        model_type: Type of embeddings to load ("resnet50", "vit", or "clip")
        
    Returns:
        Dictionary with video_name and images (mapping image names to their segment embeddings)
    """
    # Validate model type
    if model_type not in ["resnet50", "vit", "clip"]:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'resnet50', 'vit', or 'clip'")
    
    results_path = Path(results_dir)
    all_embeddings = {}
    video_name = results_path.name  # Default to directory name
    
    # First check if there's a processing summary
    summary_path = results_path / "processing_summary.json"
    if summary_path.exists():
        # Use the summary to find embedding files
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Get video name from summary if available
        if "video_name" in summary:
            video_name = summary["video_name"]
            
        for img_data in summary.get("processed_images", []):
            embeddings_file = img_data.get("embeddings_file")
            if embeddings_file and os.path.exists(embeddings_file):
                with open(embeddings_file, 'r') as f:
                    img_embeddings = json.load(f)
                
                # Check if the embedding file has a video_name that should override our default
                if "video_name" in img_embeddings and img_embeddings["video_name"]:
                    video_name = img_embeddings["video_name"]
                
                image_name = Path(img_embeddings["original_image"]).stem
                all_embeddings[image_name] = {}
                
                # Extract the requested embeddings for each segment
                for segment_id, segment_data in img_embeddings["segments"].items():
                    if model_type in segment_data["embeddings"]:
                        all_embeddings[image_name][segment_id] = {
                            "embedding": segment_data["embeddings"][model_type],
                            "path": segment_data.get("padded_path", segment_data.get("path"))  # Prefer padded_path
                        }
    else:
        # Scan directory structure to find embedding files
        for img_dir in results_path.iterdir():
            if img_dir.is_dir():
                embeddings_dir = img_dir / "embeddings"
                if embeddings_dir.exists():
                    for embed_file in embeddings_dir.glob("*_embeddings.json"):
                        with open(embed_file, 'r') as f:
                            img_embeddings = json.load(f)
                        
                        # Check if the embedding file has a video_name that should override our default
                        if "video_name" in img_embeddings and img_embeddings["video_name"]:
                            video_name = img_embeddings["video_name"]
                        
                        image_name = img_dir.name
                        all_embeddings[image_name] = {}
                        
                        # Extract the requested embeddings for each segment
                        for segment_id, segment_data in img_embeddings["segments"].items():
                            if model_type in segment_data["embeddings"]:
                                all_embeddings[image_name][segment_id] = {
                                    "embedding": segment_data["embeddings"][model_type],
                                    "path": segment_data.get("padded_path", segment_data.get("path"))  # Prefer padded_path
                                }
    
    # Return both the video name and the embeddings
    return {
        "video_name": video_name,
        "images": all_embeddings
    }


def process_master_summary(results_dir, model_type="resnet50"):
    """
    Process all videos in a master summary.
    
    Args:
        results_dir: Path to the results directory containing the master_summary.json
        model_type: Type of embeddings to use ('resnet50', 'vit', or 'clip')
        
    Returns:
        Combined array of results from all videos
    """
    master_summary_path = Path(results_dir) / "master_summary.json"
    if not master_summary_path.exists():
        print(f"No master summary found at {master_summary_path}")
        # If no master summary, try to process as a single video directory
        return search_similar_segments_in_neo4j(results_dir, model_type)
    
    print(f"Found master summary at {master_summary_path}")
    with open(master_summary_path, 'r') as f:
        master_summary = json.load(f)
    
    all_results = []
    processed_videos = []
    
    for folder_data in master_summary.get("processed_folders", []):
        video_name = folder_data.get("video_name")
        video_dir = folder_data.get("output_dir")
        
        if not video_name or not video_dir:
            print(f"Skipping invalid folder data: {folder_data}")
            continue
            
        print(f"\n========== Processing video: {video_name} ==========")
        video_results = search_similar_segments_in_neo4j(video_dir, model_type)
        
        # Add to combined results
        all_results.extend(video_results)
        processed_videos.append(video_name)
    
    if all_results:
        # Save combined results
        output_file = f"combined_search_results_{model_type}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n========== Summary ==========")
        print(f"Processed {len(processed_videos)} videos: {', '.join(processed_videos)}")
        print(f"Combined results saved to {output_file}")
        print(f"Found {len(all_results)} total matches across all videos")
    else:
        print("No results found in any video.")
    
    return all_results


def search_similar_segments_in_neo4j(embeddings_dir, model_type="resnet50"):
    """
    Load embeddings and search for similar segments in Neo4j.
    
    Args:
        embeddings_dir: Directory containing segment embeddings
        model_type: Type of embeddings to use ('resnet50', 'vit', or 'clip')
        
    Returns:
        Array of match results
    """
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    import os
    import json
    
    # Load environment variables
    load_dotenv()
    NEO4J_DB = 'ipid'
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    # Load embeddings
    print(f"Loading {model_type} embeddings from {embeddings_dir}...")
    embedding_result = load_embeddings(embeddings_dir, model_type)
    video_name = embedding_result["video_name"]
    embeddings = embedding_result["images"]
    print(f"Loaded embeddings for video '{video_name}' with {len(embeddings)} images")
    
    # Prepare results as a flat array
    results_array = []
    
    # Query for each segment's embedding
    with driver.session(database=NEO4J_DB) as session:
        for image_name, segments in embeddings.items():
            # The image_name (folder name) is used as the frame number
            frame = image_name
            
            for segment_id, data in segments.items():
                embedding = data["embedding"]
                segment_path = data["path"]
                
                # Use the Neo4j vector search query
                query = """
                CALL db.index.vector.queryNodes("resnet_index", 14752, $embedding)
                YIELD node, score
                MATCH (origin)-[:custom_hasResnetEmbedding]->(node)
                WHERE NOT origin.value CONTAINS 'Scenes' AND origin.custom_hasPadding = true
                WITH origin.value AS result, origin.omc_hasVersion as version, score AS similarity
                WHERE similarity > $threshold
                ORDER BY similarity DESC
                RETURN DISTINCT result, version, similarity
                LIMIT $top_k
                """
                
                results = session.run(
                    query, 
                    embedding=embedding, 
                    top_k=1, 
                    threshold=0.7
                )
                
                # Process results
                for record in results:
                    # Create a flat result item for each match
                    result_item = {
                        "video_name": video_name,
                        "predicted_object": record['result'],
                        "object_version": record['version'],
                        "similarity_score": record['similarity'],
                        "model": model_type,
                        "frame": frame,
                        "segment_id": segment_id,
                        "bbox_path": segment_path  # Changed from segment_path to bbox_path
                    }
                    
                    # Add to results array
                    results_array.append(result_item)
                
                print(f"Processed segment {segment_id} from frame {frame}")
    
    # Save all results to a JSON file
    output_file = f"{video_name}_search_results_{model_type}.json"
    with open(output_file, 'w') as f:
        json.dump(results_array, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Found {len(results_array)} matches across {len(embeddings)} frames")
    
    return results_array


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search for similar segments in Neo4j")
    parser.add_argument("--results_dir", type=str, required=True,
                      help="Path to the results directory")
    parser.add_argument("--model", type=str, default="resnet50",
                      choices=["resnet50", "vit", "clip"],
                      help="Embedding model to use")
    parser.add_argument("--process_all", action="store_true",
                      help="Process all videos in the master summary")
    
    args = parser.parse_args()
    
    if args.process_all:
        # Process all videos in the master summary
        process_master_summary(args.results_dir, args.model)
    else:
        # Process a single video directory
        search_similar_segments_in_neo4j(args.results_dir, args.model)

# Example usage:
# 
# # Process a specific video directory:
# python json_comp.py --results_dir results/video1 --model resnet50
#
# # Process all videos from master summary:
# python json_comp.py --results_dir results --model resnet50 --process_all
#
# # Or use the functions directly:
# results = search_similar_segments_in_neo4j("results/video1", "resnet50")
# all_results = process_master_summary("results", "resnet50")
