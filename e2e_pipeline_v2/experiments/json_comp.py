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
        Dictionary mapping image names to their segment embeddings
    """
    # Validate model type
    if model_type not in ["resnet50", "vit", "clip"]:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'resnet50', 'vit', or 'clip'")
    
    results_path = Path(results_dir)
    all_embeddings = {}
    
    # First check if there's a processing summary
    summary_path = results_path / "processing_summary.json"
    if summary_path.exists():
        # Use the summary to find embedding files
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        for img_data in summary.get("processed_images", []):
            embeddings_file = img_data.get("embeddings_file")
            if embeddings_file and os.path.exists(embeddings_file):
                with open(embeddings_file, 'r') as f:
                    img_embeddings = json.load(f)
                
                image_name = Path(img_embeddings["original_image"]).stem
                all_embeddings[image_name] = {}
                
                # Extract the requested embeddings for each segment
                for segment_id, segment_data in img_embeddings["segments"].items():
                    if model_type in segment_data["embeddings"]:
                        all_embeddings[image_name][segment_id] = {
                            "embedding": segment_data["embeddings"][model_type],
                            "path": segment_data["path"]
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
                        
                        image_name = img_dir.name
                        all_embeddings[image_name] = {}
                        
                        # Extract the requested embeddings for each segment
                        for segment_id, segment_data in img_embeddings["segments"].items():
                            if model_type in segment_data["embeddings"]:
                                all_embeddings[image_name][segment_id] = {
                                    "embedding": segment_data["embeddings"][model_type],
                                    "path": segment_data["path"]
                                }
    
    return all_embeddings


# Use the function
print("Loading embeddings...")
print("In directory:", os.getcwd())
embeddings = load_embeddings("center_padded_image_results", "resnet50")
counter = 0
# Now you have access to all embeddings
for image_name, segments in embeddings.items():
    for segment_id, data in segments.items():
        embedding = data["embedding"]
        segment_path = data["path"]
        counter += 1


print(f'Loaded {len(embeddings)} images with embeddings.')
print(counter)


from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()
NEO4J_DB = 'ipid'

driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

# with driver.session(database=NEO4J_DB) as session:
#     embedding = THIS SHOULD BE YOUR KEY 'renset50' IN YOUR JSON
#     query = f"""
#         CALL db.index.vector.queryNodes("resnet_index", 14752, $embedding)
#         YIELD node, score
#         MATCH (origin)-[:custom_hasResnetEmbedding]->(node)
#         WHERE NOT origin.value CONTAINS 'Scenes' AND origin.custom_hasPadding = true
#         WITH origin.value AS result, origin.omc_hasVersion as version, score AS similarity
#         WHERE similarity > $threshold
#         ORDER BY similarity DESC
#         RETURN DISTINCT result, version, similarity
#         LIMIT $top_k
#         """

#     results = session.run(query, embedding=embedding, top_k=1, threshold=0.7)
#     top_results = []
#     for record in results:
#         top_results.append({
#             'predicted_object': record['result'],
#             'object_version': record['version'], 
#             'similarity_score': record['similarity']
#         })
#     # save the results to a json