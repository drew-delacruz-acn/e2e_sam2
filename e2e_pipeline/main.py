import yaml
import argparse
import os
from e2e_pipeline.modules.frame_sampling import sample_frames_from_video
from e2e_pipeline.modules.segmentation import segment_image
from e2e_pipeline.modules.embedding import generate_embeddings_for_crops
from e2e_pipeline.modules.serialization import save_embeddings_json

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Running pipeline in {config['mode']} mode")

    if config['mode'] == 'video':
        frame_paths = sample_frames_from_video(config['video'])
        print(f"Sampled {len(frame_paths)} frames from video")
    elif config['mode'] == 'image':
        frame_paths = [config['image']['image_path']]
        print(f"Using single image: {frame_paths[0]}")
    else:
        raise ValueError(f"Invalid mode in config: {config['mode']}")

    all_crop_infos = []
    for frame_path in frame_paths:
        print(f"Processing frame: {frame_path}")
        crop_infos = segment_image(frame_path, config['segmentation'])
        all_crop_infos.extend(crop_infos)
    
    print(f"Generated {len(all_crop_infos)} crops across all frames")

    embeddings = generate_embeddings_for_crops(all_crop_infos, config['embedding'])
    print(f"Generated embeddings for {len(embeddings)} crops")
    
    output_path = save_embeddings_json(embeddings, config['serialization'])
    print(f"Saved embeddings to {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end pipeline for video/image segmentation and embedding generation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config) 