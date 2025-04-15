"""
Script to run the modular segmentation pipeline.
"""
import os
import sys
import argparse
import subprocess

# # Check if virtual environment exists, create if not
# def setup_environment():
#     """Set up virtual environment and install dependencies if needed"""
#     if not os.path.exists("venv"):
#         print("Creating virtual environment...")
#         os.system("python -m venv venv")
        
#         # Activate and install dependencies
#         if sys.platform == "win32":
#             subprocess.check_call("venv\\Scripts\\pip install -r requirements.txt", shell=True)
#             subprocess.check_call("cd sam2 && ..\\venv\\Scripts\\pip install -e .", shell=True)
#         else:
#             subprocess.check_call("venv/bin/pip install -r requirements.txt", shell=True)
#             subprocess.check_call("cd sam2 && ../venv/bin/pip install -e .", shell=True)
#         print("Virtual environment created and dependencies installed.")
#     else:
#         print("Using existing virtual environment.")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the object detection and segmentation pipeline")
    parser.add_argument("--image", type=str, default="data/thor_hammer.jpeg", 
                        help="Path to input image")
    parser.add_argument("--queries", type=str, nargs="+", default=["hammer"], 
                        help="Text queries for detection")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Show visualizations")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results to disk")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--detection-threshold", type=float, default=0.1,
                        help="Detection confidence threshold")
    parser.add_argument("--segmentation-threshold", type=float, default=0.5,
                        help="Segmentation confidence threshold")
    return parser.parse_args()

def main():
    """Main function to run the pipeline"""
    args = parse_args()
    
    # Custom configuration based on command line arguments
    config_override = {
        "detection": {
            "threshold": args.detection_threshold
        },
        "segmentation": {
            "score_threshold": args.segmentation_threshold
        },
        "paths": {
            "results_dir": args.results_dir
        }
    }
    
    # Import here to avoid import before environment setup
    from pipeline.pipeline import Pipeline
    
    # Initialize the pipeline with custom configuration
    pipeline = Pipeline(config_override=config_override)
    
    # Run the pipeline
    results = pipeline.run(
        image_path=args.image,
        text_queries=args.queries,
        visualize=args.visualize,
        save_results=args.save
    )
    
    print(f"Pipeline completed successfully!")
    print(f"Results saved to: {args.results_dir}")

if __name__ == "__main__":
    # setup_environment()
    main() 