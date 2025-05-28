#!/usr/bin/env python3
"""
Main entry point to run the Iterative Contrastive Learning Pipeline.
"""
import argparse
import sys
import traceback
from pathlib import Path

from iterative_pipeline.config import PipelineConfig
from iterative_pipeline.data_utils import load_and_validate_data
from iterative_pipeline.pipeline_manager import run_iterative_pipeline

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args_for_runner():
    """Parse command line arguments for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Iterative Contrastive Learning Pipeline with Negative Classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--definitive-objects', type=str, required=True)
    parser.add_argument('--resnet-predictions', type=str, required=True)
    parser.add_argument('--tracking-info', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--secondary-threshold', type=float, default=None)
    parser.add_argument('--convergence-threshold', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--secondary-margin', type=float, default=None)
    exclusion_group = parser.add_mutually_exclusive_group()
    exclusion_group.add_argument('--exclude-training-from-eval', dest='eval_strategy', action='store_const', const='exclude')
    exclusion_group.add_argument('--include-training-in-eval', dest='eval_strategy', action='store_const', const='include')
    parser.set_defaults(eval_strategy='exclude') 
    parser.add_argument('--output', type=str, default='results_negative_refactored')
    parser.add_argument('--test-mode', action='store_true')
    
    parsed_args = parser.parse_args()
    config_dict = vars(parsed_args)
    
    # Extract eval_strategy before creating config
    eval_strat = config_dict.pop('eval_strategy', 'exclude')
    
    # Convert argument names from dashes to underscores for PipelineConfig
    config_dict['definitive_objects'] = config_dict.pop('definitive_objects')
    config_dict['resnet_predictions'] = config_dict.pop('resnet_predictions') 
    config_dict['tracking_info'] = config_dict.pop('tracking_info')
    config_dict['secondary_threshold'] = config_dict.pop('secondary_threshold')
    config_dict['convergence_threshold'] = config_dict.pop('convergence_threshold')
    config_dict['secondary_margin'] = config_dict.pop('secondary_margin')
    config_dict['test_mode'] = config_dict.pop('test_mode')
    
    # Create config with properly named fields
    config = PipelineConfig(**config_dict)
    
    # Set evaluation strategy based on parsed argument
    if eval_strat == 'include':
        config.exclude_training_from_eval = False
        config.include_training_in_eval = True
    else:
        config.exclude_training_from_eval = True
        config.include_training_in_eval = False
    
    return config

def main_runner():
    """Main execution function for the refactored pipeline."""
    try:
        config = parse_args_for_runner()
        output_dir = Path(config.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Pipeline configured. Output will be in: {output_dir.resolve()}")
        definitiveObjects, resnetPredictions, trackingInfo = load_and_validate_data(
            config.definitive_objects, 
            config.resnet_predictions, 
            config.tracking_info
        )
        if config.test_mode:
            print("\n✅ Test mode: Data loading and validation successful. Exiting.")
            return
        run_iterative_pipeline(definitiveObjects, resnetPredictions, trackingInfo, config)
        print("\n🎉 Iterative contrastive learning pipeline finished successfully!")
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}. Please check input paths.")
        print("--- Traceback ---"); traceback.print_exc(); print("--- End Traceback ---")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Data validation or Value error: {e}")
        print("--- Traceback ---"); traceback.print_exc(); print("--- End Traceback ---")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Pipeline runtime error: {e}")
        print("--- Traceback ---"); traceback.print_exc(); print("--- End Traceback ---")
        sys.exit(1)
    except Exception as e: 
        print(f"❌ An unexpected error occurred: {e}")
        print("--- Traceback ---"); traceback.print_exc(); print("--- End Traceback ---")
        sys.exit(1)

if __name__ == "__main__":
    main_runner() 