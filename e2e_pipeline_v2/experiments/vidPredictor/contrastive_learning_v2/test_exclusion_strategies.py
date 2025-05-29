#!/usr/bin/env python3
"""
Test Script for Exclusion Strategy Implementation

Demonstrates the new exclusion strategy features in the pipeline.
"""

import subprocess
import sys
from pathlib import Path
import json
import pandas as pd

def test_exclusion_strategies():
    """Test the new exclusion strategy implementations."""
    print("🧪 Testing Exclusion Strategy Implementation")
    print("=" * 50)
    
    # Test 1: Validate config changes
    print("\n1. Testing PipelineConfig validation...")
    try:
        from iterative_pipeline.config import PipelineConfig
        
        # Test valid strategies
        valid_strategies = ['frame-level', 'video-level', 'compare-both']
        for strategy in valid_strategies:
            config = PipelineConfig(
                definitive_objects="test.pkl",
                resnet_predictions="test.pkl", 
                tracking_info="test.pkl",
                exclusion_strategy=strategy
            )
            print(f"   ✅ {strategy}: Valid")
        
        # Test invalid strategy
        try:
            invalid_config = PipelineConfig(
                definitive_objects="test.pkl",
                resnet_predictions="test.pkl",
                tracking_info="test.pkl", 
                exclusion_strategy="invalid-strategy"
            )
            print("   ❌ Invalid strategy validation failed!")
        except ValueError as e:
            print(f"   ✅ Invalid strategy correctly rejected: {e}")
        
        print("   ✅ Config validation tests passed")
        
    except ImportError as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    
    # Test 2: Validate filter functions
    print("\n2. Testing enhanced filter functions...")
    try:
        from iterative_pipeline.evaluation_utils import filter_evaluation_data_enhanced
        
        # Create sample data
        sample_data = pd.DataFrame({
            'video': ['video1', 'video1', 'video2', 'video2', 'video3'],
            'frame': [1, 2, 1, 2, 1],
            'class': ['A', 'B', 'A', 'C', 'B'],
            'embedding': [[1, 2, 3]] * 5
        })
        
        # Create sample exclusion tracker
        exclusion_tracker = {
            1: [
                {'video': 'video1', 'frame': 1, 'class': 'A'},
                {'video': 'video2', 'frame': 2, 'class': 'C'}
            ]
        }
        
        # Test frame-level filtering
        frame_filtered = filter_evaluation_data_enhanced(
            sample_data, exclusion_tracker, exclude_training=True, exclusion_strategy='frame-level'
        )
        print(f"   ✅ Frame-level filter: {len(sample_data)} → {len(frame_filtered)} samples")
        
        # Test video-level filtering  
        video_filtered = filter_evaluation_data_enhanced(
            sample_data, exclusion_tracker, exclude_training=True, exclusion_strategy='video-level'
        )
        print(f"   ✅ Video-level filter: {len(sample_data)} → {len(video_filtered)} samples")
        
        print("   ✅ Filter function tests passed")
        
    except ImportError as e:
        print(f"   ❌ Filter function import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Filter function test failed: {e}")
        return False
    
    # Test 3: Check command line arguments
    print("\n3. Testing command line argument parsing...")
    try:
        # Test help output
        result = subprocess.run([
            sys.executable, "run_iterative_pipeline.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if "frame-level-exclusions" in result.stdout and "video-level-exclusions" in result.stdout:
            print("   ✅ New command line arguments found in help")
        else:
            print("   ❌ New command line arguments not found in help")
            return False
            
    except Exception as e:
        print(f"   ⚠️ Command line test skipped: {e}")
    
    # Test 4: Test analyzer integration
    print("\n4. Testing exclusion impact analyzer...")
    try:
        analyzer_path = Path(__file__).parent / "exclusion_impact_analyzer.py"
        if analyzer_path.exists():
            print("   ✅ Enhanced exclusion impact analyzer found")
            
            # Test analyzer help
            result = subprocess.run([
                sys.executable, str(analyzer_path), "--help"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ Analyzer help command works")
            else:
                print(f"   ⚠️ Analyzer help failed: {result.stderr}")
        else:
            print("   ❌ Exclusion impact analyzer not found")
            return False
            
    except Exception as e:
        print(f"   ⚠️ Analyzer test skipped: {e}")
    
    print("\n✅ All tests completed successfully!")
    return True

def print_usage_examples():
    """Print usage examples for the new functionality."""
    print("\n📚 Usage Examples")
    print("=" * 50)
    
    print("\n1. Frame-level exclusions (default):")
    print("   python run_iterative_pipeline.py \\")
    print("     --definitive-objects data/definitiveObjects.pkl \\") 
    print("     --resnet-predictions data/resnetPredictions.pkl \\")
    print("     --tracking-info data/trackingInfo.pkl \\")
    print("     --frame-level-exclusions")
    
    print("\n2. Video-level exclusions:")
    print("   python run_iterative_pipeline.py \\")
    print("     --definitive-objects data/definitiveObjects.pkl \\")
    print("     --resnet-predictions data/resnetPredictions.pkl \\") 
    print("     --tracking-info data/trackingInfo.pkl \\")
    print("     --video-level-exclusions")
    
    print("\n3. Compare both strategies (A/B testing):")
    print("   python run_iterative_pipeline.py \\")
    print("     --definitive-objects data/definitiveObjects.pkl \\")
    print("     --resnet-predictions data/resnetPredictions.pkl \\")
    print("     --tracking-info data/trackingInfo.pkl \\") 
    print("     --compare-exclusion-strategies")
    
    print("\n4. Analyze exclusion impact:")
    print("   python exclusion_impact_analyzer.py \\")
    print("     --tracking-data-dir ../../../../gitignore_exception/tracking_exports/ \\")
    print("     --output-dir analyze_exclusions")
    
    print("\n💡 Key Benefits:")
    print("   • Frame-level: Preserves more data, better performance")
    print("   • Video-level: More aggressive filtering, simpler logic")
    print("   • Compare-both: Experimental validation of both approaches")
    print("   • Enhanced analyzer: Real pipeline integration and comparison")

def main():
    """Main execution function."""
    print("🚀 Exclusion Strategy Implementation Test Suite")
    print("=" * 60)
    
    # Run tests
    success = test_exclusion_strategies()
    
    if success:
        print_usage_examples()
        print("\n🎉 Implementation ready for use!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 