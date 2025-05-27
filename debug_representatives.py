import pickle
import numpy as np

def debug_representatives(experiment_name):
    """Debug the structure of representatives"""
    path = f"gitignore_exception/org/{experiment_name}/representatives.pkl"
    
    print(f"\n🔍 Debugging {experiment_name}:")
    print("=" * 40)
    
    try:
        with open(path, 'rb') as f:
            representatives = pickle.load(f)
        
        print(f"Type: {type(representatives)}")
        
        if isinstance(representatives, dict):
            print(f"Keys: {list(representatives.keys())}")
            print(f"Number of classes: {len(representatives)}")
            
            # Check first few items
            for i, (key, value) in enumerate(list(representatives.items())[:3]):
                print(f"\nClass '{key}':")
                print(f"  Type: {type(value)}")
                print(f"  Shape: {getattr(value, 'shape', 'No shape attribute')}")
                if hasattr(value, 'dtype'):
                    print(f"  Dtype: {value.dtype}")
                if isinstance(value, (list, tuple)):
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        print(f"  First element type: {type(value[0])}")
                        print(f"  First element: {value[0] if not isinstance(value[0], np.ndarray) else f'Array shape: {value[0].shape}'}")
        
        elif isinstance(representatives, (list, tuple)):
            print(f"Length: {len(representatives)}")
            if len(representatives) > 0:
                print(f"First element type: {type(representatives[0])}")
                print(f"First element shape: {getattr(representatives[0], 'shape', 'No shape')}")
        
        elif isinstance(representatives, np.ndarray):
            print(f"Shape: {representatives.shape}")
            print(f"Dtype: {representatives.dtype}")
        
        else:
            print(f"Unknown structure: {representatives}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    experiments = ['quick_test_agg', 'quick_test_balanced', 'very_agg']
    
    for exp in experiments:
        debug_representatives(exp)

if __name__ == "__main__":
    main() 