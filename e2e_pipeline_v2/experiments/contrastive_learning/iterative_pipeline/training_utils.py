import subprocess
from pathlib import Path
import pickle
import pandas as pd # Added for type hinting, though not directly used for processing here

def train_contrastive_representatives(training_data: pd.DataFrame, 
                                   output_dir: Path,
                                   epochs: int, 
                                   margin_to_use: float
                                   ) -> Path:
    """
    Train contrastive learning representatives by calling the external train_representatives.py script.
    
    Args:
        training_data: DataFrame with 'class' and 'finetuned_embedding' columns.
        output_dir: Directory to save results from train_representatives.py.
        epochs: Number of training epochs for contrastive learning.
        margin_to_use: Contrastive learning margin to use for this training run.
        
    Returns:
        Path to the saved representatives.pkl file.
        
    Raises:
        FileNotFoundError: If train_representatives.py or the output representatives.pkl is not found.
        RuntimeError: If the training script fails.
        subprocess.TimeoutExpired: If the training script times out.
    """
    print(f"🏋️ Training contrastive representatives (Epochs: {epochs}, Margin: {margin_to_use})...")
    
    temp_data_path = output_dir / "temp_training_data.pkl"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    
    with open(temp_data_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    possible_train_script_paths = [
        Path(__file__).parent.parent / "train_representatives.py", 
        Path(__file__).parent.parent.parent / "train_representatives.py", 
        Path("train_representatives.py") 
    ]
    train_script_path = None
    for p in possible_train_script_paths:
        try:
            if p.exists():
                train_script_path = p
                print(f"Found train_representatives.py at: {train_script_path.resolve()}")
                break
        except Exception: 
            pass
            
    if train_script_path is None:
        raise FileNotFoundError(f"train_representatives.py not found at expected locations: {possible_train_script_paths}")

    cmd = [
        "python", str(train_script_path),
        "--data", str(temp_data_path),
        "--output", str(output_dir),
        "--epochs", str(epochs),
        "--margin", str(margin_to_use),
        "--no-auto-name"  
    ]
    print(f"🔧 Running training command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600) 
        
        if result.returncode != 0:
            print(f"❌ Training script failed with return code {result.returncode}:")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"train_representatives.py failed. See output above.")
        
        print(f"✅ Training script completed successfully.")
        if result.stdout:
            print(f"Training script STDOUT (last 10 lines):\n" + "\n".join(result.stdout.strip().split('\n')[-10:]))
        if result.stderr:
            print(f"Training script STDERR (last 10 lines):\n" + "\n".join(result.stderr.strip().split('\n')[-10:]))
            
    except subprocess.TimeoutExpired:
        print(f"❌ Training script timed out after 10 minutes.")
        raise
    except Exception as e:
        print(f"❌ An error occurred while running the training script: {e}")
        raise
    finally:
        if temp_data_path.exists():
            temp_data_path.unlink()
            print(f"🗑️ Deleted temporary training data: {temp_data_path}")
    
    representatives_path = output_dir / "representatives.pkl"
    if not representatives_path.exists():
        raise FileNotFoundError(f"Expected output file (representatives.pkl) not found in {output_dir} after training.")
    
    print(f"✅ Representatives saved to: {representatives_path}")
    return representatives_path 