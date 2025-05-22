import pickle 
import pandas as pd
from collections import Counter
import numpy as np
def load_pickle(file_path):
    """
    Load a pickle file and return the data.
    
    Args:
        file_path (str): Path to the pickle file.
        
    Returns:
        data: Data loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(f"Loaded {data} from {file_path}")
    

    # Assuming `data` is the loaded dictionary
    representatives = data['representatives']       # Shape: (18, 2048)
    class_names = data['class_names']               # List of 18 class names
    # Create the DataFrame with class names and their representative embeddings
    # Ensure each embedding is explicitly np.float32
    embeddings = [np.array(vec, dtype=np.float32) for vec in representatives]

    # Create the DataFrame
    df = pd.DataFrame({
        'class_name': class_names,
        'embedding': embeddings
    })


    # # Optionally, name the columns for clarity (e.g., feature_0, feature_1, ..., feature_2047)
    # df.columns = [f'feature_{i}' for i in range(representatives.shape[1])]

    return df  # Save to CSV


df =load_pickle('/home/ubuntu/code/drew/e2e_sam2/data/representatives.pkl')
df.to_csv('/home/ubuntu/code/drew/e2e_sam2/data/representatives.csv', index=False)
print(type(df['embedding'][0]), df['embedding'][0].dtype)


