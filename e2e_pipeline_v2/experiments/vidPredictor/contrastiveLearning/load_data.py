import pickle 
import pandas as pd
from collections import Counter
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
    return pd.DataFrame(data)


df =load_pickle('/Users/andrewdelacruz/e2e_sam2/gitignore_exception/contrastive_embeddings_viz_e20_t0.3_dim2048/prototypes.pkl')
print(df.head())
print(df.columns)
print(df)

