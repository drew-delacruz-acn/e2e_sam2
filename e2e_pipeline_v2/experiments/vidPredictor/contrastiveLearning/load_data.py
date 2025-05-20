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


df =load_pickle('/home/ubuntu/code/drew/e2e_sam2/data/definitiveObjects_jeremiah.pkl')
print(df.head())
print(df.columns)

print(len(df.iloc[0, 2]))

print(len(df.iloc[0, 3]))
print(df.iloc[:,0].value_counts())