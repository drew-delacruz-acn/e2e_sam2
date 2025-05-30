import os
import pandas as pd
import glob
from pathlib import Path
import argparse

def combine_blur_analysis_csvs(analysis_dir, output_file='combined_blur_analysis.csv'):
    """
    Combines all blur analysis CSV files in the specified directory into a single CSV.
    
    Args:
        analysis_dir: Path to the analysis directory containing scene folders
        output_file: Path to output the combined CSV file
    """
    # Find all CSV files matching the blur data pattern
    csv_pattern = os.path.join(analysis_dir, '**/blur_data_*.csv')
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Create an empty list to store dataframes
    dfs = []
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Extract scene name from the file path
            scene_folder = os.path.basename(os.path.dirname(csv_file))
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Add a column for the scene name
            df['scene'] = scene_folder
            
            # Append to our list of dataframes
            dfs.append(df)
            
            print(f"Processed: {scene_folder}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save to a new CSV file
        combined_df.to_csv(output_file, index=False)
        
        print(f"Combined {len(dfs)} CSV files into {output_file}")
        print(f"Total rows: {len(combined_df)}")
        
        return combined_df
    else:
        print("No data to combine.")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Combine blur analysis CSV files into a single CSV.')
    parser.add_argument('analysis_dir', type=str, help='Path to the analysis directory containing scene folders')
    parser.add_argument('--output', '-o', type=str, default='combined_blur_analysis.csv',
                        help='Path to output the combined CSV file (default: combined_blur_analysis.csv)')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Combine all CSV files
    combined_df = combine_blur_analysis_csvs(args.analysis_dir, args.output)
