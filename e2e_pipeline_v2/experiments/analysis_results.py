import pandas as pd
import os
import argparse

def read_csvs_from_subdirs(directory_path: str) -> pd.DataFrame:
    """
    Reads a CSV file from each subdirectory within the given directory
    and concatenates them into a single DataFrame.

    Args:
        directory_path: The path to the main directory containing subdirectories.
    
    Returns:
        A pandas DataFrame containing all data from the CSVs, or an empty
        DataFrame if no CSVs are found or an error occurs.
    """
    all_dataframes = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            for sub_item in os.listdir(item_path):
                if sub_item.endswith('.csv'):
                    csv_path = os.path.join(item_path, sub_item)
                    try:
                        df = pd.read_csv(csv_path)
                        print(f"Successfully read {csv_path} (shape: {df.shape})")
                        print(f"Columns in {sub_item}: {list(df.columns)}")  # Debug: Print columns
                        all_dataframes.append(df)
                        # Assuming one CSV per subdirectory for this logic
                        break 
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

    if not all_dataframes:
        print("No CSV files were successfully read.")
        return pd.DataFrame() # Return an empty DataFrame
    
    try:
        # Print column names of first DataFrame for reference
        if all_dataframes:
            print(f"First DataFrame columns before concat: {list(all_dataframes[0].columns)}")
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Successfully combined {len(all_dataframes)} CSV file(s) into a DataFrame with shape: {combined_df.shape}")
        print(f"Combined DataFrame columns: {list(combined_df.columns)}")  # Debug: Print combined columns
        
        return combined_df
    except Exception as e:
        print(f"Error concatenating DataFrames: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

# Example usage (commented out):
# if __name__ == '__main__':
#     # Create dummy directories and CSV files for testing
#     if not os.path.exists("main_dir/subdir1"):
#         os.makedirs("main_dir/subdir1")
#     if not os.path.exists("main_dir/subdir2"):
#         os.makedirs("main_dir/subdir2")
#     pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_csv("main_dir/subdir1/data1.csv", index=False)
#     pd.DataFrame({'colA': ['a', 'b'], 'colB': ['c', 'd']}).to_csv("main_dir/subdir2/data2.csv", index=False)
#
#     read_csvs_from_subdirs("main_dir")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read CSVs from subdirectories of a main directory and combine them.")
    parser.add_argument("--main_dir", type=str, help="Path to the main directory.")
    parser.add_argument("--output_file", type=str, help="Name for the combined output CSV file.")
    args = parser.parse_args()

    main_dir_name = args.main_dir
    if not main_dir_name:
        main_dir_name = input("Please enter the path to the main directory: ")

    # Dummy data creation part (for testing)
    print(f"Using main directory: {main_dir_name}")
    subdir1_path = os.path.join(main_dir_name, "subdir1")
    subdir2_path = os.path.join(main_dir_name, "subdir2")

    if not os.path.exists(subdir1_path):
        print(f"Creating dummy directory: {subdir1_path}")
        os.makedirs(subdir1_path)
    if not os.path.exists(subdir2_path):
        print(f"Creating dummy directory: {subdir2_path}")
        os.makedirs(subdir2_path)
    
    dummy_csv1_path = os.path.join(subdir1_path, "data1.csv")
    dummy_csv2_path = os.path.join(subdir2_path, "data2.csv")

    if not os.path.exists(dummy_csv1_path):
        print(f"Creating dummy CSV: {dummy_csv1_path}")
        # Now using same column names for test data
        pd.DataFrame({'column1': [1, 2], 'column2': [3, 4]}).to_csv(dummy_csv1_path, index=False)
    if not os.path.exists(dummy_csv2_path):
        print(f"Creating dummy CSV: {dummy_csv2_path}")
        # Now using same column names for test data
        pd.DataFrame({'column1': [5, 6], 'column2': [7, 8]}).to_csv(dummy_csv2_path, index=False)
    # End of dummy data creation

    combined_dataframe = read_csvs_from_subdirs(main_dir_name)

    if not combined_dataframe.empty:
        # Let's check the columns again before saving
        print(f"Columns before saving to CSV: {list(combined_dataframe.columns)}")
        
        output_filename = args.output_file
        if not output_filename:
            output_filename = input("Enter the name for the combined CSV file (e.g., combined_output.csv): ")
        
        try:
            combined_dataframe.to_csv(output_filename, index=False)
            print(f"Combined data successfully saved to {output_filename}")
            
            # Verify the columns in the saved file
            verification_df = pd.read_csv(output_filename)
            print(f"Columns in saved file {output_filename}: {list(verification_df.columns)}")
        except Exception as e:
            print(f"Error saving combined data to {output_filename}: {e}")
    else:
        print("No data to save as the combined DataFrame is empty.")

