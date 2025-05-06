import pandas as pd
import os
import argparse # Added import for argparse

def read_csvs_from_subdirs(directory_path: str):
    """
    Reads a CSV file from each subdirectory within the given directory.

    Args:
        directory_path: The path to the main directory containing subdirectories.
    """
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            # This is a subdirectory, look for a CSV file inside it
            for sub_item in os.listdir(item_path):
                if sub_item.endswith('.csv'):
                    csv_path = os.path.join(item_path, sub_item)
                    try:
                        df = pd.read_csv(csv_path)
                        print(f"Successfully read {csv_path}")
                        print(df.head())
                        # TODO: Add further processing for the DataFrame 'df'
                        # For now, we just read one CSV per subdirectory.
                        # If multiple CSVs are expected, this logic needs adjustment.
                        break # Assuming one CSV per subdirectory
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
            else:
                print(f"No CSV file found in subdirectory: {item_path}")

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
    parser = argparse.ArgumentParser(description="Read CSVs from subdirectories of a main directory.")
    parser.add_argument("--main_dir", type=str, help="Path to the main directory.")
    args = parser.parse_args()

    main_dir_name = args.main_dir
    if not main_dir_name:
        main_dir_name = input("Please enter the path to the main directory: ")

    # Create dummy directories and CSV files for testing if they don't exist
    # This part is for demonstration. You might want to remove or modify it
    # if you are working with existing directories.
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
        pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_csv(dummy_csv1_path, index=False)
    if not os.path.exists(dummy_csv2_path):
        print(f"Creating dummy CSV: {dummy_csv2_path}")
        pd.DataFrame({'colA': ['a', 'b'], 'colB': ['c', 'd']}).to_csv(dummy_csv2_path, index=False)

    read_csvs_from_subdirs(main_dir_name)

