from e2e_pipeline_v2.experiments.blurriness_filter.blurriness_filter import blurriness_filter
import subprocess
import sys

# Test with different filtering methods
print("Running blurriness filter tests...")

# 1. Test filtering by laplacian using percentile method
print("\nTest 1: Filter by laplacian (percentile method)")
blurriness_filter(
    'combined_search_results_resnet50.json',
    'e2e_pipeline_v2/experiments/agg_results.csv',
    'filtered_laplacian_top25.json',
    metric='laplacian',
    method='percentile',
    threshold=75
)

# 2. Test filtering by tenengrad using standard deviation method
print("\nTest 2: Filter by tenengrad (standard deviation method)")
blurriness_filter(
    'combined_search_results_resnet50.json',
    'e2e_pipeline_v2/experiments/agg_results.csv',
    'filtered_tenengrad_std.json',
    metric='tenengrad',
    method='std_dev',
    threshold=1.0
)

# 3. Test filtering by is_blurry_fixed (boolean threshold)
print("\nTest 3: Filter by is_blurry_fixed (threshold method)")
blurriness_filter(
    'combined_search_results_resnet50.json',
    'e2e_pipeline_v2/experiments/agg_results.csv',
    'filtered_nonblurry.json',
    metric='is_blurry_fixed',
    method='threshold',
    threshold=False
)

# 4. Demonstrate command line usage
print("\nTest 4: Demonstrating command line usage")
print("The following commands can be used to run the same tests from the command line:")

cmd1 = [
    sys.executable, 
    "-m", "e2e_pipeline_v2.experiments.blurriness_filter.blurriness_filter",
    "--json_path", "combined_search_results_resnet50.json",
    "--csv_path", "e2e_pipeline_v2/experiments/agg_results.csv",
    "--output_path", "filtered_laplacian_top25_cli.json",
    "--metric", "laplacian",
    "--method", "percentile",
    "--threshold", "75"
]
print("\nCommand for Test 1:")
print(" ".join(cmd1))

cmd2 = [
    sys.executable,
    "-m", "e2e_pipeline_v2.experiments.blurriness_filter.blurriness_filter",
    "--json_path", "combined_search_results_resnet50.json",
    "--csv_path", "e2e_pipeline_v2/experiments/agg_results.csv",
    "--output_path", "filtered_tenengrad_std_cli.json",
    "--metric", "tenengrad",
    "--method", "std_dev",
    "--threshold", "1.0"
]
print("\nCommand for Test 2:")
print(" ".join(cmd2))

cmd3 = [
    sys.executable,
    "-m", "e2e_pipeline_v2.experiments.blurriness_filter.blurriness_filter",
    "--json_path", "combined_search_results_resnet50.json",
    "--csv_path", "e2e_pipeline_v2/experiments/agg_results.csv",
    "--output_path", "filtered_nonblurry_cli.json",
    "--metric", "is_blurry_fixed",
    "--method", "threshold"
    # No --boolean_threshold flag means it defaults to False
]
print("\nCommand for Test 3:")
print(" ".join(cmd3))

print("\nAll tests completed. Check the output JSON files.") 