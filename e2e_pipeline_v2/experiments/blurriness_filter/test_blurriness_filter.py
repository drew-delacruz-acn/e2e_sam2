from blurriness_filter import blurriness_filter

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

print("\nAll tests completed. Check the output JSON files.") 