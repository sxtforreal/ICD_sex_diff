import os
import sys
# Add current directory to path
sys.path.append('/workspace')

# Import the cox_model module
from cox_model import *

# Run a small test with only 3 iterations to verify everything works
print("Running Cox model test with 3 iterations...")
print("="*60)

try:
    # Run with 3 iterations for testing
    results_test, summary_test = multiple_random_splits_cox(clean_df, N=3)
    
    # Save test results
    summary_test.to_excel('/workspace/cox_test_results.xlsx', index=True, index_label='Model/Metric')
    print("\nTest completed successfully!")
    print("Results saved to cox_test_results.xlsx")
    
    # Show a preview of the results
    print("\n=== Preview of Results ===")
    print(summary_test.head(10))
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()