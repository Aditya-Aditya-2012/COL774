import sys
import numpy as np
import pandas as pd

def evaluate_predictions(output_file, test_pred_file):
    # Load the predicted values from the output file
    y_pred = np.loadtxt(output_file)
    
    # Load the actual values from the test_pred.csv file, skipping the header
    test_pred_df = pd.read_csv(test_pred_file, header=0)  # Use header=0 to skip the first row
    y_actual = test_pred_df.iloc[:, 0].values  # Assuming the actual values are in the first column
    
    # Calculate the absolute errors
    errors = np.abs(y_pred - y_actual)
    
    # Sort the errors in ascending order
    sorted_errors = np.sort(errors)
    
    # Calculate the number of samples for the best 90%
    n = len(sorted_errors)
    cutoff_index = int(0.9 * n)
    
    # Select the top 90% errors
    top_90_errors = sorted_errors[:cutoff_index]
    
    # Calculate RMSE of the top 90% errors
    rmse_top_90 = np.sqrt(np.mean(top_90_errors**2))
    
    # Print the evaluation metric
    print(f"Root Mean Squared Error of the best 90% predictions: {rmse_top_90}")
    
    # Return the evaluation metric
    return rmse_top_90

if __name__ == "__main__":
    output_file = sys.argv[1]
    test_pred = sys.argv[2]

    # Example usage:
    rmse_top_90 = evaluate_predictions(output_file, test_pred)
