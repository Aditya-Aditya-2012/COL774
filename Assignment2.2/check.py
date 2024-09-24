import pickle

# Path to your predictions.pkl file
predictions_path = 'predictions.pkl'

# Load the predictions from the pickle file
with open(predictions_path, 'rb') as file:
    predictions = pickle.load(file)

# Print the predictions
print(predictions)
