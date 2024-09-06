import os
import numpy as np
import pickle
import argparse
from preprocessor import CustomImageDataset, DataLoader, numpy_transform  

np.random.seed(0)

# Neural Network Architecture
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = {
            "fc1": np.random.randn(625, 512) * np.sqrt(2.0 / 625),
            "fc2": np.random.randn(512, 256) * np.sqrt(2.0 / 512),
            "fc3": np.random.randn(256, 128) * np.sqrt(2.0 / 256),
            "fc4": np.random.randn(128, 1) * np.sqrt(2.0 / 128)
        }
        self.biases = {
            "fc1": np.zeros((256, 512)),
            "fc2": np.zeros((256, 256)),
            "fc3": np.zeros((256, 128)),
            "fc4": np.zeros((256, 1))
        }
        self.weights = {k: v.astype(np.float64) for k, v in self.weights.items()}
        self.biases = {k: v.astype(np.float64) for k, v in self.biases.items()}
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights["fc1"]) + self.biases["fc1"]
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.weights["fc2"]) + self.biases["fc2"]
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.weights["fc3"]) + self.biases["fc3"]
        self.a3 = self.sigmoid(self.z3)

        self.z4 = np.dot(self.a3, self.weights["fc4"]) + self.biases["fc4"]
        self.a4 = self.sigmoid(self.z4)

        return self.a4

    def backward(self, X, y, output):
        m = X.shape[0]

        print("weights['fc4'] shape:", self.weights["fc4"].shape)
        print("biases['fc4'] shape:", self.biases["fc4"].shape)

        print("output shape:", output.shape)


        # Compute the error at the output layer
        output_error = np.zeros((m, 1))

        for i in range(0, m) :
            output_error[i][0] = -((y[i]/output[i][0]) - (1-y[i]) / (1-output[i][0]))

        output_delta = np.transpose(self.sigmoid_derivative(output)) @ output_error

        print("output_delta shape:", output_delta.shape)
        print("self.weights['fc4'].T shape:", self.weights["fc4"].T.shape)

        # Compute errors and deltas for hidden layers
        hidden_error_3 = np.dot(output_delta.T, self.weights["fc4"].T)
        hidden_delta_3 = hidden_error_3 * self.sigmoid_derivative(self.a3)

        hidden_error_2 = np.dot(hidden_delta_3, self.weights["fc3"].T)
        hidden_delta_2 = hidden_error_2 * self.sigmoid_derivative(self.a2)

        hidden_error_1 = np.dot(hidden_delta_2, self.weights["fc2"].T)
        hidden_delta_1 = hidden_error_1 * self.sigmoid_derivative(self.a1)

        # Update weights and biases using gradient descent
        self.weights["fc4"] -= self.learning_rate * np.dot(self.a3.T, output_delta.T) / m
        self.biases["fc4"] -= self.learning_rate * np.sum(output_delta.T, axis=0, keepdims=True) / m

        self.weights["fc3"] -= self.learning_rate * np.dot(self.a2.T, hidden_delta_3) / m
        self.biases["fc3"] -= self.learning_rate * np.sum(hidden_delta_3, axis=0, keepdims=True) / m

        self.weights["fc2"] -= self.learning_rate * np.dot(self.a1.T, hidden_delta_2) / m
        self.biases["fc2"] -= self.learning_rate * np.sum(hidden_delta_2, axis=0, keepdims=True) / m

        self.weights["fc1"] -= self.learning_rate * np.dot(X.T, hidden_delta_1) / m
        self.biases["fc1"] -= self.learning_rate * np.sum(hidden_delta_1, axis=0, keepdims=True) / m

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -(1/m) * np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss

    def train(self, train_loader, epochs=15, batch_size=256):
        
        for i in range(epochs) :
            epoch_loss = 0.
            for X_batch, y_batch in train_loader :
                # Forward pass
                Y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_batch, Y_pred)
                epoch_loss += loss
                
                # Backward pass and weight update
                self.backward(X_batch, y_batch, Y_pred)
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches}')

        
        self.save_weights()

    def save_weights(self):
        weights_dict = {'weights': self.weights, 'bias': self.biases}
        print(weights_dict["weights"]["fc1"])
        with open('weights.pkl', 'wb') as f:
            pickle.dump(weights_dict, f)


def load_data(dataset_root):
    train_csv = os.path.join(dataset_root, 'train.csv')
    val_csv = os.path.join(dataset_root, 'val.csv')

    train_dataset = CustomImageDataset(root_dir=dataset_root, csv=train_csv, transform=numpy_transform)
    val_dataset = CustomImageDataset(root_dir=dataset_root, csv=val_csv, transform=numpy_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=256)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--save', type=str, required=True, help='Path to save the weights.')

    args = parser.parse_args()

    # Load data
    train_loader, val_loader = load_data(args.dataset_root)

    # Initialize neural network
    nn = NeuralNetwork(learning_rate=0.001)

    # Train the neural network
    nn.train(train_loader, epochs=15)

    # Save the weights
    nn.save_weights()
