import os
import numpy as np
import pickle
import argparse
from preprocessor import CustomImageDataset, DataLoader, numpy_transform  
np.random.seed(0)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_data(dataset_root):
    train_csv = os.path.join(dataset_root, 'train.csv')
    val_csv = os.path.join(dataset_root, 'val.csv')

    train_dataset = CustomImageDataset(root_dir=dataset_root, csv=train_csv, transform=numpy_transform)
    val_dataset = CustomImageDataset(root_dir=dataset_root, csv=val_csv, transform=numpy_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=256)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    return train_loader, val_loader

def loader_to_numpy(loader):
    images_list = []
    labels_list = []
    for images, labels in loader:
        images_list.append(images)
        labels_list.append(labels)
    return np.vstack(images_list), np.hstack(labels_list)
    # Convert loaders to numpy arrays

def one_hot(Y) :
    n_classes = 8  
    Y_one_hot = np.eye(n_classes)[Y]
    return Y_one_hot

class NeuralNetwork:
    def __init__(self, learning_rate) :
        self.weights = {
                "fc1": np.random.randn(625, 512) * np.sqrt(2/(625)),
                "fc2": np.random.randn(512, 256) * np.sqrt(2/(512)),
                "fc3": np.random.randn(256, 128) * np.sqrt(2/(256)),
                "fc4": np.random.randn(128, 8) * np.sqrt(2/(128))
            }
        self.biases = {
                "fc1": np.zeros((512,), dtype=np.float64),
                "fc2": np.zeros((256,), dtype=np.float64),
                "fc3": np.zeros((128,), dtype=np.float64),
                "fc4": np.zeros((8,), dtype=np.float64)
            }
        
        self.weights = {k: v.astype(np.float64) for k, v in self.weights.items()}
        self.biases = {k: v.astype(np.float64) for k, v in self.biases.items()}
        self.learning_rate = learning_rate



    def softmax(self, X):
        m=np.max(X,axis=1).reshape(-1,1)        
        e=np.exp(X-m)
        s=np.sum(e,axis=1).reshape(-1,1)
        return e/s

    def sigmoid(self, z) :
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def forward(self, X) :
        self.z1 = np.dot(X, self.weights["fc1"]) + self.biases["fc1"]
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.weights["fc2"]) + self.biases["fc2"]
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.weights["fc3"]) + self.biases["fc3"]
        self.a3 = self.sigmoid(self.z3)

        self.z4 = np.dot(self.a3, self.weights["fc4"]) + self.biases["fc4"]
        self.a4 = self.softmax(self.z4)

        return self.a4

    def backward(self, X, Y, output) :
        m = Y.shape[0] 

        output_delta = output - Y

        he_3 = output_delta @ self.weights["fc4"].T
        hd_3 = he_3 * self.sigmoid_derivative(self.z3)

        he_2 = hd_3 @ self.weights["fc3"].T
        hd_2 = he_2 * self.sigmoid_derivative(self.z2)

        he_1 = hd_2 @ self.weights["fc2"].T
        hd_1 = he_1 * self.sigmoid_derivative(self.z1)

        self.weights["fc4"] -= self.learning_rate * ((self.a3.T @ output_delta) / m)
        self.biases["fc4"] -= self.learning_rate * np.mean(output_delta, axis=0) 

        self.weights["fc3"] -= self.learning_rate * ((self.a2.T @ hd_3) / m)
        self.biases["fc3"] -= self.learning_rate * np.mean(hd_3, axis=0) 

        self.weights["fc2"] -= self.learning_rate * ((self.a1.T @ hd_2) / m)
        self.biases["fc2"] -= self.learning_rate * np.mean(hd_2, axis=0) 

        self.weights["fc1"] -= self.learning_rate * ((X.T @ hd_1) / m)
        self.biases["fc1"] -= self.learning_rate * np.mean(hd_1, axis=0) 


    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        n_samples = y_true.shape[0]
        cross_entropy = -np.sum(y_true * np.log(y_pred))
        return cross_entropy

    def save_weights(self, file_path):
        weights_dict = {'weights': self.weights, 'bias': self.biases}
        with open(file_path, 'wb') as f:
            pickle.dump(weights_dict, f)

    def train(self, X_train, Y_train, epochs=15, batch_size=256, path='results') :
        n_batches = Y_train.shape[0] / batch_size

        if(n_batches > (Y_train.shape[0] // batch_size)) :
            n_batches = 1 + (Y_train.shape[0] // batch_size)

        else :
            n_batches = Y_train.shape[0] // batch_size

        for epoch in range(epochs) :
            for i in range(n_batches) :
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                X_batch, Y_batch = X_train[start_idx:end_idx], Y_train[start_idx:end_idx]

                Y_batch = one_hot(Y_batch)
                Y_pred = self.forward(X_batch)

                loss = self.compute_loss(Y_batch, Y_pred)

                self.backward(X_batch, Y_batch, Y_pred)

            self.save_weights(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights.')

    args = parser.parse_args()

    train_loader, valid_loader = load_data(args.dataset_root)
    X_train, Y_train = loader_to_numpy(train_loader)

    nn = NeuralNetwork(learning_rate=0.001)
    nn.train(X_train, Y_train, epochs=15, batch_size=256, path=args.save_weights_path)

    

    