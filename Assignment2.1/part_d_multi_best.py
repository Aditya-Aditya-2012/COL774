import os
import numpy as np
import pickle
import argparse
from preprocessor import CustomImageDataset, DataLoader, numpy_transform  
import time

np.random.seed(0)

# Dataloader
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

# Neural Network Architecture
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = {
            "fc1": np.random.randn(625, 512) * np.sqrt(2/(625)),
            "fc2": np.random.randn(512, 256) * np.sqrt(2/(512)),
            "fc3": np.random.randn(256, 128) * np.sqrt(2/(256)),
            "fc4": np.random.randn(128, 32) * np.sqrt(2/(128)),
            "fc5": np.random.randn(32, 16) * np.sqrt(2/(32)),
            "fc6": np.random.randn(16, 8) * np.sqrt(2/(16))
        }
        self.biases = {
            "b1": np.zeros((512,), dtype=np.float64),
            "b2": np.zeros((256,), dtype=np.float64),
            "b3": np.zeros((128,), dtype=np.float64),
            "b4": np.zeros((32,), dtype=np.float64),
            "b5": np.zeros((16,), dtype=np.float64),
            "b6": np.zeros((8,), dtype=np.float64)
        }
        self.weights = {k: v.astype(np.float64) for k, v in self.weights.items()}
        self.biases = {k: v.astype(np.float64) for k, v in self.biases.items()}
        self.learning_rate = learning_rate

        self.velocities = {}
        self.momentums = {}
        self.rms_cache = {}

        for layer in self.weights:
            self.velocities[layer] = np.zeros_like(self.weights[layer])  # Momentum storage
            self.momentums[layer] = np.zeros_like(self.weights[layer])   # Adam first moment
            self.rms_cache[layer] = np.zeros_like(self.weights[layer])

        for layer in self.biases:
            self.velocities[layer] = np.zeros_like(self.biases[layer])
            self.momentums[layer] = np.zeros_like(self.biases[layer])
            self.rms_cache[layer] = np.zeros_like(self.biases[layer])


    def softmax(self, X):
        m=np.max(X,axis=1).reshape(-1,1)        
        e=np.exp(X-m)
        s=np.sum(e,axis=1).reshape(-1,1)
        return e/s

    def relu(self,x):
        return np.maximum(-0.001*x, x)
    
    def relu_derivative(self, x):
        x[x<=0] = 0.001
        x[x>0] = 1
        return x

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights["fc1"]) + self.biases["b1"]
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.weights["fc2"]) + self.biases["b2"]
        self.a2 = self.relu(self.z2)

        self.z3 = np.dot(self.a2, self.weights["fc3"]) + self.biases["b3"]
        self.a3 = self.relu(self.z3)

        self.z4 = np.dot(self.a3, self.weights["fc4"]) + self.biases["b4"]
        self.a4 = self.relu(self.z4)

        self.z5 = np.dot(self.a4, self.weights["fc5"]) + self.biases["b5"]
        self.a5 = self.relu(self.z5)

        self.z6 = np.dot(self.a5, self.weights["fc6"]) + self.biases["b6"]
        self.a6 = self.softmax(self.z6)

        return self.a6

    def backward(self, X, Y, output, t):
        m = Y.shape[0]
        
        # Output layer (Layer 6)
        output_delta = output - Y

        dw_6 = (self.a5.T @ output_delta) / m
        db_6 = np.mean(output_delta, axis=0)
        
        he_5 = output_delta @ self.weights["fc6"].T
        hd_5 = he_5 * self.relu_derivative(self.a5)

        # Layer 5
        dw_5 = (self.a4.T @ hd_5) / m
        db_5 = np.mean(hd_5, axis=0)
        
        he_4 = hd_5 @ self.weights["fc5"].T
        hd_4 = he_4 * self.relu_derivative(self.a4)

        # Layer 4
        dw_4 = (self.a3.T @ hd_4) / m
        db_4 = np.mean(hd_4, axis=0)
        
        he_3 = hd_4 @ self.weights["fc4"].T
        hd_3 = he_3 * self.relu_derivative(self.a3)

        # Layer 3
        dw_3 = (self.a2.T @ hd_3) / m
        db_3 = np.mean(hd_3, axis=0)
        
        he_2 = hd_3 @ self.weights["fc3"].T
        hd_2 = he_2 * self.relu_derivative(self.a2)

        # Layer 2
        dw_2 = (self.a1.T @ hd_2) / m
        db_2 = np.mean(hd_2, axis=0)
        
        he_1 = hd_2 @ self.weights["fc2"].T
        hd_1 = he_1 * self.sigmoid_derivative(self.a1)

        # Layer 1
        dw_1 = (X.T @ hd_1) / m
        db_1 = np.mean(hd_1, axis=0)

        # Adam optimizer parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        # Adam update for Layer 6
        self.momentums["fc6"] = beta1 * self.momentums["fc6"] + (1 - beta1) * dw_6
        self.rms_cache["fc6"] = beta2 * self.rms_cache["fc6"] + (1 - beta2) * dw_6 ** 2
        m_hat_6 = self.momentums["fc6"] / (1 - beta1 ** t)
        v_hat_6 = self.rms_cache["fc6"] / (1 - beta2 ** t)
        self.weights["fc6"] -= self.learning_rate * m_hat_6 / (np.sqrt(v_hat_6) + epsilon)

        self.momentums["b6"] = beta1 * self.momentums["b6"] + (1 - beta1) * db_6
        self.rms_cache["b6"] = beta2 * self.rms_cache["b6"] + (1 - beta2) * db_6 ** 2
        mb_hat_6 = self.momentums["b6"] / (1 - beta1 ** t)
        vb_hat_6 = self.rms_cache["b6"] / (1 - beta2 ** t)
        self.biases["b6"] -= self.learning_rate * mb_hat_6 / (np.sqrt(vb_hat_6) + epsilon)

        # Adam update for Layer 5
        self.momentums["fc5"] = beta1 * self.momentums["fc5"] + (1 - beta1) * dw_5
        self.rms_cache["fc5"] = beta2 * self.rms_cache["fc5"] + (1 - beta2) * dw_5 ** 2
        m_hat_5 = self.momentums["fc5"] / (1 - beta1 ** t)
        v_hat_5 = self.rms_cache["fc5"] / (1 - beta2 ** t)
        self.weights["fc5"] -= self.learning_rate * m_hat_5 / (np.sqrt(v_hat_5) + epsilon)

        self.momentums["b5"] = beta1 * self.momentums["b5"] + (1 - beta1) * db_5
        self.rms_cache["b5"] = beta2 * self.rms_cache["b5"] + (1 - beta2) * db_5 ** 2
        mb_hat_5 = self.momentums["b5"] / (1 - beta1 ** t)
        vb_hat_5 = self.rms_cache["b5"] / (1 - beta2 ** t)
        self.biases["b5"] -= self.learning_rate * mb_hat_5 / (np.sqrt(vb_hat_5) + epsilon)

        # Adam update for Layer 4
        self.momentums["fc4"] = beta1 * self.momentums["fc4"] + (1 - beta1) * dw_4
        self.rms_cache["fc4"] = beta2 * self.rms_cache["fc4"] + (1 - beta2) * dw_4 ** 2
        m_hat_4 = self.momentums["fc4"] / (1 - beta1 ** t)
        v_hat_4 = self.rms_cache["fc4"] / (1 - beta2 ** t)
        self.weights["fc4"] -= self.learning_rate * m_hat_4 / (np.sqrt(v_hat_4) + epsilon)

        self.momentums["b4"] = beta1 * self.momentums["b4"] + (1 - beta1) * db_4
        self.rms_cache["b4"] = beta2 * self.rms_cache["b4"] + (1 - beta2) * db_4 ** 2
        mb_hat_4 = self.momentums["b4"] / (1 - beta1 ** t)
        vb_hat_4 = self.rms_cache["b4"] / (1 - beta2 ** t)
        self.biases["b4"] -= self.learning_rate * mb_hat_4 / (np.sqrt(vb_hat_4) + epsilon)

        # Adam update for Layer 3
        self.momentums["fc3"] = beta1 * self.momentums["fc3"] + (1 - beta1) * dw_3
        self.rms_cache["fc3"] = beta2 * self.rms_cache["fc3"] + (1 - beta2) * dw_3 ** 2
        m_hat_3 = self.momentums["fc3"] / (1 - beta1 ** t)
        v_hat_3 = self.rms_cache["fc3"] / (1 - beta2 ** t)
        self.weights["fc3"] -= self.learning_rate * m_hat_3 / (np.sqrt(v_hat_3) + epsilon)

        self.momentums["b3"] = beta1 * self.momentums["b3"] + (1 - beta1) * db_3
        self.rms_cache["b3"] = beta2 * self.rms_cache["b3"] + (1 - beta2) * db_3 ** 2
        mb_hat_3 = self.momentums["b3"] / (1 - beta1 ** t)
        vb_hat_3 = self.rms_cache["b3"] / (1 - beta2 ** t)
        self.biases["b3"] -= self.learning_rate * mb_hat_3 / (np.sqrt(vb_hat_3) + epsilon)

        # Adam update for Layer 2
        self.momentums["fc2"] = beta1 * self.momentums["fc2"] + (1 - beta1) * dw_2
        self.rms_cache["fc2"] = beta2 * self.rms_cache["fc2"] + (1 - beta2) * dw_2 ** 2
        m_hat_2 = self.momentums["fc2"] / (1 - beta1 ** t)
        v_hat_2 = self.rms_cache["fc2"] / (1 - beta2 ** t)
        self.weights["fc2"] -= self.learning_rate * m_hat_2 / (np.sqrt(v_hat_2) + epsilon)

        self.momentums["b2"] = beta1 * self.momentums["b2"] + (1 - beta1) * db_2
        self.rms_cache["b2"] = beta2 * self.rms_cache["b2"] + (1 - beta2) * db_2 ** 2
        mb_hat_2 = self.momentums["b2"] / (1 - beta1 ** t)
        vb_hat_2 = self.rms_cache["b2"] / (1 - beta2 ** t)
        self.biases["b2"] -= self.learning_rate * mb_hat_2 / (np.sqrt(vb_hat_2) + epsilon)

        # Adam update for Layer 1
        self.momentums["fc1"] = beta1 * self.momentums["fc1"] + (1 - beta1) * dw_1
        self.rms_cache["fc1"] = beta2 * self.rms_cache["fc1"] + (1 - beta2) * dw_1 ** 2
        m_hat_1 = self.momentums["fc1"] / (1 - beta1 ** t)
        v_hat_1 = self.rms_cache["fc1"] / (1 - beta2 ** t)
        self.weights["fc1"] -= self.learning_rate * m_hat_1 / (np.sqrt(v_hat_1) + epsilon)

        self.momentums["b1"] = beta1 * self.momentums["b1"] + (1 - beta1) * db_1
        self.rms_cache["b1"] = beta2 * self.rms_cache["b1"] + (1 - beta2) * db_1 ** 2
        mb_hat_1 = self.momentums["b1"] / (1 - beta1 ** t)
        vb_hat_1 = self.rms_cache["b1"] / (1 - beta2 ** t)
        self.biases["b1"] -= self.learning_rate * mb_hat_1 / (np.sqrt(vb_hat_1) + epsilon)



    def compute_loss(self, y_true, y_pred):

        y_pred = np.clip(y_pred, 1e-12, 1.0)
        n_samples = y_true.shape[0]
        cross_entropy = -np.sum(y_true * np.log(y_pred)) 
        return cross_entropy

    def train(self, X_train, Y_train, epochs=15, batch_size=256, optimizer='gd', adaptive = False):
        best_loss = float('inf')
        n_batches = Y_train.shape[0] / batch_size

        if(n_batches > (Y_train.shape[0] // batch_size)) :
            n_batches = 1 + (Y_train.shape[0] // batch_size)

        else :
            n_batches = Y_train.shape[0] // batch_size

        for epoch in range(epochs) :
            epoch_loss = 0.
            num_samples = 0
            k = 0
            start_time = time.time()
            for i in range(n_batches) :
                k+=1
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                X_batch, Y_batch = X_train[start_idx:end_idx], Y_train[start_idx:end_idx]
                Y_batch = one_hot(Y_batch)
                Y_pred = self.forward(X_batch)
                loss = self.compute_loss(Y_batch, Y_pred)
                epoch_loss += loss
                num_samples += Y_batch.shape[0]
                self.backward(X_batch, Y_batch, Y_pred, t=epoch+1)

            elapsed_time = time.time() - start_time
        
            if adaptive :
                k = 5e-7
                self.learning_rate /= (1 + k * (epoch+1))

            if (epoch_loss/num_samples) < best_loss:
                best_loss = epoch_loss/num_samples

            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_samples}, time taken: {elapsed_time}')

        self.save_weights()
        print(f'best training loss : {best_loss}')

    def save_weights(self):
        weights_dict = {'weights': self.weights, 'bias': self.biases}
        with open('weights_c.pkl', 'wb') as f:
            pickle.dump(weights_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--save', type=str, required=True, help='Path to save the weights.')

    args = parser.parse_args()

    # Load data
    train_loader, val_loader = load_data(args.dataset_root)

    X_train, Y_train = loader_to_numpy(train_loader)

    # Initialize neural network
    nn = NeuralNetwork(learning_rate=0.001)

    # Train the neural network
    nn.train(X_train, Y_train, epochs=1000, batch_size=256, optimizer='adam', adaptive=False)

    # Save the weights
    # nn.save_weights()
