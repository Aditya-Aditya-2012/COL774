import os
import numpy as np
import pickle
import argparse
from part_d_trainloader import TrainImageDataset, TrainDataLoader, numpy_transform 
from part_d_testloader import TestImageDataset, TestDataLoader, numpy_transform 
import math
import time

np.random.seed(0)

def erf(x):
    return np.vectorize(math.erf)(x)

# Dataloader
def load_train_data(dataset_root):
    train_csv = os.path.join(dataset_root, 'train.csv')
    val_csv = os.path.join(dataset_root, 'val.csv')

    train_dataset = TrainImageDataset(root_dir=dataset_root, csv=train_csv, transform=numpy_transform)
    val_dataset = TrainImageDataset(root_dir=dataset_root, csv=val_csv, transform=numpy_transform)

    train_loader = TrainDataLoader(dataset=train_dataset, batch_size=256)
    val_loader = TrainDataLoader(dataset=val_dataset, batch_size=1)

    return train_loader, val_loader

def load_test_data(dataset_root):
    test_csv = os.path.join(dataset_root, "val.csv")
    test_dataset = TestImageDataset(root_dir=dataset_root, csv=test_csv, transform=numpy_transform)
    test_loader = TestDataLoader(dataset=test_dataset, batch_size=1)

    return test_loader

def train_loader_to_numpy(loader):
    images_list = []
    labels_list = []
    for images, labels in loader:
        images_list.append(images)
        labels_list.append(labels)
    return np.vstack(images_list), np.hstack(labels_list)
    # Convert loaders to numpy arrays

def test_loader_to_numpy(loader):
    images_list = []
    for images in loader:
        images_list.append(images)
    return np.vstack(images_list) #only gives X_test

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
            "fc4": np.random.randn(128, 64) * np.sqrt(2/(128)),
            "fc5": np.random.randn(64, 8) * np.sqrt(2/(64))
        }
        self.biases = {
            "b1": np.zeros((512,), dtype=np.float64),
            "b2": np.zeros((256,), dtype=np.float64),
            "b3": np.zeros((128,), dtype=np.float64),
            "b4": np.zeros((64,), dtype=np.float64),
            "b5": np.zeros((8,), dtype=np.float64)
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


    def relu(self,x):
        return np.maximum(-0.001*x, x)
    
    def relu_derivative(self, x):
        x[x<=0] = 0.001
        x[x>0] = 1
        return x
    
    def softmax(self, X):
        m=np.max(X,axis=1).reshape(-1,1)        
        e=np.exp(X-m)
        s=np.sum(e,axis=1).reshape(-1,1)
        return e/s

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def swish(self, x):
        return x * self.sigmoid(x)

    def swish_derivative(self, x):
        sig_x = self.sigmoid(x)
        return sig_x + x * sig_x * (1 - sig_x)
    
    def gelu(self, x):
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    def gelu_derivative(self, x):
        return 0.5 * (1 + erf(x / np.sqrt(2))) + (x * np.exp(-x**2 / 2)) / np.sqrt(2 * np.pi)


    def forward(self, X):
        self.z1 = np.dot(X, self.weights["fc1"]) + self.biases["b1"]
        self.a1 = self.gelu(self.z1)

        self.z2 = np.dot(self.a1, self.weights["fc2"]) + self.biases["b2"]
        self.a2 = self.gelu(self.z2)

        self.z3 = np.dot(self.a2, self.weights["fc3"]) + self.biases["b3"]
        self.a3 = self.gelu(self.z3)

        self.z4 = np.dot(self.a3, self.weights["fc4"]) + self.biases["b4"]
        self.a4 = self.gelu(self.z4)

        self.z5 = np.dot(self.a4, self.weights["fc5"]) + self.biases["b5"]
        self.a5 = self.softmax(self.z5)

        return self.a5

    def backward(self, X, Y, output, optimizer, t):
        m = Y.shape[0]
        
        output_delta = output - Y

        dw_5 = (self.a4.T @ output_delta) / m
        db_5 = np.mean(output_delta, axis = 0)
        
        he_4 = output_delta @ self.weights["fc5"].T
        hd_4 = he_4 * self.gelu_derivative(self.z4)

        dw_4 = (self.a3.T @ hd_4) / m
        db_4 = np.mean(hd_4, axis = 0)
        
        he_3 = hd_4 @ self.weights["fc4"].T
        hd_3 = he_3 * self.gelu_derivative(self.z3)

        dw_3 = (self.a2.T @ hd_3) / m
        db_3 = np.mean(hd_3, axis = 0)
        
        he_2 = hd_3 @ self.weights["fc3"].T
        hd_2 = he_2 * self.gelu_derivative(self.z2)

        dw_2 = (self.a1.T @ hd_2) / m
        db_2 = np.mean(hd_2, axis = 0)
        
        he_1 = hd_2 @ self.weights["fc2"].T
        hd_1 = he_1 * self.gelu_derivative(self.z1)

        dw_1 = (X.T @ hd_1) / m
        db_1 = np.mean(hd_1, axis = 0)

        if optimizer == 'gd' :
            self.weights["fc5"] -= self.learning_rate * dw_5
            self.biases["b5"] -= self.learning_rate * db_5

            self.weights["fc4"] -= self.learning_rate * dw_4
            self.biases["b4"] -= self.learning_rate * db_4

            self.weights["fc3"] -= self.learning_rate * dw_3
            self.biases["b3"] -= self.learning_rate * db_3

            self.weights["fc2"] -= self.learning_rate * dw_2
            self.biases["b2"] -= self.learning_rate * db_2

            self.weights["fc1"] -= self.learning_rate * dw_1
            self.biases["b1"] -= self.learning_rate * db_1

        if optimizer == 'momentum' :
            theta = 0.9 
            
            self.velocities["fc5"] = theta * self.velocities["fc5"] + self.learning_rate * dw_5
            self.weights["fc5"] -= self.velocities["fc5"]

            self.velocities["b5"] = theta * self.velocities["b5"] + self.learning_rate * db_5
            self.biases["b5"] -= self.velocities["b5"]

            self.velocities["fc4"] = theta * self.velocities["fc4"] + self.learning_rate * dw_4
            self.weights["fc4"] -= self.velocities["fc4"]
            
            self.velocities["b4"] = theta * self.velocities["b4"] + self.learning_rate * db_4
            self.biases["b4"] -= self.velocities["b4"]


            self.velocities["fc3"] = theta * self.velocities["fc3"] + self.learning_rate * dw_3
            self.weights["fc3"] -= self.velocities["fc3"]

            self.velocities["b3"] = theta * self.velocities["b3"] + self.learning_rate * db_3
            self.biases["b3"] -= self.velocities["b3"]
 

            self.velocities["fc2"] = theta * self.velocities["fc2"] + self.learning_rate * dw_2
            self.weights["fc2"] -= self.velocities["fc2"]

            self.velocities["b2"] = theta * self.velocities["b2"] + self.learning_rate * db_2
            self.biases["b2"] -= self.velocities["b2"] 

            self.velocities["fc1"] = theta * self.velocities["fc1"] + self.learning_rate * dw_1
            self.weights["fc1"] -= self.velocities["fc1"]
            
            self.velocities["b1"] = theta * self.velocities["b1"] + self.learning_rate * db_1
            self.biases["b1"] -= self.velocities["b1"]

        elif optimizer == 'rmsprop' :
            beta2 = 0.999  # Decay factor
            epsilon = 1e-8

            self.rms_cache["fc5"] = beta2 * self.rms_cache["fc5"] + (1 - beta2) * dw_5 ** 2
            self.weights["fc5"] -= self.learning_rate * dw_5 / (np.sqrt(self.rms_cache["fc5"]) + epsilon)

            self.rms_cache["b5"] = beta2 * self.rms_cache["b5"] + (1 - beta2) * db_5 ** 2
            self.biases["b5"] -= self.learning_rate * db_5 / (np.sqrt(self.rms_cache["b5"]) + epsilon)

            self.rms_cache["fc4"] = beta2 * self.rms_cache["fc4"] + (1 - beta2) * dw_4 ** 2
            self.weights["fc4"] -= self.learning_rate * dw_4 / (np.sqrt(self.rms_cache["fc4"]) + epsilon)

            self.rms_cache["b4"] = beta2 * self.rms_cache["b4"] + (1 - beta2) * db_4 ** 2
            self.biases["b4"] -= self.learning_rate * db_4 / (np.sqrt(self.rms_cache["b4"]) + epsilon)

            self.rms_cache["fc3"] = beta2 * self.rms_cache["fc3"] + (1 - beta2) * dw_3 ** 2
            self.weights["fc3"] -= self.learning_rate * dw_3 / (np.sqrt(self.rms_cache["fc3"]) + epsilon)

            self.rms_cache["b3"] = beta2 * self.rms_cache["b3"] + (1 - beta2) * db_3 ** 2
            self.biases["b3"] -= self.learning_rate * db_3 / (np.sqrt(self.rms_cache["b3"]) + epsilon)

            self.rms_cache["fc2"] = beta2 * self.rms_cache["fc2"] + (1 - beta2) * dw_2 ** 2
            self.weights["fc2"] -= self.learning_rate * dw_2 / (np.sqrt(self.rms_cache["fc2"]) + epsilon)

            self.rms_cache["b2"] = beta2 * self.rms_cache["b2"] + (1 - beta2) * db_2 ** 2
            self.biases["b2"] -= self.learning_rate * db_2 / (np.sqrt(self.rms_cache["b2"]) + epsilon)

            self.rms_cache["fc1"] = beta2 * self.rms_cache["fc1"] + (1 - beta2) * dw_1 ** 2
            self.weights["fc1"] -= self.learning_rate * dw_1 / (np.sqrt(self.rms_cache["fc1"]) + epsilon)

            self.rms_cache["b1"] = beta2 * self.rms_cache["b1"] + (1 - beta2) * db_1 ** 2
            self.biases["b1"] -= self.learning_rate * db_1 / (np.sqrt(self.rms_cache["b1"]) + epsilon)

        elif optimizer == 'adam' :
            beta1 = 0.9  # Decay factor for the first moment
            beta2 = 0.99 # Decay factor for the second moment
            epsilon = 1e-8

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

    def train(self, X_train, Y_train, X_val, Y_val, X_test, epochs=15, batch_size=256, optimizer='gd', wt_path='results', pr_path='results', time_limiter = 875):
        # wt_directory = wt_path
        # wt_file_name = "weights.pkl"
        wt_file_path = wt_path

        # pr_directory = pr_path
        # pr_file_name = "predictions.pkl"
        pr_file_path = pr_path

        # if not os.path.exists(wt_directory):
        #     os.makedirs(wt_directory)

        # if not os.path.exists(pr_directory):
        #     os.makedirs(pr_directory)

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

                self.backward(X_batch, Y_batch, Y_pred, optimizer, t=epoch+1)
        
            out_val = self.forward(X_val)
            loss_val = self.compute_loss(one_hot(Y_val), out_val) / Y_val.shape[0]

            out_test = self.forward(X_test)

            print(f'loss: {loss_val}')
            if (loss_val) < best_loss:
                best_loss = loss_val
                self.save_weights(wt_file_path)
                self.save_predictions(pr_file_path, out_test)

            elapsed_time = time.time() - start_time
            if(elapsed_time > time_limiter) :
                # print(f'training stopped after {elapsed_time} seconds and {epoch+1} epochs, loss : {best_loss}')
                break

    def save_weights(self, file_path):
        new_dict = {key.replace("b", "fc"): value for key, value in self.biases.items()}
        weights_dict = {'weights': self.weights, 'bias': new_dict}
        with open(file_path, 'wb') as f:
            pickle.dump(weights_dict, f)
    
    def save_predictions(self, file_path, output):
        predictions = output.argmax(axis=1)
        predictions = np.array(predictions)
        with open(file_path, 'wb') as f:
            pickle.dump(predictions, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--test_dataset_root', type=str, required=True, help='Root directory of the test dataset.')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights.')
    parser.add_argument('--save_predictions_path', type=str, required=True, help='Path to save the predictions.')

    args = parser.parse_args()

    start_time = time.time()
    train_loader, val_loader = load_train_data(args.dataset_root)

    X_train, Y_train = train_loader_to_numpy(train_loader)
    X_val, Y_val = train_loader_to_numpy(val_loader)

    test_loader = load_test_data(args.test_dataset_root)
    X_test = test_loader_to_numpy(test_loader)

    nn = NeuralNetwork(learning_rate=0.001)

    nn.train(X_train, Y_train, X_val, Y_val, X_test, epochs=5000, batch_size=256, optimizer='adam', wt_path=args.save_weights_path, pr_path=args.save_predictions_path, time_limiter=120)
