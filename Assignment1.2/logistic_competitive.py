import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import time
from sklearn.feature_selection import SelectKBest, f_classif

def preprocess_data(train_path, test_path):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop the 'Birth Weight' column
    train_df = train_df.drop(columns=['Birth Weight'])
    test_df = test_df.drop(columns=['Birth Weight'])
    
    # Separate features and labels
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = ['Total Costs', 'Length of Stay']
    categorical_cols = X_train.columns.difference(numerical_cols)
    
    # Define preprocessing for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Apply transformations
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply feature selection
    feature_selector = SelectKBest(score_func=f_classif, k=999)
    X_train_selected = feature_selector.fit_transform(X_train_processed, y_train)
    X_test_selected = feature_selector.transform(X_test_processed)
    
    # One-hot encode labels
    y_train_processed = np.where(y_train == -1, 0, 1)
    
    # # Print number of features after preprocessing and selection
    # num_features = X_train_selected.shape[1]
    # print(f"Number of features after preprocessing and selection: {num_features}")
    
    return X_train_selected, y_train_processed, X_test_selected

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return np.float64(exp_logits / np.sum(exp_logits, axis=1, keepdims=True))


def loss_fn(X, y, w, freq):
    n, d = X.shape
    k = w.shape[1]

    logits = np.dot(X, w)  # Shape (n, k)

    g_wj_x = softmax(logits)  # Shape (n, k)

    loss = 0

    for j in range(k):
        selected_probs = g_wj_x[:, j][y[:, j] == 1]  # Shape (number of samples where y == j,)
        
        loss += -np.sum(np.log(selected_probs) / freq[j])
    
    loss = loss / (2 * n)
    
    return loss
    

def gradient(X, Y, W, freq) :
    #grad = 1/2n(X.T.(U-Y))
    U=softmax(np.dot(X,W))
    G = np.zeros_like(U, dtype=np.float64)
    n = np.float64(Y.shape[0])

    for i in range(Y.shape[0]) :
        index = np.where( Y[i] == 1 )[0][0]
        fact = np.float64(2)*n*freq[index]
        G[i] = (U[i] - Y[i]) / fact

    grad = np.transpose(X)@G

    return grad

def constant_lr(X, Y, W, lr, epochs, batch_size, freq) :
    n_batches = Y.shape[0] // batch_size

    for epoch in range(epochs) :
        loss = 0
        for i in range(n_batches) :
            X_batch = np.float64(X[i*batch_size : (i+1)*batch_size])
            Y_batch = np.float64(Y[i*batch_size : (i+1)*batch_size])

            W -=  np.float64(lr)*gradient(X_batch, Y_batch, W, freq)
            loss += loss_fn(X_batch, Y_batch, W, freq)

        # print(f"epoch : {epoch} loss : {np.mean(loss)}")
    return W

def adaptive_lr(X, Y, W, lr, k, epochs, batch_size, freq) :
    n_batches = Y.shape[0] / batch_size

    if(n_batches > (Y.shape[0] // batch_size)) :
        n_batches = 1 + (Y.shape[0] // batch_size)

    else :
        n_batches = Y.shape[0] // batch_size

    for epoch in range(epochs) :
        for i in range(n_batches) :
            X_batch = np.float64(X[i*batch_size : (i+1)*batch_size])
            Y_batch = np.float64(Y[i*batch_size : (i+1)*batch_size])

            W -=  np.float64(lr/(1 + k * (epoch+1)))*gradient(X_batch, Y_batch, W, freq)
            # print(f"strat : {2} epoch : {epoch+1} batch : {i+1} loss : {loss_fn(X_batch, Y_batch, W, freq)}")

    
    return W

def predict(X, W):
    scores = np.dot(X, W)
    predictions = np.argmax(scores, axis=1)
    predictions = np.where(predictions == 1, 1, -1)
    return predictions

def initialize_weights(n_features, n_classes, method='zeros', mean=0.0, std=0.01):
    if method == 'zeros':
        W = np.zeros((n_features, n_classes))
    elif method == 'random':
        W = np.random.uniform(-0.01, 0.01, (n_features, n_classes))
    elif method == 'normal':
        W = np.random.normal(mean, std, (n_features, n_classes))
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    return W

def hyperparameter_tuning(X_train, Y_train, X_test, y_freq):
    batch_sizes = [16]
    learning_rates = [5]
    strategies = [2]  # Only strategies 1 and 2
    epochs = 20

    best_loss = float('inf')
    best_W = None
    best_params = None

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for strategy in strategies:

                m = X_train.shape[1]
                k = Y_train.shape[1]

                # Initialize weights to zeros
                W = initialize_weights(m, k, method='zeros')

                if strategy == 1:
                    W = constant_lr(X_train, Y_train, W, lr, epochs, batch_size, y_freq)
                elif strategy == 2:
                    k = 10  
                    W = adaptive_lr(X_train, Y_train, W, lr, k, epochs, batch_size, y_freq)

                # Compute loss
                loss = loss_fn(X_train, Y_train, W, y_freq)

                if loss < best_loss:
                    best_loss = loss
                    best_W = W
                    best_params = (batch_size, lr, strategy)

    # print(best_params)
    predictions = predict(X_test, best_W)
    
    with open('output.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    return best_W, best_loss

def main():
    start_time = time.time()
    
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    
    X_train, y_train, X_test = preprocess_data(train_path, test_path)
    
    y_freq = np.bincount(y_train) / len(y_train)
    
    num_classes = len(np.unique(y_train))
    Y_train = np.eye(num_classes)[y_train]
    
    best_W, best_loss = hyperparameter_tuning(X_train, Y_train, X_test, y_freq)
    
    # print(f"Best training loss: {best_loss}")
    # print(f"Best weights shape: {best_W.shape}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
