import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import time
from sklearn.feature_selection import SelectKBest, f_classif
from train_algos import *

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
    
    return X_train_selected, y_train_processed, X_test_selected

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
    batch_size = 16
    lr = 5
    strategy = 2  
    epochs = 20

    best_loss = float('inf')
    best_W = None
    best_params = None


    m = X_train.shape[1]
    k = Y_train.shape[1]

    # Initialize weights to zeros
    W = initialize_weights(m, k, method='zeros')

    if strategy == 1:
        W = constant_lr(X_train, Y_train, W, lr, epochs, batch_size, y_freq)
    elif strategy == 2:
        ki = 10  
        W = adaptive_lr(X_train, Y_train, W, lr, ki, epochs, batch_size, y_freq)

    loss = loss_fn(X_train, Y_train, W, y_freq)

    predictions = predict(X_test, W)
    
    with open('output.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    return W

def main():
    start_time = time.time()
    
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    
    X_train, y_train, X_test = preprocess_data(train_path, test_path)
    
    y_freq = np.bincount(y_train) / len(y_train)
    
    num_classes = len(np.unique(y_train))
    Y_train = np.eye(num_classes)[y_train]

    W = hyperparameter_tuning(X_train, Y_train, X_test, y_freq)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

if __name__ == "__main__":
    main()
