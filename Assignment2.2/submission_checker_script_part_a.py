import subprocess
import os
import pandas as pd
import pickle
import numpy as np
import csv


def run_training(train_dataset_root, save_weights_path):
    try:
        train_command = [
            "python", "part_a_train.py",
            "--train_dataset_root", train_dataset_root,
            "--save_weights_path", save_weights_path
        ]
        subprocess.run(train_command, check=True)
        print("Training completed successfully.")
        

        # weights_file = os.path.join(save_weights_path, "part_a_binary_model.pth")
        if not os.path.isdir(save_weights_path):
            if os.path.exists(save_weights_path):
                if save_weights_path.split(".")[-1] == "pth":
                    print(f"Weights saved successfully in {save_weights_path}.")
                else:

                    print("Saved weights are not in .pth format")
                    exit()
            else:
                print(f"Weights not found in {save_weights_path}.")
                exit()
        else:
            print("The given path is a directory. Please append the name of the weights file.")
            exit()
            
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        exit()

def run_testing(test_dataset_root, save_weights_path, save_predictions_path):
    try:
        test_command = [
            "python", "part_a_test.py",
            "--test_dataset_root", test_dataset_root,
            "--load_weights_path", save_weights_path,
            "--save_predictions_path", save_predictions_path
        ]
        subprocess.run(test_command, check=True)

        print("Testing completed successfully!")
        if not os.path.isdir(save_predictions_path):
            if os.path.exists(save_predictions_path):
                if save_predictions_path.split(".")[-1] == "pkl":
                    print(f"Predictions saved successfully in {save_predictions_path}.")
                else:

                    print("Predictions are saved not in .pkl format")
                    exit()
            else:
                print(f"Predictions not found in path {save_predictions_path}.")
                exit()
        else:
            print("The given path is a directory. Please append the name of the Predictions file.")
            exit()
       
        
    except subprocess.CalledProcessError as e:
        print(f"Error during testing: {e}")
        exit()

def compute_accuracy(gt_csv_path, predictions_file):
    # predictions_file = os.path.join(save_predictions_path, "predictions.pkl")
    
    if not os.path.exists(predictions_file):
        print(f"Predictions file {predictions_file} not found.")
        exit()

    #Check assignment pdf for the format 
    try:
        with open(predictions_file, 'rb') as f:
            predictions = np.array(pickle.load(f))
        
        if not isinstance(predictions, np.ndarray) or predictions.ndim != 1:
            print(f"Predictions file {predictions_file} is not in the correct format.")
            exit()
    except Exception as e:
        print(f"Error loading predictions: {e}")
        exit()
    
    try:
        # gt_data = pd.read_csv(gt_csv_path)
        # gt_labels = gt_data.iloc[:, 1].values
        ground_truth = []
        with open(gt_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ground_truth.append(int(row['class'])) 
        gt_labels = np.array(ground_truth)
        
        if len(gt_labels) != len(predictions):
            print("Mismatch between number of ground truth labels and predictions.")
            exit()
        
        accuracy = np.mean(predictions == gt_labels) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        print("Please note: This is accuracy on the Public data set. The evaluation will be done a held-out private dataset")
        
    except Exception as e:
        print(f"Error loading ground truth CSV file: {e}")

if __name__ == "__main__":

    train_dataset_root = input("Enter train dataset root: ")
    save_weights_path = input("Enter path to save weights (full path with weights file name): ")
    test_dataset_root = input("Enter test dataset root: ")
    save_predictions_path = input("Enter path to save predictions (full path with prediction file name): ")
    gt_csv_path = input("Enter path to the ground truth CSV file: ")

    run_training(train_dataset_root, save_weights_path)

    run_testing(test_dataset_root, save_weights_path, save_predictions_path)

    compute_accuracy(gt_csv_path, save_predictions_path)
