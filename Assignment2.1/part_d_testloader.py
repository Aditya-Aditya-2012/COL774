import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

#Remember to import "numpy_transforms" functions if you wish to import these two classes in a different script.

np.random.seed(0)

class TestImageDataset:
    def __init__(self, root_dir, csv, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("L") #Convert image to greyscale

        if self.transform:
            image = self.transform(image)

        return np.array(image)

# Transformations using NumPy
def resize(image, size):
    # return np.array(Image.fromarray(image).resize(size))
    return np.array(image.resize(size))

def to_tensor(image):
    return image.astype(np.float32) / 255.0

def numpy_transform(image, size=(25, 25)):
    image = resize(image, size)
    image = to_tensor(image)
    image = image.flatten()
    return image

class TestDataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        # if self.shuffle:
        #     np.random.shuffle(self.indices)

    def __iter__(self):
        self.start_idx = 0
        return self
    def __len__(self):
        return int(len(self.dataset)/self.batch_size)

    def __next__(self):
        if self.start_idx >= len(self.dataset):
            raise StopIteration

        end_idx = min(self.start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.start_idx:end_idx]
        images = []
        labels = []

        for idx in batch_indices:
            image = self.dataset[idx]
            images.append(image)

        self.start_idx = end_idx

        # Stack images and labels to create batch tensors
        batch_images = np.stack(images, axis=0)

        return batch_images

if __name__ == '__main__':
    #Example usage of the script 

    # Root directory containing the 8 subfolders
    root_dir = "/path/to/dataset" #Path to the dataset directory
    csv = os.path.join(root_dir, "val.csv") #The csv file will always be placed inside the dataset directory.
    #Please ensure that you set the csv file path accordingly in all parts of the assignment so that it gets loaded correctly.
    #While evaluation, the train.csv will have same name while the test set csv will be renamed to "val.csv" to be compatible with the setting here.

    # Create the custom dataset
    dataset = TestImageDataset(root_dir=root_dir, csv = csv, transform=numpy_transform)  #Remember to import "numpy_transforms" functions.

    # Create the DataLoader
    dataloader = TestDataLoader(dataset, batch_size=1)

    # Iterate through the DataLoader [No labels will be returned in the test loader]
    for idx, images in enumerate(dataloader):
        print(images.shape)  # Should be [batch_size, 625]
        #Data being loaded!
