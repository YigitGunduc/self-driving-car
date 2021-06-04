import numpy as np
import os
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import pandas as pd
from skimage import io
import torch
from torchvision import datasets
from torch.utils.data import (
    Dataset,
    DataLoader,
)  

batch_size = 32
class SelfDrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def get_labels(self):
        return self.annotations.iloc[:, 1]
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
