import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ImageDataset(Dataset):
    """
    Custom dataset for loading a 4D NumPy array as image tensors.
    Assumes input shape (N, C, H, W), normalized to [-1, 1].
    """
    def __init__(self, data_array):
        if isinstance(data_array, np.ndarray):
            self.data = torch.from_numpy(data_array).float()
        elif isinstance(data_array, torch.Tensor):
            self.data = data_array.float()
        else:
            raise ValueError("Input data must be a NumPy array or a PyTorch tensor.")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# rescale the data to [0, 1]
def rescale_data(data):
    
    data_min = data.min()
    data = data - data_min
    data_max = data.max()
    if data_max > 0:
        data /= data_max
    else:
        raise ValueError("Data max is zero, cannot rescale.")
    
    return data, data_min, data_max


def create_dataloaders(data_np, batch_size=64, train_ratio=0.8, shuffle=True):
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        data_np (np.ndarray): Normalized data of shape (N, C, H, W).
        batch_size (int): Batch size.
        train_ratio (float): Fraction of data used for training.
        shuffle (bool): Whether to shuffle data.

    Returns:
        tuple: (train_loader, test_loader)
    """
    dataset = ImageDataset(data_np)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



class ImageDatasetLabel(Dataset):
    """
    Custom dataset for loading a 4D NumPy array as image tensors.
    Assumes input shape (N, C, H, W) for both data and labels.
    """
    def __init__(self, data_array, label_array):
        if isinstance(data_array, np.ndarray):
            self.data = torch.from_numpy(data_array).float()
        elif isinstance(data_array, torch.Tensor):
            self.data = data_array.float()
        else:
            raise ValueError("Input data must be a NumPy array or a PyTorch tensor.")

        if isinstance(label_array, np.ndarray):
            self.labels = torch.from_numpy(label_array).float()
        elif isinstance(label_array, torch.Tensor):
            self.labels = label_array.float()
        else:
            raise ValueError("Input label must be a NumPy array or a PyTorch tensor.")

        if self.data.shape != self.labels.shape:
            raise ValueError("Data and labels must have the same shape.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def create_dataloaders_label(data_np, label_np, batch_size=64, train_ratio=0.8, shuffle=True):
    """
    Create PyTorch DataLoaders for training and testing with image-label pairs.

    Args:
        data_np (np.ndarray): Normalized input data of shape (N, C, H, W).
        label_np (np.ndarray): Labels with the same shape as data_np.
        batch_size (int): Batch size.
        train_ratio (float): Fraction of data used for training.
        shuffle (bool): Whether to shuffle data.

    Returns:
        tuple: (train_loader, test_loader)
    """
    dataset = ImageDatasetLabel(data_np, label_np)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
