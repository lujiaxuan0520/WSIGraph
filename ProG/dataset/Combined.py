import os
import json
import random
import bisect
from torch.utils.data import Dataset
from torchvision import transforms

class CombinedDataset(Dataset):
    """
    A dataset that combines multiple datasets by concatenating their data_info.
    
    Args:
        datasets (list): List of dataset instances to combine.
    """
    
    def __init__(self, datasets):
        """
        Initialize the CombinedDataset.
        
        Args:
            datasets (list): List of dataset instances (e.g., TCGA_e2e, TCGA_frozen).
        """
        self.datasets = datasets
        self.cumulative_sizes = self.cumsum(self.datasets)
        
        # Concatenate data_info from all datasets
        self.data_info = []
        for dataset in datasets:
            self.data_info.extend(dataset.data_info)
        
    def cumsum(self, datasets):
        """
        Compute cumulative sizes for indexing.
        
        Args:
            datasets (list): List of dataset instances.
        
        Returns:
            list: Cumulative sizes.
        """
        cumulative = [0]
        for dataset in datasets:
            cumulative.append(cumulative[-1] + len(dataset))
        return cumulative

    def __len__(self):
        """
        Return the total number of samples across all datasets.
        
        Returns:
            int: Total length.
        """
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict or any: The sample data returned by the appropriate dataset.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        # Find the dataset that contains the idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        sample_idx = idx - self.cumulative_sizes[dataset_idx]
        
        return self.datasets[dataset_idx][sample_idx]
