"""
Visual Wake Words Dataset Loader

The Visual Wake Words dataset is a binary classification dataset derived from COCO.
Task: Detect whether an image contains a person or not.
"""

import os
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class VisualWakeWordsDataset(Dataset):
    """PyTorch Dataset wrapper for Visual Wake Words dataset."""

    def __init__(self, split='test', transform=None, data_dir=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            transform: PyTorch transforms to apply to images
            data_dir (str): Directory to store dataset
        """
        self.split = split
        self.transform = transform

        # Map split names to tensorflow_datasets naming
        split_map = {
            'train': 'train',
            'val': 'validation',
            'test': 'test'
        }

        tf_split = split_map.get(split, 'test')

        # Load dataset using tensorflow_datasets
        print(f"Loading Visual Wake Words dataset ({tf_split} split)...")
        self.dataset = tfds.load(
            'visual_wake_words',
            split=tf_split,
            data_dir=data_dir,
            shuffle_files=False
        )

        # Convert to list for indexing
        self.data_list = list(self.dataset)
        print(f"Loaded {len(self.data_list)} samples")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Returns:
            image: PIL Image or transformed tensor
            label: 0 (no person) or 1 (person present)
        """
        sample = self.data_list[idx]

        # Extract image and label
        image = sample['image'].numpy()
        label = int(sample['label'].numpy())

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


def get_visual_wake_words_dataset(split='test', batch_size=32, num_workers=4,
                                   image_size=224, data_dir=None):
    """
    Get Visual Wake Words dataset with standard preprocessing.

    Args:
        split (str): 'train', 'val', or 'test'
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        image_size (int): Size to resize images to
        data_dir (str): Directory to store dataset

    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    # Define standard transforms for evaluation
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    dataset = VisualWakeWordsDataset(
        split=split,
        transform=transform,
        data_dir=data_dir
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
