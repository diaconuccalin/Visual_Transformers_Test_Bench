"""
Visual Wake Words Dataset Loader

The Visual Wake Words dataset is a binary classification dataset derived from COCO.
Task: Detect whether an image contains a person or not.

This loader supports:
1. Wake Vision dataset (successor to VWW, 100x larger, readily available in tfds)
2. Original Visual Wake Words via pyvww library (requires manual COCO setup)
"""

import os
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class WakeVisionDataset(Dataset):
    """PyTorch Dataset wrapper for Wake Vision dataset (VWW successor)."""

    def __init__(self, split='test', transform=None, data_dir=None, use_quality_split=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            transform: PyTorch transforms to apply to images
            data_dir (str): Directory to store dataset
            use_quality_split (bool): If True and split='train', use train_quality instead of train_large
        """
        self.split = split
        self.transform = transform

        # Map split names to wake_vision naming
        if split == 'train':
            tf_split = 'train_quality' if use_quality_split else 'train_large'
        elif split == 'val':
            tf_split = 'validation'
        else:
            tf_split = 'test'

        # Load dataset using tensorflow_datasets
        print(f"Loading Wake Vision dataset ({tf_split} split)...")
        print("Note: Wake Vision is the successor to Visual Wake Words (100x larger, 6M+ images)")
        self.dataset = tfds.load(
            'wake_vision',
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

        # Extract image and label (Wake Vision uses 'person' key for label)
        image = sample['image'].numpy()
        label = int(sample['person'].numpy())

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


class OriginalVWWDataset(Dataset):
    """PyTorch Dataset wrapper for original Visual Wake Words dataset via pyvww."""

    def __init__(self, split='test', transform=None, root_dir=None, ann_file=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            transform: PyTorch transforms to apply to images
            root_dir (str): Root directory containing COCO images
            ann_file (str): Path to Visual Wake Words annotation file
        """
        try:
            import pyvww
        except ImportError:
            raise ImportError(
                "pyvww library not found. Install it with: pip install pyvww\n"
                "Also ensure you have downloaded COCO dataset and generated VWW annotations.\n"
                "See: https://github.com/Mxbonn/visualwakewords"
            )

        self.split = split
        self.transform = transform

        print(f"Loading original Visual Wake Words dataset ({split} split)...")

        # Validate inputs
        if root_dir is None or ann_file is None:
            raise ValueError(
                "For original VWW dataset, you must provide:\n"
                "  --vww-root: path to COCO images directory\n"
                "  --vww-ann: path to VWW annotations JSON file"
            )

        # Load dataset using pyvww
        self.dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root=root_dir,
            annFile=ann_file
        )

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            image: PIL Image or transformed tensor
            label: 0 (no person) or 1 (person present)
        """
        image, label = self.dataset[idx]

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


def get_visual_wake_words_dataset(split='test', batch_size=32, num_workers=4,
                                   image_size=224, data_dir=None, use_original=False,
                                   vww_root=None, vww_ann=None, use_quality_split=False):
    """
    Get Visual Wake Words dataset with standard preprocessing.

    By default, loads Wake Vision (the successor to VWW, 100x larger).
    To use the original VWW dataset, set use_original=True and provide vww_root and vww_ann.

    Args:
        split (str): 'train', 'val', or 'test'
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        image_size (int): Size to resize images to
        data_dir (str): Directory to store dataset (for Wake Vision)
        use_original (bool): If True, use original VWW via pyvww (requires manual setup)
        vww_root (str): Root directory for original VWW COCO images (only if use_original=True)
        vww_ann (str): Path to VWW annotation file (only if use_original=True)
        use_quality_split (bool): For Wake Vision train split, use train_quality instead of train_large

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

    if use_original:
        # Use original Visual Wake Words via pyvww
        dataset = OriginalVWWDataset(
            split=split,
            transform=transform,
            root_dir=vww_root,
            ann_file=vww_ann
        )
    else:
        # Use Wake Vision (default, readily available in tfds)
        dataset = WakeVisionDataset(
            split=split,
            transform=transform,
            data_dir=data_dir,
            use_quality_split=use_quality_split
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
