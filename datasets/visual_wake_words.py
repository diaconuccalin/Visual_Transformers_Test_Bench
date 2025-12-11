"""
Visual Wake Words Dataset Loader

The Visual Wake Words dataset is a binary classification dataset derived from COCO.
Task: Detect whether an image contains a person or not.

This loader uses the original Visual Wake Words dataset via pyvww library.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class OriginalVWWDataset(Dataset):
    """PyTorch Dataset wrapper for Visual Wake Words dataset via pyvww."""

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

        print(f"Loading Visual Wake Words dataset ({split} split)...")

        # Validate inputs
        if root_dir is None or ann_file is None:
            raise ValueError(
                "You must provide:\n"
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
                                   image_size=224, vww_root=None, vww_ann=None):
    """
    Get Visual Wake Words dataset with standard preprocessing.

    Args:
        split (str): 'train', 'val', or 'test'
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        image_size (int): Size to resize images to
        vww_root (str): Root directory for VWW COCO images
        vww_ann (str): Path to VWW annotation file

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

    dataset = OriginalVWWDataset(
        split=split,
        transform=transform,
        root_dir=vww_root,
        ann_file=vww_ann
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
