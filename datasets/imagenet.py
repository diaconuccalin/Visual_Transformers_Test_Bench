"""
ImageNet 1k Dataset Loader

The ImageNet dataset is a large-scale image classification dataset with 1000 classes.
This loader expects the standard ImageNet directory structure:
    root/
        train/
            n01440764/
                *.JPEG
            n01443537/
                *.JPEG
            ...
        val/
            n01440764/
                *.JPEG
            n01443537/
                *.JPEG
            ...

This loader uses torchvision.datasets.ImageFolder for loading.
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path


def get_imagenet_dataset(split='val', batch_size=32, num_workers=4,
                         image_size=224, imagenet_root=None,
                         preprocessing='imagenet'):
    """
    Get ImageNet dataset with standard preprocessing.

    Args:
        split (str): 'train' or 'val'
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        image_size (int): Size to resize images to (default: 224)
        imagenet_root (str): Root directory containing ImageNet train/val folders
        preprocessing (str): Preprocessing type - 'imagenet' for standard normalization

    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    if imagenet_root is None:
        raise ValueError(
            "You must provide:\n"
            "  --imagenet-root: path to ImageNet root directory containing train/ and val/ folders"
        )

    root_path = Path(imagenet_root)
    split_dir = root_path / split

    if not split_dir.exists():
        raise ValueError(
            f"ImageNet split directory not found: {split_dir}\n"
            f"Expected structure: {imagenet_root}/train/ and {imagenet_root}/val/"
        )

    print(f"Loading ImageNet dataset ({split} split) from {split_dir}...")

    # Define transforms based on preprocessing type and split
    if preprocessing == 'tflite':
        # TFLite models typically expect input in [0, 1] range
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
    else:  # 'imagenet' preprocessing (standard)
        # ImageNet normalization stats
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    dataset = ImageFolder(root=str(split_dir), transform=transform)
    print(f"Loaded {len(dataset)} samples across {len(dataset.classes)} classes")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
