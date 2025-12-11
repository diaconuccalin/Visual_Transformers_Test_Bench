"""
CIFAR-10 Dataset Loader

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
with 6000 images per class. There are 50000 training images and 10000 test images.

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

This loader uses torchvision's built-in CIFAR-10 dataset.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar10_dataset(split='test', batch_size=32, num_workers=4,
                        image_size=32, data_root='./data', 
                        preprocessing='cifar10'):
    """
    Get CIFAR-10 dataset with standard preprocessing.

    Args:
        split (str): 'train', 'val', or 'test'
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        image_size (int): Size to resize images to (default: 32 for CIFAR-10)
        data_root (str): Root directory to download/store CIFAR-10
        preprocessing (str): Preprocessing type - 'cifar10' for CIFAR-10 normalization,
                           'imagenet' for ImageNet normalization (for transfer learning)

    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    # CIFAR-10 specific normalization (computed from training set)
    # These are the standard values used in CIFAR-10 literature
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2470, 0.2435, 0.2616]
    
    # ImageNet normalization (for models pretrained on ImageNet)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Select normalization based on preprocessing type
    if preprocessing == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
    else:  # 'imagenet' or other
        mean = imagenet_mean
        std = imagenet_std

    # Define transforms based on split
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    print(f"Loading CIFAR-10 dataset ({split} split)...")
    print(f"Using {preprocessing} preprocessing (mean={mean}, std={std})")

    # Load dataset
    if split == 'train':
        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform
        )
    elif split == 'val':
        # Create validation split from training set
        # Use 10% of training data for validation (5000 samples)
        full_train = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform
        )
        train_size = int(0.9 * len(full_train))
        val_size = len(full_train) - train_size
        _, dataset = random_split(
            full_train, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:  # 'test'
        dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform
        )

    print(f"Loaded {len(dataset)} samples")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
