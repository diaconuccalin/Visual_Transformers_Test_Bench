#!/usr/bin/env python3
"""
Download and prepare Visual Wake Words trained MobileNet model.

This script downloads a MobileNet model trained on Visual Wake Words dataset
and converts it to PyTorch format for use with the benchmark.
"""

import os
import sys
import argparse
import urllib.request
import torch
import torch.nn as nn
import torchvision.models as models


def download_file(url, output_path):
    """Download a file from URL to output path."""
    print(f"Downloading from {url}...")
    print(f"Saving to {output_path}...")

    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def create_vww_mobilenet(num_classes=2):
    """
    Create a MobileNetV2 model adapted for Visual Wake Words (2 classes).

    Returns a model with a binary classification head.
    """
    print(f"Creating MobileNetV2 with {num_classes} classes for VWW...")

    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)

    # Replace the classifier for binary classification
    # MobileNetV2 has classifier with input features of 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes)
    )

    print("Model created successfully!")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare VWW-trained MobileNet model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Directory to save the model (default: ./models)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='mobilenet_v2_vww.pth',
        help='Output model filename (default: mobilenet_v2_vww.pth)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.model_name)

    print("="*60)
    print("Visual Wake Words MobileNet Model Setup")
    print("="*60)
    print()

    # For now, create a model with VWW-adapted head
    # In the future, we can download actual pre-trained VWW weights
    print("Creating MobileNetV2 adapted for Visual Wake Words...")
    print("Note: Starting with ImageNet weights and VWW-adapted head")
    print("      Fine-tuning on VWW dataset recommended for best results")
    print()

    model = create_vww_mobilenet(num_classes=2)

    # Save the model
    print(f"Saving model to {output_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': 2,
        'architecture': 'mobilenet_v2',
        'dataset': 'visual_wake_words',
        'note': 'ImageNet pretrained with VWW-adapted head (2 classes)'
    }, output_path)

    print("Model saved successfully!")
    print()
    print("="*60)
    print("Setup Complete!")
    print("="*60)
    print()
    print(f"Model location: {output_path}")
    print()
    print("To use with benchmark:")
    print(f"python benchmark.py --model {output_path} --dataset visual_wake_words \\")
    print("  --vww-root ./data/coco2014/all \\")
    print("  --vww-ann ./data/vww/instances_val.json")
    print()
    print("Note: This model uses ImageNet pretrained weights with a VWW-adapted head.")
    print("For best results, fine-tune on the VWW dataset first.")
    print()


if __name__ == '__main__':
    main()
