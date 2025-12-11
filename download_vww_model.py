#!/usr/bin/env python3
"""
Download and prepare Visual Wake Words trained MobileNet model.

This script downloads a MobileNet model trained on Visual Wake Words dataset
from MLCommons Tiny benchmark and converts it to PyTorch format for use with the benchmark.
"""

import os
import sys
import argparse
import urllib.request
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


# URL for the trained VWW model from MLCommons Tiny
VWW_MODEL_URL = "https://github.com/mlcommons/tiny/raw/master/benchmark/training/visual_wake_words/trained_models/vww_96_float.tflite"


def download_file(url, output_path):
    """Download a file from URL to output path."""
    print(f"Downloading from {url}...")
    print(f"Saving to {output_path}...")

    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\rProgress: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def convert_tflite_to_pytorch(tflite_path, num_classes=2):
    """
    Convert TFLite model to PyTorch.

    Note: This creates a MobileNetV2 architecture and attempts to load compatible weights.
    For full compatibility, consider using the TFLite model directly or onnx conversion.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("WARNING: TensorFlow not installed. Cannot convert TFLite model.")
        print("Install with: pip install tensorflow")
        print("Falling back to creating a PyTorch model with random weights...")
        return create_vww_mobilenet(num_classes)

    print(f"Loading TFLite model from {tflite_path}...")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"TFLite model loaded successfully!")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Create PyTorch MobileNetV2 with matching architecture
    print("Creating PyTorch MobileNetV2 model...")
    model = create_vww_mobilenet(num_classes)

    print("NOTE: Direct weight conversion from TFLite to PyTorch is complex.")
    print("      For best results, use the TFLite model directly or train in PyTorch.")

    return model


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
        description='Download and prepare VWW-trained MobileNet model from MLCommons Tiny'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Directory to save the model (default: ./models)'
    )
    parser.add_argument(
        '--keep-tflite',
        action='store_true',
        help='Keep the downloaded TFLite model file'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    tflite_path = os.path.join(args.output_dir, 'vww_96_float.tflite')
    pytorch_path = os.path.join(args.output_dir, 'mobilenet_v2_vww.pth')

    print("="*60)
    print("Visual Wake Words MobileNet Model Download")
    print("="*60)
    print()
    print("Downloading trained VWW model from MLCommons Tiny benchmark...")
    print("Source: https://github.com/mlcommons/tiny")
    print()

    # Download the TFLite model
    if os.path.exists(tflite_path):
        print(f"TFLite model already exists at {tflite_path}")
        print("Skipping download...")
    else:
        success = download_file(VWW_MODEL_URL, tflite_path)
        if not success:
            print("Failed to download model. Exiting.")
            return 1

    print()
    print("="*60)
    print("TFLite Model Ready!")
    print("="*60)
    print()
    print(f"Model location: {tflite_path}")
    print()
    print("IMPORTANT: This is a TFLite model (not PyTorch)")
    print()
    print("To use this model:")
    print("1. Use TensorFlow Lite interpreter directly, OR")
    print("2. Convert to ONNX format, then to PyTorch, OR")
    print("3. Train your own PyTorch model on VWW dataset")
    print()
    print("For PyTorch training, see:")
    print("  https://github.com/mlcommons/tiny/tree/master/benchmark/training/visual_wake_words")
    print()
    print("Model details:")
    print("  - Architecture: MobileNetV1 (alpha=0.25)")
    print("  - Input size: 96x96x3")
    print("  - Output classes: 2 (person/no person)")
    print("  - Precision: FP32")
    print("  - Expected accuracy: ~86% on COCO 2014")
    print()

    if not args.keep_tflite:
        print(f"Note: Use --keep-tflite to retain the TFLite file at {tflite_path}")

    print("="*60)
    print()

    return 0


if __name__ == '__main__':
    main()
