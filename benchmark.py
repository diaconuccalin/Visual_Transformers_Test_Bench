#!/usr/bin/env python3
"""
Visual Transformers Test Bench - Main Benchmark Script

This script evaluates visual transformer and hybrid models on various datasets,
with a focus on models targeting low-power devices.

Usage:
    python benchmark.py --model <model_path_or_name> --dataset <dataset_name> [options]

Example:
    python benchmark.py --model mobilenet_v2 --dataset visual_wake_words
    python benchmark.py --model ./models/my_model.pth --dataset visual_wake_words --batch-size 64
"""

import argparse
import torch
import sys
from pathlib import Path

from datasets import get_visual_wake_words_dataset
from utils import load_model, evaluate_model
from utils.evaluation import print_results


SUPPORTED_DATASETS = ['visual_wake_words']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark visual models on various datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate MobileNetV2 on Visual Wake Words
  python benchmark.py --model mobilenet_v2 --dataset visual_wake_words

  # Evaluate custom model with specific batch size
  python benchmark.py --model ./models/my_model.pth --dataset visual_wake_words --batch-size 64

  # Use CPU instead of GPU
  python benchmark.py --model mobilenet_v2 --dataset visual_wake_words --device cpu
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint or name of pretrained model'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help=f'Dataset to evaluate on. Choices: {SUPPORTED_DATASETS}'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to use (default: test)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )

    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to run evaluation on (default: cuda if available, else cpu)'
    )

    parser.add_argument(
        '--num-classes',
        type=int,
        default=2,
        help='Number of output classes for the model (default: 2 for binary classification)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress bars'
    )

    parser.add_argument(
        '--vww-root',
        type=str,
        default=None,
        help='Root directory for VWW COCO images (required)'
    )

    parser.add_argument(
        '--vww-ann',
        type=str,
        default=None,
        help='Path to VWW annotation JSON file (required)'
    )

    return parser.parse_args()


def get_dataset(dataset_name, split, batch_size, num_workers, image_size,
                vww_root=None, vww_ann=None):
    """
    Load the specified dataset.

    Args:
        dataset_name (str): Name of the dataset
        split (str): Dataset split
        batch_size (int): Batch size
        num_workers (int): Number of workers
        image_size (int): Image size
        vww_root (str): VWW COCO images root
        vww_ann (str): VWW annotation file

    Returns:
        DataLoader: Dataset loader
    """
    if dataset_name == 'visual_wake_words':
        return get_visual_wake_words_dataset(
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            vww_root=vww_root,
            vww_ann=vww_ann
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def main():
    """Main benchmark function."""
    args = parse_args()

    print("="*60)
    print("VISUAL TRANSFORMERS TEST BENCH")
    print("="*60)
    print(f"Model:        {args.model}")
    print(f"Dataset:      {args.dataset}")
    print(f"Split:        {args.split}")
    print(f"Device:       {args.device}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Image Size:   {args.image_size}")
    print("="*60 + "\n")

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    try:
        # Load model
        print("Loading model...")
        model = load_model(
            model_path=args.model,
            num_classes=args.num_classes,
            device=args.device
        )
        print("Model loaded successfully!\n")

        # Load dataset
        print(f"Loading {args.dataset} dataset ({args.split} split)...")
        dataloader = get_dataset(
            dataset_name=args.dataset,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            vww_root=args.vww_root,
            vww_ann=args.vww_ann
        )
        print("Dataset loaded successfully!\n")

        # Run evaluation
        print("Starting evaluation...")
        results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=args.device,
            verbose=not args.quiet
        )

        # Print results
        print_results(results)

        return 0

    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
