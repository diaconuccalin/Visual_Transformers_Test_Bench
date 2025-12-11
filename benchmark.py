#!/usr/bin/env python3
"""
Visual Transformers Test Bench - Main Benchmark Script

This script evaluates visual transformer and hybrid models on various datasets,
with a focus on models targeting low-power devices.

Usage:
    python benchmark.py --model <model_path_or_name> --dataset <dataset_name> [options]

Example:
    python benchmark.py --model mobilenet_v1 --dataset visual_wake_words --vww-root ./data/coco2014/all --vww-ann ./data/coco2014/annotations/vww/instances_val.json
    python benchmark.py --model ./models/my_model.pth --dataset visual_wake_words --vww-root ./data/coco2014/all --vww-ann ./data/coco2014/annotations/vww/instances_val.json
"""

import argparse
import torch
import sys
import subprocess
from pathlib import Path

from datasets import get_visual_wake_words_dataset
from utils import load_model, evaluate_model
from utils.evaluation import print_results


SUPPORTED_DATASETS = ['visual_wake_words']

# Mapping of model names to their expected files
VWW_TRAINED_MODELS = {
    'mobilenet_v1_vww': {
        'file': 'mobilenet/v1/vww_96_float.tflite',
        'image_size': 96,
        'description': 'MobileNetV1 (alpha=0.25) trained on VWW'
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark visual models on various datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate trained MobileNetV1 on Visual Wake Words (auto-downloads if needed)
  python benchmark.py --model mobilenet_v1 --dataset visual_wake_words \\
    --vww-root ./data/coco2014/all --vww-ann ./data/coco2014/annotations/vww/instances_val.json

  # Evaluate custom model with specific batch size
  python benchmark.py --model ./models/my_model.pth --dataset visual_wake_words --batch-size 64 \\
    --vww-root ./data/coco2014/all --vww-ann ./data/coco2014/annotations/vww/instances_val.json

  # Use CPU instead of GPU
  python benchmark.py --model mobilenet_v1 --dataset visual_wake_words --device cpu \\
    --vww-root ./data/coco2014/all --vww-ann ./data/coco2014/annotations/vww/instances_val.json
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
        default="./data/coco2014/all",
        help='Root directory for VWW COCO images (required)'
    )

    parser.add_argument(
        '--vww-ann',
        type=str,
        default="./data/coco2014/annotations/vww/instances_train.json",
        help='Path to VWW annotation JSON file (required)'
    )

    return parser.parse_args()


def download_vww_model_if_needed(model_name, models_dir='./models'):
    """
    Download VWW trained model if it doesn't exist.

    Args:
        model_name (str): Name of the model (e.g., 'mobilenet_v1_vww')
        models_dir (str): Directory to store models

    Returns:
        str: Path to the model file
    """
    if model_name not in VWW_TRAINED_MODELS:
        return None

    model_info = VWW_TRAINED_MODELS[model_name]
    model_path = Path(models_dir) / model_info['file']

    if not model_path.exists():
        print(f"\n{'='*60}")
        print(f"Model '{model_name}' not found locally.")
        print(f"Description: {model_info['description']}")
        print(f"{'='*60}")
        print(f"Downloading trained model to {model_path}...")
        print()

        # Create models directory
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Run download script
        try:
            result = subprocess.run(
                ['python', 'download_vww_model.py', '--output-dir', str(models_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            print(f"Model downloaded successfully to {model_path}")
            print()
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model: {e}")
            print(e.stderr)
            sys.exit(1)
    else:
        print(f"Using cached model: {model_path}")

    return str(model_path)


def get_dataset(dataset_name, split, batch_size, num_workers, image_size,
                vww_root=None, vww_ann=None, preprocessing='imagenet'):
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
        preprocessing (str): Preprocessing type ('imagenet' or 'tflite')

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
            vww_ann=vww_ann,
            preprocessing=preprocessing
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def main():
    """Main benchmark function."""
    args = parse_args()

    # Auto-expand model name for VWW dataset
    # If user specifies "mobilenet_v1" and dataset is VWW, use "mobilenet_v1_vww"
    if args.dataset == 'visual_wake_words' and args.model == 'mobilenet_v1':
        args.model = 'mobilenet_v1_vww'
        args.device = "cpu"
        print("Note: Using trained VWW model (mobilenet_v1_vww) for Visual Wake Words dataset")
        print()

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
        # Check if this is a known VWW model that needs downloading
        model_path = args.model
        if args.model in VWW_TRAINED_MODELS:
            model_path = download_vww_model_if_needed(args.model)
            # Auto-set image size if not specified
            if args.image_size == 224:  # Default value
                recommended_size = VWW_TRAINED_MODELS[args.model]['image_size']
                print(f"Note: Auto-setting image size to {recommended_size}x{recommended_size} for {args.model}")
                args.image_size = recommended_size
                print()

        # Load model
        print("Loading model...")
        model, model_metadata = load_model(
            model_path=model_path,
            num_classes=args.num_classes,
            device=args.device
        )
        print("Model loaded successfully!\n")

        # Get preprocessing type from model metadata
        preprocessing = model_metadata.get('preprocessing', 'imagenet')
        if preprocessing == 'tflite':
            print("Note: Using TFLite preprocessing (input range [0, 1])")
        else:
            print("Note: Using ImageNet preprocessing (normalized)")
        print()

        # Load dataset
        print(f"Loading {args.dataset} dataset ({args.split} split)...")
        dataloader = get_dataset(
            dataset_name=args.dataset,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            vww_root=args.vww_root,
            vww_ann=args.vww_ann,
            preprocessing=preprocessing
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
