"""
Evaluation utilities for benchmarking models.
"""

import torch
import time
from tqdm import tqdm
import numpy as np


def evaluate_model(model, dataloader, device='cuda', verbose=True):
    """
    Evaluate a model on a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        device (str): Device to run evaluation on
        verbose (bool): Whether to show progress bar

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    model = model.to(device)

    total_correct = 0
    total_samples = 0
    total_time = 0

    # For per-class metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)

            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            total_time += (end_time - start_time)

            # Get predictions
            if outputs.size(1) == 1:
                # Single output (binary classification with sigmoid)
                predictions = (torch.sigmoid(outputs) > 0.5).long().squeeze()
            else:
                # Multiple outputs (use softmax and argmax)
                predictions = torch.argmax(outputs, dim=1)

            # Calculate metrics
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            # Calculate confusion matrix components
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            true_negatives += ((predictions == 0) & (labels == 0)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = total_correct / total_samples
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    avg_inference_time = total_time / total_samples
    throughput = total_samples / total_time

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_samples': total_samples,
        'correct_predictions': total_correct,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_inference_time_per_sample': avg_inference_time,
        'throughput_samples_per_sec': throughput,
        'total_time': total_time
    }

    return results


def print_results(results):
    """
    Pretty print evaluation results.

    Args:
        results (dict): Results dictionary from evaluate_model
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Samples:        {results['total_samples']}")
    print(f"Correct Predictions:  {results['correct_predictions']}")
    print("-"*60)
    print(f"Accuracy:             {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision:            {results['precision']:.4f}")
    print(f"Recall:               {results['recall']:.4f}")
    print(f"F1 Score:             {results['f1_score']:.4f}")
    print("-"*60)
    print("Confusion Matrix:")
    print(f"  True Positives:     {results['true_positives']}")
    print(f"  True Negatives:     {results['true_negatives']}")
    print(f"  False Positives:    {results['false_positives']}")
    print(f"  False Negatives:    {results['false_negatives']}")
    print("-"*60)
    print(f"Avg Inference Time:   {results['avg_inference_time_per_sample']*1000:.2f} ms/sample")
    print(f"Throughput:           {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Total Time:           {results['total_time']:.2f} seconds")
    print("="*60 + "\n")
