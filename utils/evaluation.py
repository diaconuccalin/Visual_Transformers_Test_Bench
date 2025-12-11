"""
Evaluation utilities for benchmarking models.
"""

import torch
import time
from tqdm import tqdm


def evaluate_model_multiclass(model, dataloader, device='cuda', verbose=True, topk=(1, 5)):
    """
    Evaluate a model on a multi-class dataset with top-k accuracy metrics.

    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        device (str): Device to run evaluation on
        verbose (bool): Whether to show progress bar
        topk (tuple): Tuple of k values for top-k accuracy (default: (1, 5))

    Returns:
        dict: Dictionary containing evaluation metrics including top-k accuracies
    """
    model.eval()
    model = model.to(device)

    total_samples = 0
    total_time = 0
    topk_correct = {k: 0 for k in topk}
    maxk = max(topk)

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
            batch_size = labels.size(0)
            total_samples += batch_size

            # Get top-k predictions
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()  # Transpose to shape (maxk, batch_size)
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            # Calculate top-k correct counts
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                topk_correct[k] += correct_k.item()

    # Calculate metrics
    topk_accuracy = {k: topk_correct[k] / total_samples for k in topk}
    avg_inference_time = total_time / total_samples
    throughput = total_samples / total_time if total_time > 0 else 0

    results = {
        'total_samples': total_samples,
        'topk_accuracy': topk_accuracy,
        'top1_accuracy': topk_accuracy.get(1, 0),
        'top5_accuracy': topk_accuracy.get(5, 0),
        'avg_inference_time_per_sample': avg_inference_time,
        'throughput_samples_per_sec': throughput,
        'total_time': total_time
    }

    return results


def print_results_multiclass(results):
    """
    Pretty print evaluation results for multi-class classification.

    Args:
        results (dict): Results dictionary from evaluate_model_multiclass
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Multi-class)")
    print("="*60)
    print(f"Total Samples:        {results['total_samples']}")
    print("-"*60)

    # Print top-k accuracies
    topk_accuracy = results.get('topk_accuracy', {})
    for k in sorted(topk_accuracy.keys()):
        acc = topk_accuracy[k]
        print(f"Top-{k} Accuracy:       {acc:.4f} ({acc*100:.2f}%)")

    print("-"*60)
    print(f"Avg Inference Time:   {results['avg_inference_time_per_sample']*1000:.2f} ms/sample")
    print(f"Throughput:           {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Total Time:           {results['total_time']:.2f} seconds")
    print("="*60 + "\n")


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
    throughput = total_samples / total_time if total_time > 0 else 0

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
