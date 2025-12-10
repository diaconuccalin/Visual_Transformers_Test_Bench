"""
Model loading utilities for various model formats.
"""

import torch
import torch.nn as nn
from pathlib import Path


def load_model(model_path, num_classes=2, device='cuda'):
    """
    Load a model from file or use a pretrained model.

    Args:
        model_path (str): Path to model checkpoint or model name
        num_classes (int): Number of output classes
        device (str): Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded PyTorch model
    """
    model_path = Path(model_path)

    # Check if it's a file path
    if model_path.exists():
        print(f"Loading model from {model_path}")

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # Need to initialize model architecture first
                # This is a simplified approach - may need customization
                raise NotImplementedError(
                    "Loading from state_dict requires model architecture. "
                    "Please provide the full model or modify this function."
                )
            else:
                model = checkpoint
        else:
            model = checkpoint

    else:
        # Try loading as a pretrained model name from torchvision or timm
        model_name = str(model_path)
        print(f"Loading pretrained model: {model_name}")

        try:
            # Try torchvision models first
            import torchvision.models as models

            if hasattr(models, model_name):
                model = getattr(models, model_name)(pretrained=True)

                # Modify the final layer for binary classification if needed
                if num_classes != 1000:
                    if hasattr(model, 'fc'):
                        in_features = model.fc.in_features
                        model.fc = nn.Linear(in_features, num_classes)
                    elif hasattr(model, 'classifier'):
                        if isinstance(model.classifier, nn.Linear):
                            in_features = model.classifier.in_features
                            model.classifier = nn.Linear(in_features, num_classes)
                        else:
                            in_features = model.classifier[-1].in_features
                            model.classifier[-1] = nn.Linear(in_features, num_classes)
                    elif hasattr(model, 'head'):
                        in_features = model.head.in_features
                        model.head = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Model {model_name} not found in torchvision")

        except (ImportError, ValueError):
            # Try timm models
            try:
                import timm
                model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            except ImportError:
                raise ImportError(
                    "Could not load model. Install timm: pip install timm"
                )
            except Exception as e:
                raise ValueError(
                    f"Could not load model {model_name}. Error: {e}"
                )

    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model
