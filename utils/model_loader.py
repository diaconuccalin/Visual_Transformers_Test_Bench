"""
Model loading utilities for various model formats.
"""

import torch
import torch.nn as nn
from pathlib import Path
from .tflite_wrapper import TFLiteModelWrapper


def load_model(model_path, num_classes=2, device='cuda'):
    """
    Load a model from file or use a pretrained model.

    Args:
        model_path (str): Path to model checkpoint or model name
        num_classes (int): Number of output classes
        device (str): Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded PyTorch model or TFLite wrapper
        dict: Metadata including 'preprocessing' type
    """
    model_path = Path(model_path)

    # Check if it's a TFLite model
    if str(model_path).endswith('.tflite') and model_path.exists():
        print(f"Loading TFLite model from {model_path}")
        model = TFLiteModelWrapper(str(model_path), device=device)
        model.eval()
        # TFLite models need special preprocessing
        return model, {'preprocessing': 'tflite'}

    # Check if it's a file path
    if model_path.exists():
        print(f"Loading model from {model_path}")

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                # Checkpoint contains state_dict - need to reconstruct model
                state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
                architecture = checkpoint.get('architecture', 'resnet18')
                num_classes_saved = checkpoint.get('num_classes', num_classes)

                print(f"Loading {architecture} from state_dict with {num_classes_saved} classes...")

                # Import torchvision models
                import torchvision.models as models

                # Create the model architecture
                if hasattr(models, architecture):
                    model = getattr(models, architecture)(pretrained=False)

                    # Adapt classifier for the number of classes
                    if hasattr(model, 'classifier'):
                        if isinstance(model.classifier, nn.Sequential):
                            # MobileNet style
                            in_features = model.classifier[-1].in_features
                            model.classifier[-1] = nn.Linear(in_features, num_classes_saved)
                        else:
                            in_features = model.classifier.in_features
                            model.classifier = nn.Linear(in_features, num_classes_saved)
                    elif hasattr(model, 'fc'):
                        in_features = model.fc.in_features
                        model.fc = nn.Linear(in_features, num_classes_saved)

                    # Load the state dict
                    model.load_state_dict(checkpoint[state_dict_key])
                else:
                    raise ValueError(f"Unknown architecture: {architecture}")
            else:
                # Direct state_dict (e.g., from chenyaofo/pytorch-cifar-models)
                # These are bare state_dicts for specific architectures
                print("Loading model from bare state_dict...")
                
                # Try to infer architecture from checkpoint keys and num_classes
                # chenyaofo models use MobileNetV2 architecture for CIFAR-10
                if 'features.0.0.weight' in checkpoint:
                    # Check if this looks like a MobileNetV2
                    if num_classes == 10 and 'classifier.1.weight' in checkpoint:
                        print("Detected MobileNetV2 architecture for CIFAR-10")
                        import torchvision.models as models
                        model = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
                        model.load_state_dict(checkpoint)
                    else:
                        # Unknown bare state_dict - try to load as-is
                        print(f"Warning: Unknown state_dict format. Attempting to load directly...")
                        model = checkpoint
                else:
                    model = checkpoint
        else:
            # Direct model object
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
    # PyTorch models use ImageNet preprocessing
    return model, {'preprocessing': 'imagenet'}
