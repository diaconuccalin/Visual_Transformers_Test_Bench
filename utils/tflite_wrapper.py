"""
TFLite model wrapper for PyTorch-compatible inference.

This module provides a wrapper around TensorFlow Lite models to make them
compatible with the PyTorch-based benchmark infrastructure.
"""

import torch
import torch.nn as nn
import numpy as np


class TFLiteModelWrapper(nn.Module):
    """
    Wrapper to use TFLite models with PyTorch-style interface.

    This allows TFLite models to be used in the benchmark pipeline
    alongside PyTorch models.
    """

    def __init__(self, tflite_path, device='cpu'):
        """
        Initialize TFLite model wrapper.

        Args:
            tflite_path (str): Path to .tflite model file
            device (str): Device ('cpu' or 'cuda' - note: TFLite only runs on CPU)
        """
        super().__init__()

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required to use TFLite models.\n"
                "Install with: pip install tensorflow"
            )

        if device == 'cuda':
            print("WARNING: TFLite models only run on CPU. Ignoring CUDA device.")
            device = 'cpu'

        self.device_name = device
        self.tflite_path = tflite_path

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Store input/output shapes
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']

        print(f"TFLite model loaded from {tflite_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output shape: {self.output_shape}")

    def forward(self, x):
        """
        Forward pass through TFLite model.

        Args:
            x (torch.Tensor): Input tensor in PyTorch format (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor
        """
        # Convert PyTorch tensor to numpy
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        # TFLite expects NHWC format, PyTorch uses NCHW
        # Transpose from (B, C, H, W) to (B, H, W, C)
        if len(x_np.shape) == 4 and x_np.shape[1] == 3:  # NCHW format
            x_np = np.transpose(x_np, (0, 2, 3, 1))

        batch_size = x_np.shape[0]
        outputs = []

        # Process each sample in the batch
        for i in range(batch_size):
            sample = np.expand_dims(x_np[i], axis=0).astype(np.float32)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], sample)

            # Run inference
            self.interpreter.invoke()

            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            outputs.append(output)

        # Stack outputs
        output_np = np.vstack(outputs)

        # Convert back to PyTorch tensor
        output_tensor = torch.from_numpy(output_np).float()

        return output_tensor

    def eval(self):
        """Set model to evaluation mode (for compatibility with PyTorch)."""
        return self

    def to(self, device):
        """Move to device (for compatibility with PyTorch)."""
        if device != 'cpu' and str(device) != 'cpu':
            print(f"WARNING: TFLite models only run on CPU. Cannot move to {device}")
        return self

    def __call__(self, x):
        """Make the wrapper callable like a PyTorch model."""
        return self.forward(x)
