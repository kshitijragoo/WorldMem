import torch
import torch.nn as nn
from typing import Dict

class DINOv3FeatureExtractor:
    """
    A wrapper class for loading a frozen DINOv3 model and extracting dense patch features.
    This class ensures the model is loaded only once and operates in an efficient,
    inference-only mode.
    """
    def __init__(self, hf_model_name: str = 'facebook/dinov3-vitl16-pretrain-lvd1689m', device: str = 'cuda'):
        """
        Initializes the feature extractor using a Hugging Face model identifier.

        Args:
            hf_model_name (str): The Hugging Face model identifier.
            device (str): The compute device ('cuda' or 'cpu').
        """
        self.device = device
        self.model = self._load_and_freeze_model(hf_model_name)

    def _load_and_freeze_model(self, hf_model_name: str) -> nn.Module:
        """
        Loads the specified DINOv3 model from Hugging Face, sets it to
        evaluation mode, and freezes all its parameters.
        """
        try:
            # Load the model using the transformers library
            model = AutoModel.from_pretrained(hf_model_name)
            model.eval()
            model.to(self.device)

            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            print(f"Successfully loaded and froze DINOv3 model '{hf_model_name}' on {self.device}.")
            return model
        except Exception as e:
            print(f"Error loading DINOv3 model from Hugging Face: {e}")
            return None

    def extract_patch_features(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Extracts dense patch features from a batch of images.

        Args:
            image_batch (torch.Tensor): A batch of preprocessed images with shape
                                         [batch_size, channels, height, width], 
                                         already on the correct device.

        Returns:
            torch.Tensor: A tensor of patch features with shape 
                          [batch_size, num_patches, feature_dimension].
        """

        if self.model is None:
            raise RuntimeError("DINOv3 model is not loaded. Cannot extract features.")


        if image_batch.device!= self.device:
            image_batch = image_batch.to(self.device)

        with torch.inference_mode():
            # The model's forward pass returns a dictionary of feature types
            features_dict = self.model.forward_features(image_batch)
            
            # We are interested in the final, normalized patch tokens
            patch_features = features_dict.get('x_norm_patchtokens')
            
            if patch_features is None:
                raise KeyError("Could not find 'x_norm_patchtokens' in model output. "
                               "Available keys: " + str(features_dict.keys()))
                               
        return patch_features
