import torch
import torch.nn as nn
from typing import Dict

class DINOv3FeatureExtractor:
    """
    A wrapper class for loading a frozen DINOv3 model and extracting dense patch features.
    This class ensures the model is loaded only once and operates in an efficient,
    inference-only mode.
    """
    def __init__(self, model_name: str = 'dinov3_vitl16', device: str = 'cuda'):
        """
        Initializes the feature extractor.

        Args:
            model_name (str): The name of the DINOv3 model from torch.hub.
            device (str): The compute device ('cuda' or 'cpu').
        """
        self.device = device
        self.model = self._load_and_freeze_model(model_name)
        
    def _load_and_freeze_model(self, model_name: str) -> nn.Module:
        """
        Loads the specified DINOv3 model, sets it to evaluation mode, and freezes
        all its parameters.
        """
        try:
            model = torch.hub.load('facebookresearch/dinov3', model_name, pretrained=True)
            model.eval()
            model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            print(f"Successfully loaded and froze DINOv3 model '{model_name}' on {self.device}.")
            return model
        except Exception as e:
            print(f"Error loading DINOv3 model: {e}")
            raise

    def extract_patch_features(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Extracts dense patch features from a batch of images.

        Args:
            image_batch (torch.Tensor): A batch of preprocessed images with shape
                                       , already on the correct device.

        Returns:
            torch.Tensor: A tensor of patch features with shape 
                         .
        """
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