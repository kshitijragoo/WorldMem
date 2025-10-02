# algorithms/worldmem/dinov3_feature_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import AutoModel


class DINOv3FeatureExtractor:
    """
    A wrapper class for loading a frozen DINOv3 model and extracting dense patch features.
    This class ensures the model is loaded only once and operates in an efficient,
    inference-only mode.
    """
    def __init__(self, model_id: str = 'facebook/dinov3-vits16-pretrain-lvd1689m', device: str = 'cuda'):
        """
        Initializes the feature extractor using a Hugging Face model identifier.

        Args:
            hf_model_name (str): The Hugging Face model identifier.
            device (str): The compute device ('cuda' or 'cpu').
        """
        self.device = device
        self.model = self._load_and_freeze_model(model_id)

    def _load_and_freeze_model(self, model_id: str) -> nn.Module:
        """
        Loads the specified DINOv3 model from Hugging Face, sets it to
        evaluation mode, and freezes all its parameters.
        """
        try:
            model = AutoModel.from_pretrained(model_id)
            model.eval()
            model.to(self.device)

            for param in model.parameters():
                param.requires_grad = False
            
            print(f"Successfully loaded and froze DINOv3 model '{model_id}' on {self.device}.")
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
            # Hugging Face DINOv3 models return a BaseModelOutput with last_hidden_state
            # Shape: [batch_size, num_tokens (1 + num_patches), hidden_dim]
            outputs = self.model(image_batch)
            if not hasattr(outputs, 'last_hidden_state'):
                raise AttributeError("Model output does not have 'last_hidden_state'.")

            token_embeddings = outputs.last_hidden_state
            if token_embeddings.dim() != 3 or token_embeddings.size(1) < 2:
                raise ValueError("Unexpected DINOv3 output shape: "
                                 f"{tuple(token_embeddings.shape)}")

            # Exclude class token at index 0 to keep only patch tokens
            patch_features = token_embeddings[:, 1:, :]

            # Optionally L2-normalize patch features to match common DINO usage
            patch_features = F.normalize(patch_features, dim=-1)
                               
        return patch_features
