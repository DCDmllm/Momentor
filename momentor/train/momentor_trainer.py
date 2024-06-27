import os
import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class MomentorTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Save the model
        _state_dict = state_dict
        if _state_dict is None:
            # Only save the model itself if we are using distributed training
            model_to_save = unwrap_model(self.model)
            _state_dict = model_to_save.state_dict()

        weight_to_save = {}
        keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in', 'temporal_input_embeddings', 'temporal_output_embeddings']
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in keys_to_match):
                weight_to_save[k] = v

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'), )