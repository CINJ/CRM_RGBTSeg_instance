import torch
import json


class TorchTensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()  # Convert single-value tensor to a Python scalar
        return super(TorchTensorEncoder, self).default(obj)
