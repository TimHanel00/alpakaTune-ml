"""PyTorch implementation of the deployment-compatible DeepSets ranker."""

from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exception:
        raise RuntimeError(
            "PyTorch is required for training; install requirements-train.txt in a project-local environment"
        ) from exception
    return torch


def build_ranker(
    context_feature_count: int,
    token_hidden_sizes: tuple[int, int] = (16, 32),
    embedding_size: int = 32,
):
    torch = require_torch()
    first_hidden, second_hidden = token_hidden_sizes

    class DeepSetsRanker(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = torch.nn.Sequential(
                torch.nn.Linear(18, first_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(first_hidden, second_hidden),
                torch.nn.ReLU(),
            )
            self.context = torch.nn.Sequential(
                torch.nn.Linear(2 * second_hidden + context_feature_count, embedding_size),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_size, embedding_size),
                torch.nn.ReLU(),
            )
            # "cpu" is an existing torch.nn.Module method and therefore cannot
            # be used as a ModuleDict child name on current PyTorch releases.
            self.cpu_adapter = torch.nn.Linear(embedding_size, 1)
            self.gpu_adapter = torch.nn.Linear(embedding_size, 1)

        def forward(self, dimensions, mask, context, device_class):
            encoded = self.token(dimensions)
            expanded_mask = mask.unsqueeze(-1)
            denominator = expanded_mask.sum(dim=1).clamp_min(1.0)
            mean_pool = (encoded * expanded_mask).sum(dim=1) / denominator
            minimum = torch.finfo(encoded.dtype).min
            max_pool = encoded.masked_fill(~expanded_mask.bool(), minimum).max(dim=1).values
            embedding = self.context(torch.cat((mean_pool, max_pool, context), dim=1))
            cpu = self.cpu_adapter(embedding).squeeze(1)
            gpu = self.gpu_adapter(embedding).squeeze(1)
            prediction = torch.where(device_class.bool(), gpu, cpu)
            return prediction, embedding

    return DeepSetsRanker()
