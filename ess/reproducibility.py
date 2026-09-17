import os
import random
import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    """Seed coin flips, action noise and model initialization before creating a model."""
    if deterministic:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8', ':16:8'}:
            raise RuntimeError(
                'Deterministic runs require CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8. '
                'Set it before starting Python/Jupyter, restart the kernel, and run all cells.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(requested):
    """Resolve automatic CPU/CUDA selection and reject unavailable requested CUDA."""
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available')
    return torch.device(requested)
