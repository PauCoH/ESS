import os


def _configure_cuda_workspace():
    """Set the cuBLAS workspace before importing modules that load PyTorch.

    Import this package before using CUDA. An already-used notebook kernel
    must be restarted because cuBLAS may have cached its workspace settings.
    Preserve an explicit environment setting for validation at run startup.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


_configure_cuda_workspace()

from .config import ESSConfig, FineTuningConfig, LearningRateConfig
from .runner import run_experiment

__all__ = ['ESSConfig', 'FineTuningConfig', 'LearningRateConfig', 'run_experiment']
