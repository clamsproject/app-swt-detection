"""Shared pytest configuration for the SWT test suite.

The suite mixes dependency-free logic tests (bins, naming) with tests
that need the training stack (torch, torchvision, transformers). On
machines without that stack, the stack-dependent modules skip themselves
at import time (``pytest.importorskip``); this conftest just makes those
skips loudly visible up front instead of silently shrinking the run.
"""
import importlib.util
import sys
import warnings
from pathlib import Path

# make `modeling` importable regardless of the invocation cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

TORCH_STACK = ('torch', 'torchvision', 'transformers')
MISSING_TORCH_STACK = tuple(
    m for m in TORCH_STACK if importlib.util.find_spec(m) is None)


def pytest_report_header(config):
    if MISSING_TORCH_STACK:
        return ('WARNING: training stack not installed (missing: '
                + ', '.join(MISSING_TORCH_STACK)
                + '); torch-based test modules will be skipped')
    return None


def pytest_configure(config):
    if MISSING_TORCH_STACK:
        warnings.warn(
            'training stack not installed (missing: '
            + ', '.join(MISSING_TORCH_STACK)
            + '); torch-based test modules will be skipped',
            UserWarning)


def install_mock_backbone(dim=256):
    """Replace the backbone registry with a weight-free dummy encoder.

    Lets FeatureExtractor-based tests run without downloading models or
    holding a GPU. Import-safe: only called from modules that already
    passed their ``pytest.importorskip`` guards.

    :return: the ``modeling.data_loader`` module, mock installed
    """
    import collections

    from modeling import data_loader

    class _DummyModel:
        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    class _DummyEncoder:
        def __init__(self):
            self.dim = dim
            self.model = _DummyModel()

    data_loader.backbones.model_map = collections.defaultdict(
        lambda: _DummyEncoder)
    return data_loader
