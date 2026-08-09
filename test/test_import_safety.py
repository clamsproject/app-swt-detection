"""Import-safety smoke tests for the modeling package.

Written test-first: the preprocessor test is RED against the current
code. ``modeling.data_loader`` imports pandas and h5py inside ``main()``
(function scope), so ``TrainingDataPreprocessor`` methods cannot see
them and the preprocessing CLI dies with ``NameError`` on first use.
"""
import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('transformers')

from modeling import backbones, data_loader  # noqa: F401


def test_core_modules_importable():
    # modules app.py / classify.py depend on must import cleanly
    from modeling import classify, gridsearch, train  # noqa: F401


def test_validate_importable():
    pytest.importorskip('torchmetrics')
    from modeling import validate  # noqa: F401


def test_preprocessor_constructible(tmp_path):
    pytest.importorskip('pandas')
    csv = tmp_path / 'GUID1.csv'
    csv.write_text('at,scene-type,scene-subtype\n'
                   '00:00:01.0,B,\n'
                   '00:00:02.500,S,\n')
    prep = data_loader.TrainingDataPreprocessor(
        csv, tmp_path / 'GUID1.zip', tmp_path)
    assert prep.guid == 'GUID1'
    assert list(prep.metadata['at']) == [1000, 2500]


if __name__ == '__main__':
    pytest.main([__file__])
