"""Structural and domain checks for modeling.config.bins.

Torch-free: runs in any environment.

The schemes in ``binning_schemes`` serve two different domains (kept
together for historical reasons):

- prebin schemes map raw single-letter annotation labels (FRAME_TYPES)
  into training bins; used as the ``prebin`` axis at training time.
- postbin presets map collapse-close bin names into display bins; used
  as ``tfLabelMapPreset`` values at stitching time.

These tests pin each scheme to its domain so a label typo (which would
silently send real labels to the negative bin) fails loudly, and so any
newly added scheme must be classified below before the suite passes.
"""
import pytest

from modeling import FRAME_TYPES, negative_label
from modeling.config import bins

EMPTY_SCHEMES = {'noprebin', 'nomap'}
PREBIN_SCHEMES = {
    'collapse-close',
    'collapse-close-reduce-difficulty',
    'collapse-close-bin-lower-thirds',
    'ignore-difficulties',
}
POSTBIN_SCHEMES = {
    'strict', 'simple', 'simpler', 'relaxed',
    'binary-bars', 'binary-slate', 'binary-chyron-strict',
    'binary-chyron-relaxed', 'binary-credits',
}


def test_every_scheme_is_domain_classified():
    # a new scheme must be added to one of the sets above, so its label
    # domain gets checked by the tests below
    assert (set(bins.binning_schemes)
            == EMPTY_SCHEMES | PREBIN_SCHEMES | POSTBIN_SCHEMES)


@pytest.mark.parametrize('name', sorted(EMPTY_SCHEMES))
def test_empty_schemes_are_empty(name):
    assert bins.binning_schemes[name] == {}


@pytest.mark.parametrize('name',
                         sorted(PREBIN_SCHEMES | POSTBIN_SCHEMES))
def test_scheme_structure(name):
    scheme = bins.binning_schemes[name]
    assert isinstance(scheme, dict) and scheme
    for bin_name, labels in scheme.items():
        assert isinstance(bin_name, str) and bin_name
        assert isinstance(labels, list) and labels
        for label in labels:
            assert isinstance(label, str) and label


@pytest.mark.parametrize('name',
                         sorted(PREBIN_SCHEMES | POSTBIN_SCHEMES))
def test_no_duplicate_labels_within_scheme(name):
    scheme = bins.binning_schemes[name]
    all_labels = [lbl for labels in scheme.values() for lbl in labels]
    assert len(all_labels) == len(set(all_labels))


@pytest.mark.parametrize('name', sorted(PREBIN_SCHEMES))
def test_prebin_schemes_use_raw_labels(name):
    scheme = bins.binning_schemes[name]
    labels = {lbl for lbls in scheme.values() for lbl in lbls}
    assert labels <= set(FRAME_TYPES), labels - set(FRAME_TYPES)


@pytest.mark.parametrize('name', sorted(POSTBIN_SCHEMES))
def test_postbin_schemes_use_collapse_close_names(name):
    valid = (set(bins.binning_schemes['collapse-close'].keys())
             | {negative_label})
    scheme = bins.binning_schemes[name]
    labels = {lbl for lbls in scheme.values() for lbl in lbls}
    assert labels <= valid, labels - valid


def test_gridsearch_prebin_values_are_prebin_schemes():
    # gridsearch imports the backbone registry, which needs the
    # torch/transformers stack, so guard
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from modeling import gridsearch
    for value in gridsearch.prebin:
        if isinstance(value, str):
            assert value in PREBIN_SCHEMES | EMPTY_SCHEMES, value


if __name__ == '__main__':
    pytest.main([__file__])
