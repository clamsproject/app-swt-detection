"""Regression tests for the stitcher half of app.py.

Written test-first: both tests are RED against the current code and pin
two known bugs; they flip green when the fixes land.

- an empty ``tfLabelMap`` (what ``tfLabelMapPreset=nomap`` or
  ``nopreset`` resolves to) crashes ``_annotate_timeframes`` with a
  ``NameError``: ``src_label_set`` is defined inside the
  ``if parameters['tfLabelMap']:`` block but used outside it.
- the uniform-sampling guard is one-sided: it only rejects inputs whose
  later gap is *smaller* than the first, so gaps that grow pass the
  check they were meant to fail.
"""
import json

import pytest

# app.py imports modeling.data_loader, which needs the training stack
pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('transformers')
pytest.importorskip('mmif')
pytest.importorskip('clams')

from mmif import (Mmif, Document, AnnotationTypes, DocumentTypes,
                  __specver__)

import app as swt_app

STITCH_PARAMS = {
    'useClassifier': False,
    'tfLabelMap': {},
    'tfMinTFDuration': 2000,
    'tfMinTPScore': 0.5,
    'tfMinTFScore': 0.5,
    'tfDynamicSceneLabels': ['credits'],
    'tfAllowOverlap': True,
}


def build_tp_mmif(timepoints, labelset=('bars', '-')):
    """Minimal MMIF: one video document and one TimePoint view."""
    skel = json.dumps(
        {'metadata': {'mmif': f'http://mmif.clams.ai/{__specver__}'},
         'documents': [], 'views': []})
    mmif = Mmif(skel, validate=False)
    vd = Document()
    vd.at_type = DocumentTypes.VideoDocument
    vd.id = 'd1'
    vd.location = 'file:///dev/null'
    vd.add_property('fps', 30)
    mmif.add_document(vd)
    view = mmif.new_view()
    view.metadata.app = 'test/fixture'
    view.new_contain(AnnotationTypes.TimePoint, document='d1',
                     timeUnit='milliseconds', labelset=list(labelset))
    for t in timepoints:
        view.new_annotation(AnnotationTypes.TimePoint, timePoint=t,
                            label='bars',
                            classification={'bars': 0.9, '-': 0.1})
    # roundtrip so the annotations carry the contains-level properties
    # (document, timeUnit) the way a deserialized MMIF would
    return Mmif(mmif.serialize())


def test_empty_label_map_does_not_crash():
    # tfLabelMapPreset=nomap / nopreset both resolve to an empty map;
    # stitching must run with labels kept as-is, not raise
    mmif = build_tp_mmif(list(range(0, 10_000, 1000)))
    app = swt_app.SwtDetection()
    app._annotate_timeframes(mmif, **STITCH_PARAMS)
    tf_views = [v for v in mmif.views
                if AnnotationTypes.TimeFrame in v.metadata.contains]
    assert tf_views, 'stitching produced no TimeFrame view'


def test_nonuniform_sampling_rejected_when_later_gap_larger():
    # gaps are 1000 then 3000; a one-sided check lets this through
    mmif = build_tp_mmif([0, 1000, 4000])
    app = swt_app.SwtDetection()
    with pytest.raises(ValueError):
        app._annotate_timeframes(mmif, **STITCH_PARAMS)


if __name__ == '__main__':
    pytest.main([__file__])
