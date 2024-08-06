from unittest import TestCase

from mmif import Mmif, Document, Annotation
from mmif.vocabulary import AnnotationTypes, DocumentTypes

import app


class TestSimpleTimepointsStitcher(TestCase):
    def setUp(self):
        self.m = Mmif(validate=False)
        self.v = self.m.new_view()
        self.v.metadata.app = 'dummy-timepoint-annotations'
        self.m.add_document(Document(
            {'@type': DocumentTypes.VideoDocument, 
             'properties': {'id':'dummy', 'fps': 30}}))
        self.tu = 'milliseconds'
        self.tp_lbls = ['A', 'B', 'C']
        self.app = app.get_app()
        
    def _add_timepoints(self, view, start, end, step, label, score=0.5):
        for i in range(start, end, step):
            view.new_annotation(
                AnnotationTypes.TimePoint,
                timeUnit=self.tu, timePoint=i,
                labelset=self.tp_lbls, label=label,
                classification={lbl: score if lbl == label else (1 - score) / (len(self.tp_lbls) - 1) 
                                for lbl in self.tp_lbls},
                document='dummy'
            )
            
    def test_can_stitch_single_sequence(self):
        self._add_timepoints(self.v, 0, 10000, 1000, self.tp_lbls[0])
        stitched = Mmif(self.app.annotate(self.m.serialize()))
        self.assertEqual(len(stitched.views), 2)
        tf_view = stitched.views.get_last()
        self.assertEqual(len(list(tf_view.get_annotations(AnnotationTypes.TimeFrame))), 1)
        self.assertEqual(next(tf_view.get_annotations(AnnotationTypes.TimeFrame)).get('label'), self.tp_lbls[0])
    
    def test_can_stitch_overlapping_sequences(self):
        self._add_timepoints(self.v, 0, 10000, 1000, self.tp_lbls[0])
        stitched = Mmif(self.app.annotate(self.m.serialize(), allowOverlap=['true'], minTFScore=['0.1']))
        self.assertEqual(len(stitched.views), 2)
        tf_view = stitched.views.get_last()
        self.assertEqual(len(list(tf_view.get_annotations(AnnotationTypes.TimeFrame))), 3)
        tf_labels = []
        for tf in tf_view.get_annotations(AnnotationTypes.TimeFrame):
            self.assertTrue(tf.get('label') in self.tp_lbls)
            tf_labels.append(tf.get('label'))
        self.assertEqual(set(tf_labels), set(self.tp_lbls))
    
    def test_can_stitch_disallowing_overlaps(self):
        self._add_timepoints(self.v, 0, 10000, 1000, self.tp_lbls[0])
        stitched = Mmif(self.app.annotate(self.m.serialize(), allowOverlap=['false'], minTFScore=['0.1']))
        self.assertEqual(len(stitched.views), 2)
        tf_view = stitched.views.get_last()
        self.assertEqual(len(list(tf_view.get_annotations(AnnotationTypes.TimeFrame))), 1)
        self.assertEqual(next(tf_view.get_annotations(AnnotationTypes.TimeFrame)).get('label'), self.tp_lbls[0])
        self._add_timepoints(self.v, 10000, 15000, 1000, self.tp_lbls[1])
        # at this point, we have 
        # A: [.5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .25 .25 .25 .25 .25] (10 halves and 5 quarters)
        # B: [.25 .25 .25 .25 .25 .25 .25 .25 .25 .25 .5 .5 .5 .5 .5] (10 quarters and 5 halves)
        # C: [.25 .25 .25 .25 .25 .25 .25 .25 .25 .25 .25 .25 .25 .25 .25] (15 quarters)
        # if we set the minTFScore low enough, we should get 3 competing timeframes with mean scores of
        # A: .4166
        # B: .3333
        # C: .25
        tf_view = Mmif(self.app.annotate(self.m.serialize(), allowOverlap=['false'], 
                                         minTFScore=['0.1'])).views.get_last()
        self.assertEqual(len(list(tf_view.get_annotations(AnnotationTypes.TimeFrame))), 1)
        self.assertEqual(next(tf_view.get_annotations(AnnotationTypes.TimeFrame)).get('label'), self.tp_lbls[0])

    def test_can_stitch_disallowing_partial_overlaps(self):
        self._add_timepoints(self.v, 0, 10000, 1000, self.tp_lbls[0], score=0.4)
        self._add_timepoints(self.v, 10000, 15000, 1000, self.tp_lbls[1])
        # at this point, we have 
        # A: [.4 .4 .4 .4 .4 .4 .4 .4 .4 .4 .25 .25 .25 .25 .25] (10 2/5 and 5 quarters)
        # B: [.3 .3 .3 .3 .3 .3 .3 .3 .3 .3 .5 .5 .5 .5 .5] (10 thirds and 5 halves)
        # C: [.3 .3 .3 .3 .3 .3 .3 .3 .3 .3 .25 .25 .25 .25 .25] (10 thirds and 5 quarters)
        # if we set the minTPScore .3, the algorithm should ignore all the quarters, hence 
        # A: [.4 .4 .4 .4 .4 .4 .4 .4 .4 .4] (10 2/5 = 0.4 mean score through 10 points)
        # B: [.3 .3 .3 .3 .3 .3 .3 .3 .3 .3 .5 .5 .5 .5 .5] (10 thirds and 5 halves = 0.36 mean score through 15 points)
        # C: [.3 .3 .3 .3 .3 .3 .3 .3 .3 .3] (10 thirds = 0.3 mean score through 10 points)
        tf_view = Mmif(self.app.annotate(self.m.serialize(), allowOverlap=['false'], 
                                         minTPScore=['0.3'], minTFScore=['0.1'])).views.get_last()
        # only A survives
        self.assertEqual(len(list(tf_view.get_annotations(AnnotationTypes.TimeFrame))), 1)
        self.assertEqual(next(tf_view.get_annotations(AnnotationTypes.TimeFrame)).get('label'), self.tp_lbls[0])

