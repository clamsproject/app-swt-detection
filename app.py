"""
CLAMS app to detect scenes with text.

The kinds of scenes that are recognized depend on the model used but typically
include slates, chyrons and credits.

"""

import argparse
import logging
import math
import time
import warnings
from collections import namedtuple
from typing import Union

from clams import ClamsApp, Restifier
from mmif import Mmif, View, AnnotationTypes, DocumentTypes, Annotation, Document
from mmif.utils import video_document_helper as vdh
from mmif.utils import sequence_helper as sqh

from metadata import default_model_storage
from modeling import classify, stitch, negative_label, FRAME_TYPES


class SwtDetection(ClamsApp):

    def __init__(self, log_to_file: bool = False) -> None:
        super().__init__()
        if log_to_file:
            fh = logging.FileHandler(f'{self.__class__.__name__}.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

    def _appmetadata(self):
        # using metadata.py
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        # parameters here is a "refined" dict, so hopefully its values are properly
        # validated and casted at this point.
        self.configs = parameters.copy()
        self._configure_postbin()
        for k, v in self.configs.items():
            self.logger.debug(f"Final Configuration: {k} :: {v}")
        videos = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not videos:
            warnings.warn('There were no video documents referenced in the MMIF file', UserWarning)
            return mmif
        video = videos[0]
        self.logger.info(f"Processing video {video.id} at {video.location_path()}")
        if parameters.get('useClassifier'):
            self._annotate_timepoints(mmif, **parameters)
        if parameters.get('useStitcher'):
            self._annotate_timeframes(mmif, **parameters)
        return mmif
    
    @staticmethod
    def _get_first_videodocument(mmif: Mmif) -> Union[Document, None]:
        videos = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not videos:
            warnings.warn('There were no video documents referenced in the MMIF file', UserWarning)
            return None
        return videos[0]

    def _configure_postbin(self):
        """
        Set the postbin property of the configs configuration dictionary, using the
        label mapping parameters if there are any, otherwise using the label mapping from
        the default configuration file.

        This should set the postbin property to something like 

            {'bars': ['B'],
             'chyron': ['I', 'N', 'Y'],
             'credit': ['C', 'R'],
             'other_opening': ['W', 'L', 'O', 'M'],
             'other_text': ['E', 'K', 'G', 'T', 'F'],
             'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G']}

        Note that the labels cannot have colons in them, but historically we did have
        colons in the SWT annotation for subtypes of "slate". Syntactically, we cannot
        have mappings like S:H:slate. This here assumes the mapping is S-H:slate and
        that the dash is replaced with a colon. This is not good if we intend there to
        be a dash.
        """
        self.configs['postbin'] = invert_mappings(self.configs['map'])
        
    def _annotate_timepoints(self, mmif: Mmif, **parameters) -> Mmif:
        video = self._get_first_videodocument(mmif)
        if video is None:
            return mmif
        tp_labels = FRAME_TYPES + [negative_label]
        extracted, positions, total_ms = self._extract_images(video)

        v = mmif.new_view()
        self.sign_view(v, parameters)
        v.new_contain(
            AnnotationTypes.TimePoint,
            document=video.id, timeUnit='milliseconds', labelset=tp_labels)

        predictions = self._classify(extracted, positions, total_ms)
        self._add_classifier_results_to_view(predictions, v)

    def _extract_images(self, video):
        open_video(video)
        fps = video.get_property('fps')
        total_frames = video.get_property(vdh.FRAMECOUNT_DOCPROP_KEY)
        total_ms = int(vdh.framenum_to_millisecond(video, total_frames))
        start_ms = max(0, self.configs['startAt'])
        final_ms = min(total_ms, self.configs['stopAt'])
        sframe, eframe = [vdh.millisecond_to_framenum(video, p) for p in [start_ms, final_ms]]
        sampled = vdh.sample_frames(sframe, eframe, self.configs['sampleRate'] / 1000 * fps)
        self.logger.info(f'Sampled {len(sampled)} frames ' +
                         f'btw {start_ms} - {final_ms} ms (every {self.configs["sampleRate"]} ms)')
        t = time.perf_counter()
        positions = [int(vdh.framenum_to_millisecond(video, sample)) for sample in sampled]
        extracted = vdh.extract_frames_as_images(video, sampled, as_PIL=True)
        self.logger.debug(f"Seeking time: {time.perf_counter() - t:.2f} seconds\n")
        # the last `total_ms` (as a fixed value) only works since the app is processing only
        # one video at a time     
        return extracted, positions, total_ms

    def _classify(self, extracted: list, positions: list, total_ms: int):
        t = time.perf_counter()
        # in the following, the .glob() should always return only one, otherwise we have a problem
        model_filestem = next(default_model_storage.glob(
            f"*.{self.configs['modelName']}.pos{'T' if self.configs['usePosModel'] else 'F'}.pt")).stem
        self.logger.info(f"Initiating classifier with {model_filestem}")
        classifier = classify.Classifier(default_model_storage / model_filestem, 
                                         self.logger.name if self.logger.isEnabledFor(logging.DEBUG) else None)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Classifier initiation took {time.perf_counter() - t:.2f} seconds")
        predictions = classifier.classify_images(extracted, positions, total_ms)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Processing took {time.perf_counter() - t:.2f} seconds")
        return predictions

    def _add_classifier_results_to_view(self, predictions: list, view: View):
        for prediction in predictions:
            timepoint_annotation = view.new_annotation(AnnotationTypes.TimePoint)
            prediction.annotation = timepoint_annotation
            scores = [prediction.score_for_label(lbl) for lbl in prediction.labels]
            classification = {l: s for l, s in zip(prediction.labels, scores)}
            label = max(classification, key=classification.get)
            timepoint_annotation.add_property('timePoint', prediction.timepoint)
            timepoint_annotation.add_property('label', label)
            timepoint_annotation.add_property('classification', classification)

    def _annotate_timeframes(self, mmif: Mmif, **parameters) -> Mmif:
        video = self._get_first_videodocument(mmif)
        if video is None:
            return mmif
        tf_labels = list(self.configs['postbin'].keys())
        predictions = list(mmif.get_annotations(AnnotationTypes.TimePoint))  # or something like this
        
        v = mmif.new_view()
        self.sign_view(v, parameters)
        v.new_contain(
            AnnotationTypes.TimeFrame,
            document=video.id, timeUnit='milliseconds', labelset=tf_labels)
        stitcher = stitch.Stitcher(**self.configs)
        self.logger.info('Minimum time frame score: %s', stitcher.min_timeframe_score)
        timeframes = stitcher.create_timeframes(predictions)
        self._add_stitcher_results_to_view(timeframes, v)
        
    def _add_stitcher_results_to_view(self, timeframes: list, view: View):
        for tf in timeframes:
            targets = [target.annotation.long_id for target in tf.targets]
            representatives = [p.annotation.long_id for p in tf.representative_predictions()]
            timeframe_annotation = view.new_annotation(AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("label", tf.label),
            timeframe_annotation.add_property('classification', {tf.label: tf.score})
            timeframe_annotation.add_property('targets', targets)
            timeframe_annotation.add_property("representatives", representatives)
    
    def _alternative_annotate_timeframes(self, mmif: Mmif, **parameters) -> Mmif:
        TimeFrameTuple = namedtuple('TimeFrame', 
                                    ['label', 'tf_score', 'targets', 'representatives'])
        tp_view = mmif.get_view_contains(AnnotationTypes.TimePoint)
        if not tp_view:
            self.logger.info("No TimePoint annotations found.")
            return mmif
        tps = list(tp_view.get_annotations(AnnotationTypes.TimePoint))

        # first, figure out time point sampling rate by looking at the first three annotations
        # why 3? just as a sanity check
        if len(tps) < 3:
            raise ValueError("At least 3 TimePoint annotations are required to stitch.")
        # and then figure out the time point sampling rate
        testsamples = [vdh.convert_timepoint(mmif, tp, 'milliseconds') for tp in tps[:3]]
        if parameters['useClassifier']:
            tp_sampling_rate = parameters['tpSampleRate']
        else:
            tp_sampling_rate = testsamples[1] - testsamples[0]
        tolerance = 1000 / mmif.get_document_by_id(tps[0].get_property('document')).get_property('fps')
        self.logger.debug(f"TimePoint sampling rate 0-1: {tp_sampling_rate}")
        self.logger.debug(f"TimePoint sampling rate 1-2: {testsamples[2] - testsamples[1]}")
        if tp_sampling_rate - (testsamples[2] - testsamples[1]) > tolerance:
            raise ValueError("TimePoint annotations are not uniformly sampled.")

        # next, validate labels in the input annotations
        src_labels = sqh.validate_labelset(tps)

        # TODO: fill in `tfLabelMap` parameter value if a preset is used by the user
        self.logger.debug(f"Label map: {parameters['tfLabelMap']}")
        label_remapper = sqh.build_label_remapper(src_labels, parameters['tfLabelMap'])

        # then, build the score lists
        label_idx, scores = sqh.build_score_lists([tp.get_property('classification') for tp in tps],
                                                  label_remapper=label_remapper, score_remap_op=max)

        # keep track of the timepoints that have been included as TF targets
        used_timepoints = set()

        def has_overlapping_timeframes(timepoints: list):
            """
            Given a list of TPs, return True if there is a TP in the list that has already been used.
            """
            for timepoint in timepoints:
                if timepoint in used_timepoints:
                    return True
            return False

        all_tf = []
        # and stitch the scores
        for label, lidx in label_idx.items():
            if label == sqh.NEG_LABEL:
                continue
            stitched = sqh.smooth_outlying_short_intervals(
                scores[lidx],
                # parameters['minTFDuration']/1000, 
                math.ceil(parameters['tfMinTFDuration'] / tp_sampling_rate),
                1,  # does not smooth negative intervals
                parameters['tfMinTPScore']
            )
            self.logger.debug(f"\"{label}\" stitched: {stitched}")
            for positive_interval in stitched:
                tp_scores = scores[lidx][positive_interval[0]:positive_interval[1]]
                tf_score = tp_scores.mean()
                rep_idx = tp_scores.argmax() + positive_interval[0]
                if tf_score >= parameters['tfMinTFScore']:
                    target_list = [a.long_id for a in tps[positive_interval[0]:positive_interval[1]]]
                    all_tf.append(TimeFrameTuple(label=label, tf_score=tf_score, targets=target_list,
                                                 representatives=[tps[rep_idx].long_id]))
        if not parameters['tfAllowOverlap']:
            overlap_filter = []
            for tf in sorted(all_tf, key=lambda x: x.tf_score, reverse=True):
                if has_overlapping_timeframes(tf.targets):
                    continue
                for target_id in tf.targets:
                    used_timepoints.add(target_id)
                overlap_filter.append(tf)
            all_tf = overlap_filter

        # finally add everything to the output view
        v = mmif.new_view()
        self.sign_view(v, parameters)
        v.new_contain(AnnotationTypes.TimeFrame, labelset=list(set(label_remapper.values())))
        for tf in sorted(all_tf, key=lambda x: x.targets[0]):
            v.new_annotation(AnnotationTypes.TimeFrame,
                             label=tf.label,
                             classification={tf.label: tf.tf_score},
                             targets=tf.targets,
                             representatives=tf.representatives)

def invert_mappings(mappings: dict) -> dict:
    inverted_mappings = {}
    for in_label, out_label in mappings.items():
        in_label = restore_colon(in_label)
        inverted_mappings.setdefault(out_label, []).append(in_label)
    return inverted_mappings


def restore_colon(label_in: str) -> str:
    """Replace dashes with colons."""
    return label_in.replace('-', ':')


def open_video(video):
    """Open the video using the video_document_helper MMIF utility. This is done
    for the side effect of adding basic metadata properties to the video document."""
    vdh.capture(video)


def transform(classification: dict, postbin: dict):
    """Transform a classification using basic prelables into a classification with
    post labels only."""
    transformed = {}
    for postlabel, prelabels in postbin.items():
        transformed[postlabel] = sum([classification[lbl] for lbl in prelabels])
    return transformed


def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    return SwtDetection(log_to_file=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
