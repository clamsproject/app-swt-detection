"""

CLAMS app to detect scenes with text.

The kinds of scenes that are recognized depend on the model used but typically
include slates, chyrons and credits.

"""

import time
import argparse
import logging
from pathlib import Path
from typing import Union

import yaml
from clams import ClamsApp, Restifier
from mmif import Mmif, View, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

from modeling import classify, stitch, negative_label, FRAME_TYPES

default_config_fname = Path(__file__).parent / 'modeling/config/classifier.yml'
default_model_storage = Path(__file__).parent / 'modeling/models'


class SwtDetection(ClamsApp):

    def __init__(self, preconf_fname: str = None, log_to_file: bool = False) -> None:
        super().__init__()
        self.preconf = yaml.safe_load(open(preconf_fname))
        if log_to_file:
            fh = logging.FileHandler(f'{self.__class__.__name__}.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

    def _appmetadata(self):
        # using metadata.py
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        # parameters here is a "refined" dict, so hopefully its values are properly
        # validated and casted at this point. 
        self.parameters = parameters
        self.configs = {**self.preconf, **parameters}
        self._configure_model()
        self._configure_postbin()
        for k, v in self.configs.items():
            self.logger.debug(f"Final Configuration: {k} :: {v}")

        videos = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not videos:
            warning = Warning('There were no video documents referenced in the MMIF file')
            classifier_view.metadata.add_warnings(warning)
            return mmif
        video = videos[0]
        self.logger.info(f"Processing video {video.id} at {video.location_path()}")

        extracted, positions, total_ms = self._extract_images(video)

        predictions = self._classify(extracted, positions, total_ms)
        at_types = [AnnotationTypes.TimePoint]
        labels = FRAME_TYPES + [negative_label]
        classifier_view = self._new_view(at_types, video, labels, mmif)
        self._add_classifier_results_to_view(predictions, classifier_view)

        if self.configs.get('useStitcher'):
            stitcher = stitch.Stitcher(**self.configs)
            self.logger.info('Minimum time frame score: %s', stitcher.min_timeframe_score)
            timeframes = stitcher.create_timeframes(predictions)
            at_types = [AnnotationTypes.TimePoint, AnnotationTypes.TimeFrame]
            labels = list(self.configs['postbin'].keys())
            stitcher_view = self._new_view(at_types, video, labels, mmif)
            self._add_stitcher_results_to_view(timeframes, stitcher_view)

        return mmif

    def _configure_model(self):
        model_name = self.parameters["modelName"]
        self.configs['model_file'] = default_model_storage / f'{model_name}.pt'
        self.configs['model_config_file'] = default_model_storage / f'{model_name}.yml'

    def _configure_postbin(self):
        """
        Set the postbin property of the the configs configuration dictionary, using the
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
        that the underscore is replaced with a colon. This is not good if we intend
        there to be a dash.
        """
        # TODO: this is ugly, but I do not know a better way yet. The default value
        # of the map parameter in metadata.py is an empty list. If the user sets those
        # parameters during invocation (for example "?map=S:slate&map=B:bar") then in
        # the user parameters we have ['S:slate', 'B:bar'] for map and in the refined
        # parameters we get {'S': 'slate', 'B': 'bar'}. If the user adds no map
        # parameters then there is no map value in the user parameters and the value
        # is [] in the refined  parameters (which is a bit inconsistent).
        # Two experiments:
        # 1. What if I set the default to a list like ['S:slate', 'B:bar']?
        #    Then the map value in refined parameters is that same list, which means
        #    that I have to turn it into a dictionary before I hand it off.
        # 2. What if I set the default to a dictionary like {'S': 'slate', 'B': 'bar'}?
        #    Then the map value in the refined parameters is a list with one element,
        #    which is the wanted dictionary as a string: ["{'S': 'slate', 'B': 'bar'}"]
        if type(self.parameters['map']) is list:
            newmap = {}
            for kv in self.parameters['map']:
                k, v = kv.split(':')
                newmap[k] = v
            self.parameters['map'] = newmap
            self.configs['map'] = newmap
        postbin = invert_mappings(self.parameters['map'])
        self.configs['postbin'] = postbin

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
        self.logger.info(f"Initiating classifier with {self.configs['model_file']}")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.configs['logger_name'] = self.logger.name
        classifier = classify.Classifier(**self.configs)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Classifier initiation took {time.perf_counter() - t:.2f} seconds")
        predictions = classifier.classify_images(extracted, positions, total_ms)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Processing took {time.perf_counter() - t:.2f} seconds")
        return predictions

    def _new_view(self, annotation_types: list, video, labels: list, mmif):
        view: View = mmif.new_view()
        self.sign_view(view, self.parameters)
        for annotation_type in annotation_types:
            view.new_contain(
                annotation_type, document=video.id, timeUnit='milliseconds', labelset=labels)
        return view

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

    def _add_stitcher_results_to_view(self, timeframes: list, view: View):
        for tf in timeframes:
            timeframe_annotation = view.new_annotation(AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("label", tf.label),
            timeframe_annotation.add_property('classification', {tf.label: tf.score})
            timeframe_annotation.add_property('targets', [target.annotation.id for target in tf.targets])
            timeframe_annotation.add_property("representatives",
                                              [p.annotation.id for p in tf.representative_predictions()])
            for prediction in tf.targets:
                timepoint_annotation = view.new_annotation(AnnotationTypes.TimePoint)
                prediction.annotation = timepoint_annotation
                scores = [prediction.score_for_label(lbl) for lbl in prediction.labels]
                classification = {l:s for l, s in zip(prediction.labels, scores)}
                classification = transform(classification, self.configs['postbin'])
                label = max(classification, key=classification.get)
                timepoint_annotation.add_property('timePoint', prediction.timepoint)
                timepoint_annotation.add_property('label', label)
                timepoint_annotation.add_property('classification', classification)


def invert_mappings(mappings: dict) -> dict:
    print('-'*80)
    print(mappings)
    inverted_mappings = {}
    for in_label, out_label in mappings.items():
        in_label = restore_colon(in_label)
        inverted_mappings.setdefault(out_label, []).append(in_label)
    return inverted_mappings


def restore_colon(label_in: str) -> str:
    """Replace a dash with a colon."""
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
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The YAML config file", default=default_config_fname)
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = SwtDetection(preconf_fname=parsed_args.config, log_to_file=False)

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
