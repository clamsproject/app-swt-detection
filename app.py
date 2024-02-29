"""

CLAMS app to detect scenes with text.

The kinds of scenes that are recognized depend on the model used but typically
include slates, chyrons and credits.

"""

import argparse
import logging
from pathlib import Path
from typing import Union

import yaml
from clams import ClamsApp, Restifier
from mmif import Mmif, View, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

from modeling import classify, stitch, negative_label, train

logging.basicConfig(filename='swt.log', level=logging.DEBUG)

default_config_fname = Path(__file__).parent / 'modeling/config/classifier.yml'
default_model_storage = Path(__file__).parent / 'modeling/models'


class SwtDetection(ClamsApp):

    def __init__(self, preconf_fname: str = None, log_to_file: bool = False) -> None:
        super().__init__()
        self.preconf = yaml.safe_load(open(preconf_fname))
        # self.logger.addHandler(logging.StreamHandler())
        if log_to_file:
            fh = logging.FileHandler('swt.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        # possible bug here, as the configuration will be updated with the parameters that's not defined in the 
        # app metadata, but passed at the run time.
        configs = {**self.preconf, **parameters}
        if 'modelName' in parameters:
            configs['model_file'] = default_model_storage / f'{parameters["modelName"]}.pt'
            # model files from k-fold training have the fold number as three-digit suffix, trim it
            configs['model_config_file'] = default_model_storage / f'{parameters["modelName"][:-4]}_config.yml'
        self.logger.info(f"Initiating classifier with {configs['model_file']}")
        self.classifier = classify.Classifier(**configs)
        self.stitcher = stitch.Stitcher(**configs)

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        self.logger.info('Minimum time frame score: %s', self.stitcher.min_timeframe_score)

        vds = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not vds:
            warning = Warning('There were no video documents referenced in the MMIF file')
            new_view.metadata.add_warnings(warning)
            return mmif
        vd = vds[0]
        self.logger.info(f"Processing video {vd.id} at {vd.location_path()}")
        vcap = vdh.capture(vd)

        predictions = self.classifier.process_video(vcap)

        if self.use_stitcher:

            labelset = self.classifier.postbin_labels
            bins = self.classifier.model_config['bins']
            new_view.new_contain(
                AnnotationTypes.TimeFrame,
                document=vd.id, timeUnit='milliseconds', labelset=labelset)
            new_view.new_contain(
                AnnotationTypes.TimePoint,
                document=vd.id, timeUnit='milliseconds', labelset=labelset)

            timeframes = self.stitcher.create_timeframes(predictions)
            for tf in timeframes:
                timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
                timeframe_annotation.add_property("label", tf.label),
                timeframe_annotation.add_property('classification', {tf.label: tf.score})
                timepoint_annotations = []
                for prediction in tf.targets:
                    timepoint_annotation = new_view.new_annotation(AnnotationTypes.TimePoint)
                    prediction.annotation = timepoint_annotation
                    scores = [prediction.score_for_label(lbl) for lbl in prediction.labels]
                    classification = {l:s for l, s in zip(prediction.labels, scores)}
                    classification = self._transform(classification, bins)
                    label = max(classification, key=classification.get)
                    timepoint_annotation.add_property('timePoint', prediction.timepoint)
                    timepoint_annotation.add_property('label', label)
                    timepoint_annotation.add_property('classification', classification)
                    timepoint_annotations.append(timepoint_annotation)
                timeframe_annotation.add_property(
                    'targets', [tp.id for tp in timepoint_annotations])
                reps = [p.annotation.id for p in tf.representative_predictions()]
                timeframe_annotation.add_property("representatives", reps)
                #print(timeframe_annotation.serialize(pretty=True))

        else:

            raw_labelset = train.RAW_LABELS
            new_view.new_contain(
                AnnotationTypes.TimePoint,
                document=vd.id, timeUnit='milliseconds', labelset=raw_labelset)

            for prediction in predictions:
                timepoint_annotation = new_view.new_annotation(AnnotationTypes.TimePoint)
                prediction.annotation = timepoint_annotation
                scores = [prediction.score_for_label(lbl) for lbl in prediction.labels]
                classification = {l:s for l, s in zip(prediction.labels, scores)}
                label = max(classification, key=classification.get)
                timepoint_annotation.add_property('timePoint', prediction.timepoint)
                timepoint_annotation.add_property('label', label)
                timepoint_annotation.add_property('classification', classification)

        return mmif


    @staticmethod
    def _transform(classification: dict, bins: dict):
        """Take the raw classification and turn it into a classification of user
        labels. Also includes modeling.negative_label."""
        # TODO: this may not work when there is pre-binning
        transformed = {}
        for postlabel in bins['post'].keys():
            score = sum([classification[lbl] for lbl in bins['post'][postlabel]])
            transformed[postlabel] = score
        transformed[negative_label] = 1 - sum(transformed.values())
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
