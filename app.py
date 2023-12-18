"""

CLAMS app to detect scenes with text.

The kinds of scenes that are recognized depend on the model used but typically
include slates, chryons and credits.

"""

import argparse
import logging
from typing import Union

import yaml
from clams import ClamsApp, Restifier
from mmif import Mmif, View, AnnotationTypes, DocumentTypes

from modeling import classify, stitch

logging.basicConfig(filename='swt.log', level=logging.DEBUG)


class SwtDetection(ClamsApp):

    def __init__(self, configs):
        super().__init__()
        self.classifier = classify.Classifier(**configs)
        self.stitcher = stitch.Stitcher(**configs)

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._annotate

        vds = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not vds:
            # TODO: should add warning
            return mmif
        vd = vds[0]

        # add the timeframes to a new view and return the updated Mmif object
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)

        # NOTE: commented out for now because it broke the app and I reintroduced
        # the previous way of setting parameters.
        # parameters = self.get_configuration(parameters)

        # calculate the frame predictions and extract the timeframes
        # use `parameters` as needed as runtime configuration
        self.classifier.set_parameters(parameters)
        predictions = self.classifier.process_video(vd.location)
        timeframes = self.stitcher.create_timeframes(predictions)

        new_view.new_contain(AnnotationTypes.TimeFrame, document=vd.id)
        for tf in timeframes:
            timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
            timeframe_annotation.add_property("start", tf.start)
            timeframe_annotation.add_property("end", tf.end)
            timeframe_annotation.add_property("frameType", tf.label),
            timeframe_annotation.add_property("score", tf.score)

        return mmif


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The YAML config file", default='modeling/config/classifier.yaml')
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()
    classifier_configs = yaml.safe_load(open(parsed_args.config))

    app = SwtDetection(classifier_configs)

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
