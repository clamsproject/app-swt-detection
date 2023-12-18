"""classify.py

Stand-alone classifier script.

Run "python classify.py -h" to see how to run the script.

Note that the Classifier.configure() typically has to be executed each time you
classify frames from a video. If you don't then parameter settings from a previous
invocation may seep into the new invocation, that is, once a default configuration
setting is overwritten then the classifier instance will not revert back to the
default setting when a new video is processed.

"""

import argparse
import json
import logging
import os
import sys
from operator import itemgetter

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from modeling import train, data_loader, negative_label, stitch


DEBUG = False


class Classifier:

    def __init__(self, **config):
        self.config = config
        self.model_config = yaml.safe_load(open(config["model_config_file"]))
        # the "labels" list from the config file does not include the "negative"
        # label, but we need it here so we add it
        # TODO: do not use labels
        self.labels = self.model_config['labels'] + [negative_label]
        self.featurizer = data_loader.FeatureExtractor(
            img_enc_name=self.model_config["img_enc_name"],
            pos_enc_name=self.model_config.get("pos_enc_name", None),
            pos_enc_dim=self.model_config.get("pos_enc_dim", 0),
            max_input_length=self.model_config.get("max_input_length", 0),
            pos_unit=self.model_config.get("pos_unit", 0),
        )
        self.classifier = train.get_net(
            in_dim=self.featurizer.feature_vector_dim(),
            n_labels=len(self.model_config['bins']['pre'].keys()) + 1,
            num_layers=self.model_config["num_layers"],
            dropout=self.model_config["dropouts"],
        )
        self.classifier.load_state_dict(torch.load(config["model_file"]))
        # TODO (krim @ 12/14/23): deal with post bin
        # self.postbin = config.get("postbin", None)
        self.sample_rate = self.config["sample_rate"]
        self.debug = DEBUG

    def __str__(self):
        return (f"<Classifier "
                + f'img_enc_name="{self.model_config["img_enc_name"]}" '
                + f'pos_enc_name="{self.model_config["pos_enc_name"]}" '
                + f'sample_rate={self.sample_rate}>')

    def set_parameters(self, parameters):
        """Take the parameters from the configuration file and update them with
        parameters handed in by the app if needed. Note that the parameters from
        the app follow standard camel case while the classifier parameters are
        python variables."""
        # NOTE: this was reintroduced because the get_configuration() method in app.py
        # was broken
        self.sample_rate = self.config["sample_rate"]
        self.minimum_frame_score = self.config["minimum_frame_score"]
        self.minimum_timeframe_score = self.config["minimum_timeframe_score"]
        self.minimum_frame_count = self.config["minimum_frame_count"]
        for parameter, value in parameters.items():
            if parameter == "sampleRate":
                self.sample_rate = value
            elif parameter == "minFrameScore":
                self.minimum_timeframe_score = value
            elif parameter == "minFrameCount":
                self.minimum_frame_count = value

    def process_video(self, mp4_file: str):
        """Loops over the frames in a video and for each frame extracts the features
        and applies the classifier. Returns a list of predictions, where each prediction
        is an instance of numpy.ndarray."""
        if self.debug:
            print(f'Processing {mp4_file}...')
            print(f'Labels: {self.labels}')
        logging.info(f'processing {mp4_file}...')
        predictions = []
        vidcap = cv2.VideoCapture(mp4_file)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS), 2)
        fc = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = round(fc / fps, 3) * 1000
        for ms in range(0, sys.maxsize, self.sample_rate):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, ms)
            success, image = vidcap.read()
            if not success:
                break
            img = Image.fromarray(image[:,:,::-1])
            features = self.featurizer.get_full_feature_vectors(img, ms, dur)
            prediction = self.classifier(features).detach()
            prediction = Prediction(ms, self.labels, prediction)
            if self.debug:
                print(prediction)
            predictions.append(prediction)
        return predictions

    def pp(self):
        # debugging method
        print(f"Classifier {self.model_file}")
        print(f"   sample_rate         = {self.sample_rate}")
        print(f"   minimum_frame_score = {self.minimum_timeframe_score}")
        print(f"   minimum_frame_count = {self.minimum_frame_count}")


def save_predictions(predictions: list, filename: str):
    json_obj = []
    for prediction in predictions:
        json_obj.append(prediction.as_json())
    with open(filename, 'w') as fh:
        json.dump(json_obj, fh)
        print(f'Saved predictions to {filename}')


def load_predictions(filename: str, labels: list) -> list:
    predictions = []
    with open(filename) as fh:
        for (n, tensor, data) in json.load(fh):
            p = Prediction(n, labels, torch.Tensor(tensor), data=data)
            predictions.append(p)
    return predictions


def print_predictions(predictions, filename=None):
    # Debugging method
    fh = sys.stdout if filename is None else open(filename, 'w')
    fh.write('\n        slate  chyron creds  other\n')
    for prediction in predictions:
        milliseconds = prediction.timepoint
        p1, p2, p3, p4 = prediction.data[:4]
        fh.write(f'{milliseconds:7}  {p1:.4f} {p2:.4f} {p3:.4f} {p4:.4f}\n')
    fh.write(f'\nTOTAL PREDICTIONS: {len(predictions)}\n')


class Prediction:

    """Class to store a prediction from the model. Runs softmax over the scores in the
    tensor and can retrieve scores for all labels.

    timepoint  -  the location of the frame, in milliseconds
    labels     -  labels for the prediction, taken from the model
    tensor     -  the tensor that results from running the model on the features
    data       -  the tensor simplified into a simple list with scores for each label

    """

    def __init__(self, timepoint: int, labels: list,
                 prediction: torch.Tensor, data: list = None):
        self.timepoint = timepoint
        self.labels = labels
        self.tensor = prediction
        if data is None:
            # TODO: it seems like the dimension has to be -1 or 0. Minor changes
            # in the updated scores result, not sure which one is better. 
            self.data = torch.nn.Softmax(dim=0)(self.tensor).detach().numpy().tolist()
        else:
            self.data = data

    def __str__(self):
        label_scores = ' '.join(["%.4f" % d for d in self.data[:3]])
        neg_score = self.data[-1]
        return f'<Prediction {self.timepoint:6} {label_scores} {neg_score:.4f}>'

    def score_for_label(self, label: str):
        return self.data[self.labels.index(label)]

    def as_json(self):
        return [self.timepoint, self.tensor.detach().numpy().tolist(), self.data]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    conf_help = "the YAML config file"
    pred_help = "use cached predictions"
    parser.add_argument("--input", help="input video file")
    parser.add_argument("--config", default='modeling/config/classifier.yml', help=conf_help)
    parser.add_argument("--use-predictions", action='store_true', help=pred_help)
    parser.add_argument("--debug", action='store_true', help="turn on debugging")
    args = parser.parse_args()

    configs = yaml.safe_load(open(args.config))
    classifier = Classifier(**configs)
    stitcher = stitch.Stitcher(**configs)
    if args.debug:
        classifier.debug = True
        stitcher.debug = True
    print(classifier)
    print(stitcher)

    input_basename, extension = os.path.splitext(args.input)
    predictions_file = f'{input_basename}.json'
    if args.use_predictions:
        predictions = load_predictions(predictions_file, classifier.labels)
    else:
        predictions = classifier.process_video(args.input)
    timeframes = stitcher.create_timeframes(predictions)
