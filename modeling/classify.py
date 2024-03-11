"""Classifier module.

Used by app.py in the parent directory.

See app.py for hints on how to uses this, the main workhorse method is process_video(),
which takes a video and returns a list of predictions from the image classification model.

For debugging you can run this a standalone script from the parent directory:

$ python -m modeling.classify \
    --config modeling/config/classifier.yml \
    --input MP4_FILE \
    --debug

The above will also use the stitcher in stitch.py.

For help on parameters use:

$ python -m modeling.classify -h

The requirements are the same as the requirements for ../app.py.

"""

import time
import argparse
import json
import logging
import os
import sys
from typing import List

import cv2
import torch
import yaml
from PIL import Image

from modeling import train, data_loader, stitch


class Classifier:

    def __init__(self, **config):
        self.config = config
        self.model_config = yaml.safe_load(open(config["model_config_file"]))
        self.prebin_labels = train.pre_bin_label_names(self.model_config, train.RAW_LABELS)
        self.postbin_labels = train.post_bin_label_names(self.model_config)
        self.featurizer = data_loader.FeatureExtractor(
            img_enc_name=self.model_config["img_enc_name"],
            pos_enc_name=self.model_config.get("pos_enc_name", None),
            pos_enc_dim=self.model_config.get("pos_enc_dim", 0),
            max_input_length=self.model_config.get("max_input_length", 0),
            pos_unit=self.model_config.get("pos_unit", 0))
        label_count = train.RAW_LABEL_COUNT
        if 'pre' in self.model_config['bins']:
            label_count = len(self.model_config['bins']['pre'].keys()) + 1
        self.classifier = train.get_net(
            in_dim=self.featurizer.feature_vector_dim(),
            n_labels=label_count,
            num_layers=self.model_config["num_layers"],
            dropout=self.model_config["dropouts"])
        self.classifier.load_state_dict(torch.load(config["model_file"]))
        self.sample_rate = self.config.get("sampleRate")
        self.start_at = config.get("startAt", 0)
        self.stop_at = config.get('stopAt', sys.maxsize)
        self.debug = False
        self.logger = logging.getLogger(config.get('logger_name', self.__class__.__name__))

    def __str__(self):
        return (f"<Classifier "
                + f'img_enc_name="{self.model_config["img_enc_name"]}" '
                + f'pos_enc_name="{self.model_config["pos_enc_name"]}" '
                + f'sample_rate={self.sample_rate}>')

    def process_video(self, vidcap: cv2.VideoCapture) -> list:
        """
        Image classification for a video without MMIF SDK helpers, for standalone mode without MMIF/CLAMS involved
        Loops over the frames in a video and for each frame extracts the features
        and applies the classifier. Returns a list of predictions, where each prediction
        is an instance of numpy.ndarray.
        """
        featurizing_time = 0
        classifier_time = 0
        extract_time = 0
        seek_time = 0
        if self.debug:
            print(f'Labels: {self.prebin_labels}')
        predictions = []
        fps = round(vidcap.get(cv2.CAP_PROP_FPS), 2)
        fc = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = round(fc / fps, 3) * 1000
        for ms in range(0, sys.maxsize, self.sample_rate):
            if ms < self.start_at:
                continue
            if ms > self.stop_at:
                break
            t = time.perf_counter()
            vidcap.set(cv2.CAP_PROP_POS_MSEC, ms)
            if self.logger.isEnabledFor(logging.DEBUG):
                seek_time += time.perf_counter() - t
            t = time.perf_counter()
            success, image = vidcap.read()
            if self.logger.isEnabledFor(logging.DEBUG):
                extract_time += time.perf_counter() - t
            if not success:
                break
            img = Image.fromarray(image[:, :, ::-1])
            t = time.perf_counter()
            features = self.featurizer.get_full_feature_vectors(img, ms, dur)
            if self.logger.isEnabledFor(logging.DEBUG):
                featurizing_time += time.perf_counter() - t
            t = time.perf_counter()
            prediction = self.classifier(features).detach()
            prediction = Prediction(ms, self.prebin_labels, prediction)
            if self.logger.isEnabledFor(logging.DEBUG):
                classifier_time += time.perf_counter() - t
            if self.debug:
                print(prediction)
            predictions.append(prediction)
        self.logger.debug(f'Featurizing time: {featurizing_time:.2f} seconds\n')
        self.logger.debug(f'Classifier time: {classifier_time:.2f} seconds\n')
        self.logger.debug(f'Extract time: {extract_time:.2f} seconds\n')
        self.logger.debug(f'Seeking time: {seek_time:.2f} seconds\n')
        return predictions

    def classify_images(self, images: List[Image.Image], positions: List[int], final_pos: int) -> list:
        """
        Image classification for a set of extract images (in PIL.Image format). 
        Useful with using ``mmif.utils.video_document_handler.extract_frames_as_images()``
        """
        predictions = []
        featurizing_time = 0
        classifier_time = 0
        for pos, img in zip(positions, images):
            t = time.perf_counter()
            features = self.featurizer.get_full_feature_vectors(img, pos, final_pos)
            if self.logger.isEnabledFor(logging.DEBUG):
                featurizing_time += time.perf_counter() - t
            t = time.perf_counter()
            prediction = self.classifier(features).detach()
            prediction = Prediction(pos, self.prebin_labels, prediction)
            if self.logger.isEnabledFor(logging.DEBUG):
                classifier_time += time.perf_counter() - t
            predictions.append(prediction)
        self.logger.debug(f'Featurizing time: {featurizing_time:.2f} seconds\n')
        self.logger.debug(f'Classifier time: {classifier_time:.2f} seconds\n')
        return predictions

    def pp(self):
        # debugging method
        print(f"Classifier {self.model_file}")
        print(f"   sample_rate     = {self.sample_rate}")
        print(f"   min_frame_score = {self.min_timeframe_score}")
        print(f"   min_frame_count = {self.min_frame_count}")


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

    """Class to store a prediction from the model. The prediction is stored as a
    Torch tensor, but also as a list with softmaxed values. Softmaxed scores can
    be retrieve using the label.

    timepoint   -  the location of the frame in the video, in milliseconds
    labels      -  labels for the prediction, taken from the classifier model
    tensor      -  the tensor that results from running the model on the features
    data        -  the tensor simplified into a simple list with softmax scores
    annotation  -  a MMIF annotation associated with the prediction

    Note that instances of this class know nothing about binning. The labels that
    they get handed in are the pre-binning labels and those are the ones that the
    classifier calculates scores for. To get post-binning score (scores where the 
    pre-binning scores are summed) use the score_for_labels() method and let the
    caller figure out what labels are post-binned. The annotation variable is not
    used unless this code is embedded in a CLAMS App.

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
        self.annotation = None

    def __str__(self):
        # TODO: this needs to be generalized to any set of labels
        label_scores = ' '.join(["%.4f" % d for d in self.data[:-1]])
        neg_score = self.data[-1]
        return f'<Prediction {self.timepoint:6} {label_scores} {neg_score:.4f}>'

    def score_for_label(self, label: str):
        """Return the score for a label."""
        return self.data[self.labels.index(label)]

    def score_for_labels(self, labels: list):
        """Return the  score for a list of labels. This is used when the SWT app
        uses postbinning. The score of a list of labels is defined to be the score
        for the highest scoring label."""
        scores = [self.score_for_label(label) for label in labels]
        return max(scores)

    def as_json(self):
        return [self.timepoint, self.tensor.detach().numpy().tolist(), self.data]


def parse_args():
    parser = argparse.ArgumentParser()
    default_config = 'modeling/config/classifier.yml'
    conf_help = "the YAML config file"
    pred1_help = "use saved predictions"
    pred2_help = "save predictions"
    parser.add_argument("--input", help="input video file")
    parser.add_argument("--config", default=default_config, help=conf_help)
    parser.add_argument("--start", default=0, help="start N milliseconds into the video")
    parser.add_argument("--stop", default=None, help="stop N milliseconds into the video")
    parser.add_argument("--use-predictions", action='store_true', help=pred1_help)
    parser.add_argument("--save-predictions", action='store_true', help=pred2_help)
    parser.add_argument("--debug", action='store_true', help="turn on debugging")
    return parser.parse_args()


def add_parameters(args: argparse.Namespace, classifier: Classifier, stitcher: stitch.Stitcher):
    """Add arguments to the classifier and the stitcher."""
    if args.debug:
        classifier.debug = True
        stitcher.debug = True
    if args.start:
        classifier.start_at = int(args.start)
    if args.stop:
        classifier.stop_at = int(args.stop)


def open_mp4_file(mp4_file, verbose=False):
    if verbose:
        print(f'Processing {args.input}...')
    mp4_vidcap = cv2.VideoCapture(mp4_file)
    if not mp4_vidcap.isOpened():
        raise IOError(f'Could not open {mp4_file}')
    return mp4_vidcap


if __name__ == '__main__':

    args = parse_args()
    configs = yaml.safe_load(open(args.config))
    classifier = Classifier(**configs)
    stitcher = stitch.Stitcher(**configs)
    add_parameters(args, classifier, stitcher)

    if args.debug:
        print(classifier)
        print(stitcher)
    mp4_vidcap = open_mp4_file(args.input, args.debug)
    if not mp4_vidcap.isOpened():
        raise IOError(f'Could not open {args.input}')

    input_basename, extension = os.path.splitext(args.input)
    predictions_file = f'{input_basename}.json'
    if args.use_predictions:
        predictions = load_predictions(predictions_file, classifier.prebin_labels)
    else:
        predictions = classifier.process_video(mp4_vidcap)
        if args.save_predictions:
            save_predictions(predictions, predictions_file)

    timeframes = stitcher.create_timeframes(predictions)

    if not args.debug:
        for timeframe in timeframes:
            print(timeframe)
