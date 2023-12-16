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

from modeling import train, data_loader, negative_label


class Classifier:

    def __init__(self, **config):
        self.config = config
        model_config = yaml.safe_load(open(config["model_config_file"]))
        # the "labels" list from the config file should not include "negative" label from the beginning
        self.labels = train.get_final_label_names(model_config)
        self.featurizer = data_loader.FeatureExtractor(
            img_enc_name=model_config["img_enc_name"],
            pos_enc_name=model_config.get("pos_enc_name", None),
            pos_enc_dim=model_config.get("pos_enc_dim", 0),
            max_input_length=model_config.get("max_input_length", 0),
            pos_unit=model_config.get("pos_unit", 0),
        )
        self.classifier = train.get_net(
            in_dim=self.featurizer.feature_vector_dim(),
            n_labels=len(model_config['bins']['pre'].keys()) + 1,
            num_layers=model_config["num_layers"],
            dropout=model_config["dropouts"],
        )
        self.classifier.load_state_dict(torch.load(config["model_file"]))
        # TODO (krim @ 12/14/23): deal with post bin
        # self.postbin = config.get("postbin", None)

        # configuration settings from the config file
        self.sample_rate = self.config["sample_rate"]
        self.minimum_frame_score = self.config["minimum_frame_score"]
        self.minimum_timeframe_score = self.config["minimum_timeframe_score"]
        self.minimum_frame_count = self.config["minimum_frame_count"]
        # debugging
        self.dribble = False

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
        if self.dribble:
            print(f'Processing {mp4_file}...')
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
            if self.dribble:
                print(prediction)
            predictions.append(prediction)
        return predictions

    def extract_timeframes(self, predictions):
        timeframes = self.collect_timeframes(predictions)
        if self.dribble:
            print_timeframes("Potential timeframes", timeframes)
        timeframes = self.filter_timeframes(timeframes)
        timeframes = self.remove_overlapping_timeframes(timeframes)
        if self.dribble:
            print_timeframes("Selected timeframes", timeframes)
        return timeframes

    def collect_timeframes(self, predictions: list) -> dict:
        """Find sequences of frames for all labels where the score of each frame
        is at least the mininum value as defined in self.minimum_frame_score."""
        timeframes = []
        open_frames = { label: TimeFrame(label) for label in self.labels}
        for prediction in predictions:
            for i, label in enumerate(prediction.labels):
                if label == negative_label:
                    continue
                score = prediction.data[i]
                if score < self.minimum_frame_score:
                    if open_frames[label]:
                        timeframes.append(open_frames[label])
                    open_frames[label] = TimeFrame(label)
                else:
                    open_frames[label].add_point(prediction.timepoint, score)
        for label in self.labels:
            if open_frames[label]:
                timeframes.append(open_frames[label])
        for tf in timeframes:
            tf.finish()
        return timeframes

    def filter_timeframes(self, timeframes: list) -> list:
        """Filter out all timeframes with an average score below the threshold defined
        in the configuration settings."""
        # TODO: this now also uses the minimum number of samples, but maybe do this
        # filtering later in case we want to use short competing timeframes as a way
        # to determine whther another frame is viable
        return [tf for tf in timeframes
                if (tf.score > self.minimum_timeframe_score
                    and len(tf) >= self.minimum_frame_count)]

    def remove_overlapping_timeframes(self, timeframes: list) -> list:
        all_frames = list(sorted(timeframes, key=lambda tf: tf.score, reverse=True))
        outlawed_timepoints = set()
        final_frames = []
        for frame in all_frames:
            if self.is_included(frame, outlawed_timepoints):
                continue
            final_frames.append(frame)
            for p in range(frame.start, frame.end + self.sample_rate, self.sample_rate):
                outlawed_timepoints.add(p)
        return final_frames

    def is_included(self, frame, outlawed_timepoints: set):
        #start, end, _, _ = frame
        for i in range(frame.start, frame.end + self.sample_rate, self.sample_rate):
            if i in outlawed_timepoints:
                return True
        return False

    def pp(self):
        # debugging method
        print(f"Classifier {self.model_file}")
        print(f"   sample_rate         = {self.sample_rate}")
        print(f"   minimum_frame_score = {self.minimum_timeframe_score}")
        print(f"   minimum_frame_count = {self.minimum_frame_count}")


def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


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


def print_timeframes(header, timeframes: list):
    print(f'\n{header} ({len(timeframes)})')
    for tf in sorted(timeframes, key=lambda tf: tf.start):
        print(tf)


class TimeFrame:

    def __init__(self, label: str):
        self.label = label
        self.points = []
        self.scores = []
        self.start = None
        self.end = None
        self.score = None

    def __len__(self):
        return len(self.points)

    def __nonzero__(self):
        return len(self) != 0

    def __str__(self):
        if self.is_empty():
            return "<TimePoint empty>"
        else:
            return f"<TimeFrame {self.label} {self.points[0]}:{self.points[-1]} score={self.score:0.4f}>"

    def add_point(self, point, score):
        self.points.append(point)
        self.scores.append(score)

    def finish(self):
        """Once all points have been added to a timeframe, use this method to
        calculate the timeframe score from the points and to set start and end."""
        self.score = sum(self.scores) / len(self)
        self.start = self.points[0]
        self.end = self.points[-1]

    def is_empty(self):
        return len(self) == 0


class Prediction:

    """Class to store a prediction from the model. It is meant to simplify the rest
    of the code a bit and manage some of the intricacies of the data structures that
    are involved. One thing it does is to run softmax over the scores in the tensor.

    timepoint  -  the location of the frame, in milliseconds
    tensor     -  the tensor that results from running the model on the features
    data       -  the tensor simplified into a simple list with scores for each label

    """

    def __init__(self, timepoint: int, labels: list,
                 prediction: torch.Tensor, data: list = None):
        self.timepoint = timepoint
        self.labels = labels
        self.tensor = prediction
        if data is None:
            # TODO: probably use torch.nn.Softmax()
            self.data = softmax(self.tensor.detach().numpy()).tolist()
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
    parser.add_argument("-i", "--input", help="input video file")
    parser.add_argument("-c", "--config", default='modeling/config/classifier.yml', help=conf_help)
    parser.add_argument("--use-predictions", action='store_true', help=pred_help)
    parser.add_argument("--debug", action='store_true', help="turn on debugging")
    args = parser.parse_args()

    classifier = Classifier(**yaml.safe_load(open(args.config)))
    if args.debug:
        classifier.dribble = True

    input_basename, extension = os.path.splitext(args.input)
    predictions_file = f'{input_basename}.json'
    if args.use_predictions:
        predictions = load_predictions(predictions_file, classifier.labels)
    else:
        predictions = classifier.process_video(args.input)
        #save_predictions(predictions, predictions_file)
    #print_predictions(predictions, filename='predictions.txt')

    timeframes = classifier.collect_timeframes(predictions)
    print_timeframes('Collected frames', timeframes)
    
    classifier.filter_timeframes(timeframes)
    print_timeframes('Filtered frames', timeframes)

    timeframes = classifier.remove_overlapping_timeframes(timeframes)
    print_timeframes('Final frames', timeframes)

