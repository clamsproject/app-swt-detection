"""classify.py

This started out as an exact copy of modeling/classify.py. It was copied because I did
not want to take the effort to turn the modeling directory into a package.

It started diverging from its source very quickly.

"""

import os
import sys
import json
import yaml
import logging
import argparse
from operator import itemgetter

import torch
import numpy as np
import cv2
from PIL import Image

from mmif.utils import video_document_helper as vdh

from modeling import backbones


def get_net(in_dim, n_labels, num_layers, dropout=0.0):
    # Copied from modeling.train
    # TODO: use the one from the train module, requires creating a proper package
    dropouts = [dropout] * (num_layers - 1) if isinstance(dropout, (int, float)) else dropout
    if len(dropouts) + 1 != num_layers:
        raise ValueError("length of dropout must be equal to num_layers - 1")
    net = torch.nn.Sequential()
    for i in range(1, num_layers):
        neurons = max(128 // i, n_labels)
        net.add_module(f"fc{i}", torch.nn.Linear(in_dim, neurons))
        net.add_module(f"relu{i}", torch.nn.ReLU())
        net.add_module(f"dropout{i}", torch.nn.Dropout(p=dropouts[i - 1]))
        in_dim = neurons
    net.add_module("fc_out", torch.nn.Linear(neurons, n_labels))
    # no softmax here since we're using CE loss which includes it
    # net.add_module(Softmax(dim=1))
    return net


class Classifier:

    def __init__(self, config_file: str):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        # model and model configuration
        self.model_file = config["model_file"]
        with open(config["model_config"]) as f:
            self.model_config = yaml.safe_load(f)
        # classifier parameters
        self.time_unit = config["time_unit"]
        self.sample_rate = config["sample_rate"]
        self.minimum_frame_score = config["minimum_frame_score"]
        self.minimum_timeframe_score = config["minimum_timeframe_score"]
        self.minimum_frame_count = config["minimum_frame_count"]
        # not including the "other" label
        self.labels = tuple(self.model_config["labels"][:-1])
        # debugging parameters
        self.dribble = config.get("dribble", False)
        self.load_model()

    def load_model(self):
        self.model = get_net(
            self.model_config["in_dim"],
            len(self.model_config["labels"]),
            self.model_config["num_layers"],
            self.model_config["dropout"])
        self.model.load_state_dict(torch.load(self.model_file))
        self.model_type = self.model_config["model_type"]
        self.featurizer = backbones.model_map[self.model_type]()

    def process_video(self, mp4_file: str):
        """Loops over the frames in a video and for each frame extract the features
        and apply the model. Returns a list of predictions, where each prediction is
        an instance of numpy.ndarray."""
        print(f'Processing {mp4_file}...')
        logging.info(f'processing {mp4_file}...')
        basename = os.path.splitext(os.path.basename(mp4_file))[0]
        all_predictions = []
        for n, image in get_frames(mp4_file, self.sample_rate):
            img = Image.fromarray(image[:,:,::-1])
            features = self.extract_features(img, self.featurizer)
            prediction = self.model(features)
            prediction = Prediction(n, self.labels, prediction)
            if self.dribble:
                print(f'{n:07d}', prediction)
            all_predictions.append(prediction)
        logging.info(f'number of predictions = {len(all_predictions)}')
        return(all_predictions)

    def extract_features(self, frame_vec: np.ndarray, model: torch.nn.Sequential) -> torch.Tensor:
        """Extract the features of a single frame. Based on, but not identical to, the
        process_frame() method of the FeatureExtractor class in data_ingestion.py."""
        frame_vec = model.preprocess(frame_vec)
        frame_vec = frame_vec.unsqueeze(0)
        if torch.cuda.is_available():
            if self.dribble:
                print('CUDA is available')
            frame_vec = frame_vec.to('cuda')
            model.model.to('cuda')
        with torch.no_grad():
            feature_vec = model.model(frame_vec)
        return feature_vec.cpu()

    def extract_timeframes(self, predictions):
        timeframes = self.collect_timeframes(predictions)
        self.compress_timeframes(timeframes)
        self.filter_timeframes(timeframes)
        timeframes = self.remove_overlapping_timeframes(timeframes)
        return timeframes

    def collect_timeframes(self, predictions: list) -> dict:
        """Find sequences of frames for all labels where the score is not 0."""
        timeframes = { label: [] for label in self.labels}
        open_frames = { label: [] for label in self.labels}
        for prediction in predictions:
            timepoint = prediction.timepoint
            bins = prediction.data[-1]
            for label, score in bins:
                if score == 0:
                    if open_frames[label]:
                        timeframes[label].append(open_frames[label])
                    open_frames[label] = []
                elif score >= 1:
                    open_frames[label].append((timepoint, score, label))
        # TODO: this is fragile because it depends on a variable in the loop above
        for label, score in bins:
            if open_frames[label]:
                timeframes[label].append(open_frames[label])
        return timeframes

    def collect_timeframes(self, predictions: list) -> dict:
        """Find sequences of frames for all labels where the score is not 0."""
        timeframes = { label: [] for label in self.labels}
        open_frames = { label: [] for label in self.labels}
        for prediction in predictions:
            #print(prediction)
            timepoint = prediction.timepoint
            for i, label in enumerate(prediction.labels):
                score = prediction.data[i]
                if score < self.minimum_frame_score:
                    if open_frames[label]:
                        timeframes[label].append(open_frames[label])
                    open_frames[label] = []
                else:
                    open_frames[label].append((timepoint, score, label))
        for label in self.labels:
            if open_frames[label]:
                timeframes[label].append(open_frames[label])
        return timeframes

    def compress_timeframes(self, timeframes: dict):
        """Compresses all timeframes from [(t_1, score_1), ...  (t_n, score_n)] into the
        shorter representation (t_1, t_n, average_score)."""
        for label in self.labels:
            frames = timeframes[label]
            for i in range(len(frames)):
                start = frames[i][0][0]
                end = frames[i][-1][0]
                score = sum([e[1] for e in frames[i]]) / len(frames[i])
                frames[i] = (start, end, score)

    def filter_timeframes(self, timeframes: dict):
        """Filter out all timeframes with an average score below the threshold defined
        in MINIMUM_SCORE."""
        for label in self.labels:
            #timeframes[label] = [tf for tf in timeframes[label] if tf[2] > 0.25]
            timeframes[label] = [tf for tf in timeframes[label] if tf[2] > self.minimum_timeframe_score]

    def remove_overlapping_timeframes(self, timeframes: dict) -> list:
        all_frames = []
        for label in timeframes:
            for frame in timeframes[label]:
                all_frames.append(frame + (label,))
        all_frames = list(sorted(all_frames, key=itemgetter(2), reverse=True))
        outlawed_timepoints = set()
        final_frames = []
        for frame in all_frames:
            if self.is_included(frame, outlawed_timepoints):
                continue
            final_frames.append(frame)
            start, end, _, _ = frame
            for p in range(start, end + self.sample_rate, self.sample_rate):
                outlawed_timepoints.add(p)
        return all_frames

    def is_included(self, frame, outlawed_timepoints):
        start, end, _, _ = frame
        for i in range(start, end + self.sample_rate, self.sample_rate):
            if i in outlawed_timepoints:
                return True
        return False

    def experiment(self):
        """This is an old experiment. It was the first one that I could get to work
        and it was fully based on the code in data_ingestion.py"""
        outdir = 'vectorized2'
        featurizer = data_ingestion.FeatureExtractor('vgg16')
        in_file = 'data/cpb-aacip-690722078b2-shrunk.mp4'
        #in_file = 'data/cpb-aacip-690722078b2.mp4'
        metadata_file = 'data/cpb-aacip-690722078b2.csv'
        feat_metadata, feat_mats = featurizer.process_video(in_file, metadata_file)
        print('extraction complete')
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        for name, vectors in feat_mats.items():
            with open(f"{outdir}/{feat_metadata['guid']}.json", 'w', encoding='utf8') as f:
                json.dump(feat_metadata, f)
            np.save(f"{outdir}/{feat_metadata['guid']}.{name}", vectors)
            outputs = self.model(torch.from_numpy(vectors))
            print(outputs)


def get_frames(mp4_file: str, step: int = 1000):
    """Generator to get frames from an mp4 file. The step parameter defines the number
    of milliseconds between the frames."""
    vidcap = cv2.VideoCapture(mp4_file)
    for n in range(0, sys.maxsize, step):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, n)
        success, image = vidcap.read()
        if not success:
            break
        yield n, image


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
    fh = sys.stdout if filename is None else open(filename, 'w')
    fh.write('\n        slate  chyron creds  other\n')
    for prediction in predictions:
        milliseconds = prediction.timepoint
        p1, p2, p3, p4 = prediction.data[:4]
        fh.write(f'{milliseconds:7}  {p1:.4f} {p2:.4f} {p3:.4f} {p4:.4f}\n')
    fh.write(f'\nTOTAL PREDICTIONS: {len(predictions)}\n')


def print_timeframes(labels, timeframes):
    if timeframes:
        if type(timeframes) is dict:
            print(f'\nNumber of time frames is {sum([len(v) for v in timeframes.values()])}\n')
            for label in labels:
                for tf in timeframes[label]:
                    print(label, tf)
        elif type(timeframes) is list:
            print(f'\nNumber of time frames is {len(timeframes)}\n')
            for tf in timeframes:
                print(tf)
        
        else:
            print("\nWARNING: cannot print timeframes")
            print(timeframes)
    else:
        print(f'\nNumber of time frames is 0\n')


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
            self.data = softmax(self.tensor.detach().numpy())[0].tolist()
        else:
            self.data = data

    def __str__(self):
        label_scores = ' '.join(["%.4f" % d for d in self.data[:3]])
        other_score = self.data[len(self.labels)]
        return f'<Prediction {self.timepoint:6} {label_scores} {other_score:.4f}>'

    def score_for_label(self, label: str):
        return self.data[self.labels.index(label)]

    def as_json(self):
        return [self.timepoint, self.tensor.detach().numpy().tolist(), self.data]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    conf_help = "the YAML config file"
    pred_help = "use cached predictions"
    parser.add_argument("-i", "--input", help="Input video file")
    parser.add_argument("-c", "--config", default='example-config.yml', help=conf_help)
    parser.add_argument("--use-predictions", action='store_true', help=pred_help)
    args = parser.parse_args()

    classifier = Classifier(args.config)

    input_basename, extension = os.path.splitext(args.input)
    predictions_file = f'{input_basename}.json'
    if args.use_predictions:
        predictions = load_predictions(predictions_file, classifier.labels)
    else:
        predictions = classifier.process_video(args.input)
        save_predictions(predictions, predictions_file)
    #print_predictions(predictions, filename='predictions.txt')

    timeframes = classifier.collect_timeframes(predictions)
    
    classifier.compress_timeframes(timeframes)
    print_timeframes(classifier.labels, timeframes)

    classifier.filter_timeframes(timeframes)
    print_timeframes(classifier.labels, timeframes)

    timeframes = classifier.remove_overlapping_timeframes(timeframes)
    print_timeframes(classifier.labels, timeframes)

