"""classify.py

This started out as an exact copy of modeling/classify.py. It was copied because I did
not want to take the effort to turn the modeling directory into a package.

It started diverging from its source very quickly.

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

from modeling import train, data_loader


class Classifier:

    def __init__(self, **config):
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
        
        # stitcher config
        self.time_unit = config["time_unit"]
        self.sample_rate = config["sample_rate"]
        self.minimum_frame_score = config["minimum_frame_score"]
        self.minimum_timeframe_score = config["minimum_timeframe_score"]
        self.minimum_frame_count = config["minimum_frame_count"]

        # debugging
        self.dribble = config.get("dribble", False)

    def process_video(self, mp4_file: str):
        """Loops over the frames in a video and for each frame extract the features
        and apply the model. Returns a list of predictions, where each prediction is
        an instance of numpy.ndarray."""
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
        self.compress_timeframes(timeframes)
        self.filter_timeframes(timeframes)
        timeframes = self.remove_overlapping_timeframes(timeframes)
        return timeframes

    def collect_timeframes(self, predictions: list) -> dict:
        """Find sequences of frames for all labels where the score is not 0."""
        timeframes = {label: [] for label in self.labels}
        open_frames = {label: [] for label in self.labels}
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
        neg_score = self.data[len(self.labels)]
        return f'<Prediction {self.timepoint:6} {label_scores} {neg_score:.4f}>'

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

    classifier = Classifier(**yaml.safe_load(open(args.config)))

    input_basename, extension = os.path.splitext(args.input)
    predictions_file = f'{input_basename}.json'
    if args.use_predictions:
        predictions = load_predictions(predictions_file, classifier.labels)
    else:
        predictions = classifier.process_video(args.input)
        #save_predictions(predictions, predictions_file)
    #print_predictions(predictions, filename='predictions.txt')

    timeframes = classifier.collect_timeframes(predictions)
    
    classifier.compress_timeframes(timeframes)
    print_timeframes(classifier.labels, timeframes)

    classifier.filter_timeframes(timeframes)
    print_timeframes(classifier.labels, timeframes)

    timeframes = classifier.remove_overlapping_timeframes(timeframes)
    print_timeframes(classifier.labels, timeframes)

