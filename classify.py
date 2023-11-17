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
import torch.nn as nn

import numpy as np
import cv2
from PIL import Image

from mmif.utils import video_document_helper as vdh

from modeling import backbones, data_ingestion
from utils import get_net


class Classifier:

    def __init__(self, configs):
        with open(configs) as f:
            config = yaml.safe_load(configs)
        self.step_size = config["step_size"]
        self.minimum_score = config["minimum_score"]
        self.score_mapping = config["score_mapping"]
        if "safe_frames" in config:
            self.safe_frames = config["safe_frames"]
        else:
            self.safe_frames = False
        if "dribble" in config:
            self.dribble = config["dribble"]
        else:
            self.dribble = False
        model_type, self.label_mappings, self.model = read_model_config(config["model_config"])
        self.model.load_state_dict(torch.load(config["model"]))
        self.featurizer = backbones.model_map[model_type]()


    def process_video(self, mp4_file: str):
        """Loops over the frames in a video and for each frame extract the features
        and apply the model. Returns a list of predictions, where each prediction is
        an instance of numpy.ndarray."""
        print(f'Processing {mp4_file}...')
        logging.info(f'processing {mp4_file}...')
        all_predictions = []
        for n, image in get_frames(mp4_file, self.step_size):
            img = Image.fromarray(image[:,:,::-1])
            features = self.extract_features(img, self.featurizer)
            prediction = self.model(features)
            prediction = Prediction(n, prediction)
            if self.dribble:
                print(f'{n:07d}', prediction)
            all_predictions.append(prediction)
            if self.safe_frames:
                cv2.imwrite(f"frames/frame-{n:06d}.jpg", image)
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
    
    def save_predictions(self, predictions: list, filename: str):
        json_obj = []
        for prediction in predictions:
            json_obj.append(prediction.as_json())
        with open(filename, 'w') as fh:
            json.dump(json_obj, fh)
            if self.dribble:
                print(f'Saved predictions to {filename}')

    def compute_labels(self, scores: list):
        return (
            ('slate', self.scale(scores[0])),
            ('chyron', self.scale(scores[1])),
            ('credit', self.scale(scores[2])))

    def scale(self, score):
        """Put the score on a scale from 0 through 4, where 0 means the score is less
        than 0.01 and 1 though 4 are quartiles for score bins 0.01-0.25, 0.25-0.50,
        0.50-0.75 and 0.75-1.00."""
        for score_in, score_out in self.score_mapping:
            if score < score_in:
                return score_out
            
    
    def enrich_predictions(self, predictions: list):
        """For each prediction, add a nominal score for each label. The scores go from
        0 through 4. For example if the raw probability score for the slate is in the
        0.5-0.75 range than ('slate', 3) will be added."""
        for prediction in predictions:
            binned_scores = self.compute_labels(prediction.data)
            prediction.data.append(binned_scores)


    def extract_timeframes(self, predictions):
        self.enrich_predictions(predictions)
        #print_predictions(predictions)
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
            timeframes[label] = [tf for tf in timeframes[label] if tf[2] > self.minimum_score]

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
            for p in range(start, end + self.step_size, self.step_size):
                outlawed_timepoints.add(p)
        return all_frames

    def is_included(self, frame, outlawed_timepoints):
        start, end, _, _ = frame
        for i in range(start, end + self.step_size, self.step_size):
            if i in outlawed_timepoints:
                return True
        return False


    def experiment(self):
        """This is an older experiment. It was the first one that I could get to work
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


def read_model_config(configs):
    with open(configs) as f:
        config = yaml.safe_load(configs)
    labels = config["labels"]
    in_dim = config["in_dim"]
    n_labels = len(labels)
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    model = get_net(in_dim, n_labels, num_layers, dropout)
    model_type = config["model_type"]
    label_mappings = {i: label for i, label in enumerate(labels)}
    return model_type, label_mappings, model


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


def load_predictions(filename: str) -> list:
    # TODO: needs to recreate the Prediction instances
    with open(filename) as fh:
        predictions = json.load(fh)
        return predictions


def print_predictions(predictions):
    print('\n        slate  chyron creds  other')
    for prediction in predictions:
        milliseconds = prediction.timepoint
        p1, p2, p3, p4 = prediction.data[:4]
        binned_scores = prediction.data[-1]
        labels = ' '.join([f'{label}-{score}' for label, score in binned_scores])
        print(f'{milliseconds:6}  {p1:.4f} {p2:.4f} {p3:.4f} {p4:.4f}  {labels}')
    print(f'\nTOTAL PREDICTIONS: {len(predictions)}\n')


class Prediction:

    """Class to store a prediction from the model. It is meant to simplify the rest
    of the code a bit and manage some of the intricacies of the data structures that
    are involved. One thing it does is to run softmax over the scores in the tensor.

    timepoint  -  the location of the frame, in milliseconds
    tensor     -  the tensor that results from running the model on the features
    data       -  the tensor simplified into a simple list with scores for each label

    """

    def __init__(self, timepoint: int, prediction: torch.Tensor):
        self.timepoint = timepoint
        self.tensor = prediction
        self.data = softmax(self.tensor.detach().numpy())[0].tolist()

    def __str__(self):
        return f'<Prediction {self.timepoint} {self.data}>'

    def as_json(self):
        return [self.timepoint, self.data]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The YAML config file")
    args = parser.parse_args()
    
    classifier = Classifier(args.config)
    predictions = classifier.process_video('modeling/data/cpb-aacip-690722078b2-shrunk.mp4')
    classifier.enrich_predictions(predictions)
    timeframes = classifier.collect_timeframes(predictions)
    classifier.compress_timeframes(timeframes)
    classifier.filter_timeframes(timeframes)
    timeframes = classifier.remove_overlapping_timeframes(timeframes)
    print(timeframes)

