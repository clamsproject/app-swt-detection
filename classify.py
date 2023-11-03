"""classify.py

This started out as an exact copy of modeling/classify.py. It was copied because I did
not want to take the effort to turn the modeling directory into a package.

It started diverging from its source very quickly.

"""

import os
import sys
import json
from operator import itemgetter

import torch
import torch.nn as nn

import numpy as np
import cv2
from PIL import Image

from mmif.utils import video_document_helper as vdh

from modeling import backbones
from utils import get_net


# For now just some random model from the first fold of a test (and also assuming
# that it has the labels as listed in config/default.yml).
MODEL = 'modeling/models/20231026-164841.kfold_000.pt'

# The above model's feature extractor uses the VGG16 model.
MODEL_TYPE = 'vgg16'

# Mappings from prediction indices to label name. Another temporary assumption,
# it should be read from a config file or an input parameter.
LABEL_MAPPINGS = {0: 'slate', 1: 'chyron', 2: 'credit', 3: 'other'}

# Milliseconds between frames.
STEP_SIZE = 1000

# Minimum average score for a timeframe. We require at least one frame score
# higher than 1.
MINIMUM_SCORE = 1.01

# For debugging, set to True if you want to save the frames that were extracted.
SAFE_FRAMES = False

# Set to True if you want the script to be more verbose.
DRIBBLE = False

# Defining the bins for the labels.
SCORE_MAPPING = ((0.01, 0), (0.25, 1), (0.50, 2), (0.75, 3), (1.01, 4))


# Getting the non-other labels.
LABELS = {label for label in sorted(LABEL_MAPPINGS.values()) if label != 'other'}


# Loading the model and featurizer.
model = get_net(4096, 4, 3, 0.2)
model.load_state_dict(torch.load(MODEL))
featurizer = backbones.model_map[MODEL_TYPE]()


def process_video(mp4_file: str, step: int = 1000):
    """Loops over the frames in a video and for each frame extract the features
    and apply the model. Returns a list of predictions, where each prediction is
    an instance of numpy.ndarray."""
    print(f'Processing {mp4_file}...')
    all_predictions = []
    for n, image in get_frames(mp4_file, step):
        img = Image.fromarray(image[:,:,::-1])
        features = extract_features(img, featurizer)
        prediction = model(features)
        prediction = Prediction(n, prediction)
        if DRIBBLE:
            print(f'{n:07d}', prediction)
        all_predictions.append(prediction)
        if SAFE_FRAMES:
            cv2.imwrite(f"frames/frame-{n:06d}.jpg", image)
    return(all_predictions)


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


def extract_features(frame_vec: np.ndarray, model: torch.nn.Sequential) -> torch.Tensor:
    """Extract the features of a single frame. Based on, but not identical to, the
    process_frame() method of the FeatureExtractor class in data_ingestion.py."""
    frame_vec = model.preprocess(frame_vec)
    frame_vec = frame_vec.unsqueeze(0)
    if torch.cuda.is_available():
        if DRIBBLE:
            print('CUDA is available')
        frame_vec = frame_vec.to('cuda')
        model.model.to('cuda')
    with torch.no_grad():
        feature_vec = model.model(frame_vec)
    return feature_vec.cpu()


def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


def save_predictions(predictions: list, filename: str):
    json_obj = []
    for prediction in predictions:
        json_obj.append(prediction.as_json())
    with open(filename, 'w') as fh:
        json.dump(json_obj, fh)
        if DRIBBLE:
            print(f'Saved predictions to {filename}')


def load_predictions(filename: str) -> list:
    with open(filename) as fh:
        predictions = json.load(fh)
        return predictions


def enrich_predictions(predictions: list):
    """For each prediction, add a nominal score for each label. The scores go from
    0 through 4. For example if the raw probability score for the slate is in the
    0.5-0.75 range than ('slate', 3) will be added."""
    for prediction in predictions:
        binned_scores = compute_labels(prediction[1])
        prediction[1].append(binned_scores)


def print_predictions(predictions):
    print('\n     slate  chyron creds  other')
    for prediction in predictions:
        milliseconds = prediction[0]
        p1, p2, p3, p4 = prediction[1][:4]
        binned_scores = prediction[1][4]
        labels = ' '.join([f'{label}-{score}' for label, score in binned_scores])
        print(f'{milliseconds:6}  {p1:.4f} {p2:.4f} {p3:.4f} {p4:.4f}  {labels}')
    print(f'\nTOTAL PREDICTIONS: {len(predictions)}\n')


def compute_labels(scores: list):
    return (
        ('slate', scale(scores[0])),
        ('chyron', scale(scores[1])),
        ('credit', scale(scores[2])))


def scale(score):
    """Put the score on a scale from 0 through 4, where 0 means the score is less
    than 0.01 and 1 though 4 are quartiles for score bins 0.01-0.25, 0.25-0.50,
    0.50-0.75 and 0.75-1.00."""
    for score_in, score_out in SCORE_MAPPING:
        if score < score_in:
            return score_out


def collect_timeframes(predictions: list) -> dict:
    """Find sequences of frames for all labels where the score is not 0."""
    timeframes = { label: [] for label in LABELS}
    open_frames = { label: [] for label in LABELS}
    for prediction in predictions:
        timepoint = prediction[0]
        bins = prediction[1][4]
        for label, score in bins:
            if score == 0:
                if open_frames[label]:
                    timeframes[label].append(open_frames[label])
                open_frames[label] = []
            elif score >= 1:
                open_frames[label].append((timepoint, score, label))
    for label, score in bins:
        if open_frames[label]:
            timeframes[label].append(open_frames[label])
    return timeframes

def compress_timeframes(timeframes: dict):
    """Compresses all timeframes from [(t_1, score_1), ...  (t_n, score_n)] into the
    shorter representation (t_1, t_n, average_score)."""
    for label in LABELS:
        frames = timeframes[label]
        for i in range(len(frames)):
            start = frames[i][0][0]
            end = frames[i][-1][0]
            score = sum([e[1] for e in frames[i]]) / len(frames[i])
            frames[i] = (start, end, score)

def filter_timeframes(timeframes: dict):
    """Filter out all timeframes with an average score below the threshold defined
    in MINIMUM_SCORE."""
    for label in LABELS:
        timeframes[label] = [tf for tf in timeframes[label] if tf[2] > MINIMUM_SCORE]

def remove_overlapping_timeframes(timeframes: dict) -> list:
    all_frames = []
    for label in timeframes:
        for frame in timeframes[label]:
            all_frames.append(frame + (label,))
    all_frames = list(sorted(all_frames, key=itemgetter(2), reverse=True))
    outlawed_timepoints = set()
    final_frames = []
    for frame in all_frames:
        if is_included(frame, outlawed_timepoints):
            continue
        final_frames.append(frame)
        start, end, _, _ = frame
        for p in range(start, end + STEP_SIZE, STEP_SIZE):
            outlawed_timepoints.add(p)
    return all_frames

def is_included(frame, outlawed_timepoints):
    start, end, _, _ = frame
    for i in range(start, end + STEP_SIZE, STEP_SIZE):
        if i in outlawed_timepoints:
            return True
    return False


def experiment():
    """This is an older experiment. It was the first one that I could get to work
    and it was fully based on the code in data_ingestion.py"""
    outdir = 'vectorized2'
    featurizer = FeatureExtractor('vgg16')
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
        outputs = model(torch.from_numpy(vectors))
        print(outputs)


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

    create_frame_predictions = False
    create_timeframes = False

    if create_frame_predictions:
        predictions = process_video('data/cpb-aacip-690722078b2-shrunk.mp4', step=STEP_SIZE)
        save_predictions(predictions, 'predictions.json')

    if create_timeframes:
        predictions = load_predictions('predictions.json')
        enrich_predictions(predictions)
        #print_predictions(predictions)
        timeframes = collect_timeframes(predictions)
        compress_timeframes(timeframes)
        filter_timeframes(timeframes)
        #for label in timeframes:
        #    print(label, timeframes[label])
        timeframes = remove_overlapping_timeframes(timeframes)
        print(timeframes)


