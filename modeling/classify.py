import os
import sys
import json

import torch
import numpy as np
import cv2
from PIL import Image

from mmif.utils import video_document_helper as vdh

import train
import backbones


# For now just using the model from the first fold of a test (and also assuming
# that it has the labels as listed in config/default.yml).
BEST_MODEL = 'results-test/20231026-135541.kfold_000.pt'
#BEST_MODEL = 'results-test/20231026-164841.kfold_000.pt'

# Assuming for now that the model's feature extractor uses the VGG16 model.
MODEL_TYPE = 'vgg16'

# Mappings from prediction indices to label name. Another temporary assumption,
# it should be read from a config file or an input parameter.
LABEL_MAPPINGS = {0: 'slate', 1: 'chyron', 2: 'credit', 3: 'other'}

# Milliseconds between frames.
STEP_SIZE = 1000

# For debugging, set to True if you want to save the frames that were extracted.
SAFE_FRAMES = False

# Set to True if you want the script to be more verbose.
DRIBBLE = False


# Loading the model and featurizer.
model = train.get_net(4096, 4, 3, 0.2)
model.load_state_dict(torch.load(BEST_MODEL))
featurizer = backbones.model_map[MODEL_TYPE]()


def process_video(mp4_file: str, step: int = 1000):
    """Loops over the frames in a video and for each frame extract the features
    and apply the model. Returns a list of predictions, where each prediction is
    an instance of numpy.ndarray."""
    print(f'Processing {mp4_file}...')
    all_predictions = []
    for n, image in get_frames(mp4_file, step):
        img = Image.fromarray(image[:,:,::-1])
        features = process_frame(img, featurizer)
        prediction = model(features)
        prediction = softmax(prediction.detach().numpy())
        if DRIBBLE:
            print(f'{n:05d}', prediction)
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


def process_frame(frame_vec: np.ndarray, model: torch.nn.Sequential) -> torch.Tensor:
    """Extract the features of a single frame. Based on, but not identical to, the
    process_frame() method of the FeatureExtractor class in data_ingestion.py."""
    frame_vec = model.preprocess(frame_vec)
    frame_vec = frame_vec.unsqueeze(0)
    if torch.cuda.is_available():
        frame_vec = frame_vec.to('cuda')
        model.model.to('cuda')
    with torch.no_grad():
        feature_vec = model.model(frame_vec)
    return feature_vec.cpu()


def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


def save_predictions(predictions: list, filename: str):
    with open(filename, 'w') as fh:
        json.dump(predictions, fh)
        print(f'Saved output to {filename}')


def load_predictions(filename: str) -> list:
    with open(filename) as fh:
        predictions = json.load(fh)
        return predictions


def print_predictions(predictions):
    print('\n     bars   slate  creds  other   label')
    for n, (p1, p2, p3, p4) in enumerate(predictions):
        scores = [p1, p2, p3, p4]
        label = LABEL_MAPPINGS[scores.index(max(scores))]
        print(f'{n:03}  {p1:.4f} {p2:.4f} {p3:.4f} {p4:.4f}  {label}')
    print(f'\nTOTAL NUMBER: {len(predictions)}\n')


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


if __name__ == '__main__':

    create_frame_predictions = False
    create_timeframes = True

    if create_frame_predictions:
        predictions = process_video('data/cpb-aacip-690722078b2-shrunk.mp4', step=STEP_SIZE)
        predictions = [prediction[0].tolist() for prediction in predictions]
        save_predictions(predictions, 'predictions.json')

    if create_timeframes:
        predictions = load_predictions('predictions.json')
        print_predictions(predictions)