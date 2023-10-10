"""Scenes with Text / Frames of Interest Data Ingestion

Extracts features for "frames of interest" in video.
grabs stills at each listed timeframe, processes them into 
VGG features, and serializes the data into an output.

INPUT: a video file location and CSV file containing timepoint information+metadata
for labeled stills

OUTPUT:
 - a numpy matrix, in which each row is a (4096,1) vector representing the features of a 
 particularframe of interest.

 - a dictionary, serialized into JSON, representing the following metadata of each still:
    - timestamp
    - label 
    - subtype label
"""

import argparse
import csv
import os
import json
import numpy as np
from typing import List, Union, Tuple, Callable, Dict
from collections import defaultdict

from tqdm import tqdm

import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from PIL import Image


class AnnotatedImage:
    """Object representing a single frame and its metadata"""
    def __init__(self, filename: str, label: str, subtype_label: str):
        self.image = None
        self.guid, self.total_time, self.curr_time = self.split_name(filename)
        self.label = label
        self.subtype_label = subtype_label

    @staticmethod
    def split_name(filename:str) -> List[str]:
        """pulls apart the filename into components
        @param: filename = filename of the format **GUID_TOTAL_CURR**
        @returns: a tuple containing all the significant metadata"""
        guid, total, curr = filename.split("_")
        curr = curr[:-4]
        return guid, total, curr
    

class ExtractorModel:
    name: str
    model: torch.nn.Module
    preprocess: Callable


class Vgg16Extractor(ExtractorModel):
    def __init__(self):
        self.name = "vgg16"
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier = self.model.classifier[:-1]
        self.preprocess = VGG16_Weights.IMAGENET1K_V1.transforms()
        
        
class Resnet50Extractor(ExtractorModel):
    def __init__(self):
        self.name = "resnet50"
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()


class FeatureExtractor:
    """Convert an annotated video set into a machine-readable format
    uses <model> as a backbone to featurize the annotated still images 
    into 4096-dim vectors.
    """
    models: List[ExtractorModel]

    def __init__(self, model_name: str = None):
        if model_name is None:
            self.models = [Vgg16Extractor(), Resnet50Extractor()]
        elif model_name == "vgg16":
            self.models = [Vgg16Extractor()]
        elif model_name == "resnet50":
            self.models = [Resnet50Extractor()]
        else:
            raise ValueError("No valid model found")

    def process_video(self, 
                      vid_path: Union[os.PathLike, str], 
                      csv_path: Union[os.PathLike, str],) -> Tuple[Dict, Dict[str, np.ndarray]]:
        """Extract the features for every annotated timepoint in a video.
        
        @param: vid_path = filename of the video
        @param: csv_path = filename of the csv containing timepoints
        @returns: A list of metadata dictionaries and associated feature matrix"""
        
        frame_metadata = {'frames': []}
        frame_vecs = defaultdict(list)
        print(f'processing video: {vid_path}')
        for i, frame in tqdm(enumerate(self.get_stills(vid_path, csv_path))):
            if 'guid' not in frame_metadata:
                frame_metadata['guid'] = frame.guid
            if 'duration' not in frame_metadata:
                frame_metadata['duration'] = frame.total_time
        
            for model in self.models:
                frame_vecs[model.name].append(self.process_frame(frame.image, model))
            frame_dict = {k: v for k, v in frame.__dict__.items() if k != "image" and k != "guid" and k != "total_time"}
            frame_dict['vec_idx'] = i
            frame_metadata["frames"].append(frame_dict)

        frame_mats = {k: np.vstack(v) for k, v in frame_vecs.items()}
        return frame_metadata, frame_mats

    def process_frame(self, frame_vec: np.ndarray, model) -> np.ndarray:
        """Extract the features of a single frame.
        
        @param: frame = a frame as a numpy array
        @returns: a numpy array representing the frame as <model> features"""
        frame_vec = model.preprocess(frame_vec)
        frame_vec = frame_vec.unsqueeze(0)
        if torch.cuda.is_available():
            frame_vec = frame_vec.to('cuda')
            model.model.to('cuda')
        with torch.no_grad():
            feature_vec = model.model(frame_vec)
        return feature_vec.cpu().numpy()

    @staticmethod
    def get_stills(vid_path: Union[os.PathLike, str], 
                   csv_path: Union[os.PathLike, str]) -> List[AnnotatedImage]:
        """Extract stills at given timepoints from a video file
        
        @param: vid_path = the filename of the video
        @param: timepoints = a list of the video's annotated timepoints
        @return: a list of Frame objects"""

        with open(csv_path, encoding='utf8') as f:
            reader = csv.reader(f)
            next(reader)
            frame_list = [AnnotatedImage(filename=row[0],
                                         label=row[2],
                                         subtype_label=row[3]) for row in reader]

        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # for each frame, move the VideoCapture and read @ frame
        # i = 0
        for frame in frame_list:
            # if i > 10:
            #     break
            # i += 1
            frame_id = get_framenum(frame, fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, img = cap.read()
            if ret:
                frame.image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            yield frame


def get_framenum(frame: AnnotatedImage, fps: float) -> int:
    """Returns the frame number of the given FrameOfInterest
    (converts from ms to frame#)"""
    return int(int(frame.curr_time)/1000 * fps)


def main(args):
    in_file = args.input_file
    metadata_file = args.csv_file
    featurizer = FeatureExtractor(args.model_name)
    print('extractor ready')
    feat_metadata, feat_mats = featurizer.process_video(in_file, metadata_file)
    print('extraction complete')
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/{feat_metadata['guid']}.json", 'w', encoding='utf8') as f:
        json.dump(feat_metadata, f)
    for name, vectors in feat_mats.items():
        np.save(f"{args.outdir}/{feat_metadata['guid']}.{name}", vectors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file",
                        help="filepath for the video to be featurized",
                        required=True)
    parser.add_argument("-c", "--csv_file",
                        help="filepath for the csv containing timepoints + labels",
                        required=True)
    parser.add_argument("-m", "--model_name",
                        type=str,
                        help="name of backbone model to use for feature extraction",
                        choices=["resnet50", "vgg16"],
                        default=None)
    parser.add_argument("-o", "--outdir",
                        help="directory to save output files",
                        default="vectorized")
    clargs = parser.parse_args()
    main(clargs)
