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

# ============================================================================|
# Imports
import argparse
import csv
import os
import json
import numpy as np
from typing import List, Union, Tuple


import torch
from torchvision.models import vgg16, VGG16_Weights
import cv2
from PIL import Image


# ============================================================================|
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
    
# ============================================================================|
class FeatureExtractor:
    """Convert an annotated video set into a machine-readable format
    uses <model> as a backbone to featurize the annotated still images 
    into 4096-dim vectors.
    """
    
    # model = torch.hub.load('pytorch/vision', 'vgg16', weights=VGG16_Weights.IMAGENET1K_V1)
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # remove last layer
    model.classifier = model.classifier[:-1]
    preprocess = VGG16_Weights.IMAGENET1K_V1.transforms()

    # def __init__(self, **params):
        # self.model = VGG16(include_top=False, weights="imagenet", pooling="avg")

    def process_video(self, 
                      vid_path: Union[os.PathLike, str], 
                      csv_path: Union[os.PathLike, str],) -> Tuple[dict, np.ndarray]:
        """Extract the features for every annotated timepoint in a video.
        
        @param: vid_path = filename of the video
        @param: csv_path = filename of the csv containing timepoints
        @returns: A list of metadata dictionaries and associated feature matrix"""
        
        frame_metadata = {'frames': []}
        frame_vecs = []
        #get image stills
        for i, frame in enumerate(self.get_stills(vid_path, csv_path)):
            print(i)
            if 'guid' not in frame_metadata:
                frame_metadata['guid'] = frame.guid
            if 'duration' not in frame_metadata:
                frame_metadata['duration'] = frame.total_time
        
            #primary VGG Loop
            frame_vecs.append(self.process_frame(frame.image))
            frame_metadata["frames"].append(
                {k: v for k, v in frame.__dict__.items() 
                 if k != "image" and k != "guid" and k != "total_time"})

        frame_matrix = np.vstack(frame_vecs)
        return frame_metadata, frame_matrix


    def process_frame(self, frame_vec: np.ndarray) -> np.ndarray:
        """Extract the features of a single frame.
        
        @param: frame = a frame as a numpy array
        @returns: a numpy array representing the frame as <model> features"""
        frame_vec = self.preprocess(frame_vec)
        frame_vec = frame_vec.unsqueeze(0)
        if torch.cuda.is_available():
            frame_vec = frame_vec.to('cuda')
            self.model.to('cuda')
        with torch.no_grad():
            feature_vec = self.model(frame_vec)
        print(feature_vec.shape)
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

        #for each frame, move the VideoCapture and read @ frame
        for frame in frame_list:
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
#=============================================================================|
def serialize_data(metadata:dict, features: np.ndarray) -> None:
    """Serialize the dictionary and feature matrix into JSON/NP
    
    @param: metadata = a python dictionary
    @param: features = a numpy array"""
    with open(f"{metadata['guid']}.json",'w', encoding='utf8') as f:
        json.dump(metadata, f)
    np.save(metadata["guid"], features)

# ============================================================================|
def main(args):
    in_file = args.input_file
    metadata_file = args.csv_file
    featurizer = FeatureExtractor()
    print('extractor ready')
    feat_metadata, feat_matrix = featurizer.process_video(in_file, metadata_file)
    serialize_data(feat_metadata, feat_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file",
                        help ="filepath for the video to be featurized",
                        required=True)
    parser.add_argument("-c", "--csv_file",
                        help="filepath for the csv containing timepoints + labels",
                        required=True)
    clargs = parser.parse_args()
    main(clargs)
