"""Scenes with Text / Frames of Interest Data Ingestion

Extracts features for "frames of interest" in video.
grabs stills at each listed timeframe, processes them into 
VGG features, and serializes the data into an output.

INPUT: a video file location and CSV file containing timepoint information+metadata
for labeled stills

OUTPUT:
 - a numpy matrix, in which each row is a (512,1) vector representing the features of a 
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


import cv2
from keras.applications.vgg16 import VGG16, preprocess_input

# ============================================================================|
class FrameOfInterest:
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
class SceneText_Featurizer:
    """Convert an annotated video set into a machine readable format
    uses <model> as a backbone to 
    """
    def __init__(self, **params):
        self.model = VGG16(include_top=False, weights="imagenet", pooling="avg")

    def process_video(self, 
                      vid_path: Union[os.PathLike, str], 
                      csv_path: Union[os.PathLike, str],) -> Tuple[dict, np.ndarray]:
        """Extract the features for every annotated timepoint in a video.
        
        @param: vid_path = filename of the video
        @param: csv_path = filename of the csv containing timepoints
        @returns: A list of metadata dictionaries and associated feature matrix"""
        frames: List[FrameOfInterest] = self.get_stills(vid_path, csv_path)
        
        frame_metadata = {"guid": frames[0].guid,
                          "duration": frames[0].total_time,
                          "frames":[]}
        
        frame_matrix = np.zeros((512, len(frames)))
        for i, frame in enumerate(frames):
            frame_matrix[:,i] = self.process_frame(frame.image)
            frame_metadata["frames"].append(
                {k:v for k, v in frame.__dict__.items() 
                 if k != "image" and k != "guid" and k != "total_time"})

        return frame_metadata, frame_matrix


    def process_frame(self, frame_vec: np.ndarray) -> np.ndarray:
        """Extract the features of a single frame.
        
        @param: frame = a frame as a numpy array
        @returns: a numpy array representing the frame as <model> features"""
        x = np.expand_dims(frame_vec, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)

    @staticmethod
    def get_stills(vid_path: Union[os.PathLike, str], 
                   csv_path: Union[os.PathLike, str]) -> List[FrameOfInterest]:
        """Extract stills at given timepoints from a video file
        
        @param: vid_path = the filename of the video
        @param: timepoints = a list of the video's annotated timepoints
        @return: a list of Frame objects"""
        frame_list = []

        with open(csv_path, encoding='utf8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                frame_list.append(FrameOfInterest(filename=row[0], label=row[2], subtype_label=row[3]))

        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        for frame in frame_list:
            frame_id = get_framenum(frame, fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, img = cap.read()
            if ret:
                frame.image = img 

        return frame_list

def get_framenum(frame: FrameOfInterest, fps: float) -> int:
    """Returns the frame number of the given frame"""
    return int(int(frame.curr_time)/1000 * fps)
#=============================================================================|
def serialize_data(metadata:dict, features: np.ndarray) -> None:
    """Serialize the dictionary and feature matrix into JSON/NP
    
    @param: metadata = a python dictionary
    @param: features = a numpy array"""
    metadata_json = json.dumps(metadata)
    with open(f"{metadata['guid']}.json",'w', encoding='utf8') as f:
        f.write(metadata_json)
    np.save(metadata["guid"], features)

# ============================================================================|
def main(args):
    in_file = args.input_file
    metadata_file = args.csv_file
    featurizer = SceneText_Featurizer()
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
