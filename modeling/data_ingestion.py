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
import json
import os
from collections import defaultdict
from typing import List, Union, Tuple, Dict

import av
import numpy as np
import torch
from tqdm import tqdm

import backbones


class AnnotatedImage:
    """Object representing a single frame and its metadata"""
    def __init__(self, filename: str, label: str, subtype_label: str, mod: bool = False):
        self.image = None
        self.guid, self.total_time, self.curr_time = self.split_name(filename)
        self.total_time = int(self.total_time)
        self.curr_time = int(self.curr_time)
        self.label = label
        self.subtype_label = subtype_label
        self.mod = mod

    @staticmethod
    def split_name(filename:str) -> List[str]:
        """pulls apart the filename into components
        @param: filename = filename of the format **GUID_TOTAL_CURR**
        @returns: a tuple containing all the significant metadata"""
        guid, total, curr = filename.split("_")
        curr = curr[:-4]
        return guid, total, curr


class FeatureExtractor:
    """Convert an annotated video set into a machine-readable format
    uses <model> as a backbone to featurize the annotated still images 
    into 4096-dim vectors.
    """
    models: List[backbones.ExtractorModel]

    def __init__(self, model_name: str = None):
        if model_name is None:
            self.models = [model() for model in backbones.model_map.values()]
        else:
            if model_name in backbones.model_map:
                self.models = [backbones.model_map[model_name]()]
            else:
                raise ValueError("No valid model found")
        print(f'using model(s): {[model.name for model in self.models]}')

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

    def get_stills(self, vid_path: Union[os.PathLike, str], 
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
                                         subtype_label=row[3],
                                         mod=row[4].lower() == 'true') for row in reader if row[1] == 'true']
        # mod=True should discard (taken as "unseen")
        # performace jump from additional batch (from 2nd) 
        # maybe we can throw away the video with the least (88) frames annotation from B2 to make 20/20 split on dense vs sparse annotation 

        # this part is doing the same thing as the get_stills function in getstills.py
        # (copied from https://github.com/WGBH-MLA/keystrokelabeler/blob/df4d2bc936fa3a73cdf3004803a0c35c290caf93/getstills.py#L36 )
        
        container = av.open(vid_path)
        video_stream = next((s for s in container.streams if s.type == 'video'), None)
        if video_stream is None:
            raise Exception("No video stream found in {}".format(vid_path))
        fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
        cur_target_frame = 0
        fcount = 0 
        for frame in container.decode(video=0):
            # if fcount % 10000 == 0:
            #     print(f'processing frame {fcount}')
            if cur_target_frame == len(frame_list):
                break
            ftime = int((fcount / fps) * 1000)
            if ftime == frame_list[cur_target_frame].curr_time:
                frame_list[cur_target_frame].image = frame.to_image()
                yield frame_list[cur_target_frame]
                cur_target_frame += 1
            fcount += 1


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
    parser.add_argument("-i", "--input_file",
                        help="filepath for the video to be featurized",
                        required=True)
    parser.add_argument("-c", "--csv_file",
                        help="filepath for the csv containing timepoints + labels",
                        required=True)
    parser.add_argument("-m", "--model_name",
                        type=str,
                        help="name of backbone model to use for feature extraction, when not given use all available models",
                        choices=list(backbones.model_map.keys()),
                        default=None)
    parser.add_argument("-o", "--outdir",
                        help="directory to save output files",
                        default="vectorized")
    clargs = parser.parse_args()
    main(clargs)
