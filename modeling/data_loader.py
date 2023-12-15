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
from pathlib import Path
from typing import List, Union, Tuple, Dict, ClassVar

import av
import numpy as np
import torch
from tqdm import tqdm

from modeling import backbones


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


class FeatureExtractor(object):
    
    img_encoder: backbones.ExtractorModel
    pos_encoder: str
    max_input_length: int
    pos_dim: int
    sinusoidal_embeddings: ClassVar[Dict[Tuple[int, int], torch.Tensor]] = {}

    def __init__(self, img_enc_name: str,
                 pos_enc_name: str = None,
                 pos_enc_dim: int = 512,
                 max_input_length: int = 5640000,  # 94 min = the longest video in the first round of annotation
                 pos_unit: int = 60000):
        """
        Initializes the FeatureExtractor object.
        
        @param: model_name = a name of backbone model (e.g. CNN) to use for image vector extraction
        @param: positional_encoder = type of positional encoder to use, one of 'fractional', sinusoidal-add', 'sinusoidal-concat', when not given use no positional encoding
        @param: positional_embedding_dim = dimension of positional embedding, only relevant to 'sinusoidal-add' scheme, when not given use 512
        @param: max_input_length = maximum length of input video in milliseconds, used for padding positional encoding
        @param: positional_unit = unit of positional encoding in milliseconds (e.g., 60000 for minutes, 1000 for seconds)
        """
        if img_enc_name is None:
            raise ValueError("A image vector model must be specified")
        else:
            self.img_encoder: backbones.ExtractorModel = backbones.model_map[img_enc_name]()
        self.pos_encoder = pos_enc_name
        self.pos_dim = pos_enc_dim
        if pos_enc_name in ['sinusoidal-add', 'sinusoidal-concat']:
            position_dim = int(max_input_length / pos_unit)
            if position_dim % 2 == 1:
                position_dim += 1
            if pos_enc_name == 'sinusoidal-concat':
                self.pos_vec_lookup = self.get_sinusoidal_embeddings(position_dim, pos_enc_dim)
            elif pos_enc_name == 'sinusoidal-add':
                self.pos_vec_lookup = self.get_sinusoidal_embeddings(position_dim, self.img_encoder.dim)

    def get_sinusoidal_embeddings(self, n_pos, dim):
        if (n_pos, dim) in self.__class__.sinusoidal_embeddings:
            return self.__class__.sinusoidal_embeddings[(n_pos, dim)]
        matrix = torch.zeros(n_pos, dim)
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
        matrix[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        matrix[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        self.__class__.sinusoidal_embeddings[(n_pos, dim)] = matrix
        return matrix

    def get_img_vector(self, raw_img, as_numpy=True):
        img_vec = self.img_encoder.preprocess(raw_img)
        img_vec = img_vec.unsqueeze(0)
        if torch.cuda.is_available():
            img_vec = img_vec.to('cuda')
            self.img_encoder.model.to('cuda')
        with torch.no_grad():
            feature_vec = self.img_encoder.model(img_vec)
        if as_numpy:
            return feature_vec.cpu().numpy()
        else:
            return feature_vec.cpu()
    
    def encode_position(self, cur_time, tot_time, img_vec):
        pos = cur_time / tot_time
        if isinstance(img_vec, np.ndarray):
            img_vec = torch.from_numpy(img_vec)
        img_vec = img_vec.squeeze(0)
        if self.pos_encoder is None:
            return img_vec
        elif self.pos_encoder == 'fractional':
            return torch.concat((img_vec, torch.tensor([pos])))
        elif self.pos_encoder == 'sinusoidal-add':
            return torch.add(img_vec, self.pos_vec_lookup[round(pos)])
        elif self.pos_encoder == 'sinusoidal-concat':
            return torch.concat((img_vec, self.pos_vec_lookup[round(pos)]))
    
    def feature_vector_dim(self):
        if self.pos_encoder == 'sinusoidal-add' or self.pos_encoder is None:
            return self.img_encoder.dim
        elif self.pos_encoder == 'sinusoidal-concat':
            return self.img_encoder.dim + self.pos_dim
        elif self.pos_encoder == 'fractional':
            return self.img_encoder.dim + 1
                    
    def get_full_feature_vectors(self, raw_img, cur_time, tot_time):
        img_vecs = self.get_img_vector(raw_img, as_numpy=False)
        return self.encode_position(cur_time, tot_time, img_vecs)


class TrainingDataPreprocessor(object):
    """
    Refactor of an early feature extraction code, where we only used CNN vectors
    """
    def __init__(self, model_name: str):
        if model_name is None:
            self.models = [FeatureExtractor(model_name) for model_name in backbones.model_map.keys()]
        else:
            if model_name in backbones.model_map:
                self.models = [FeatureExtractor(model_name)]
            else:
                raise ValueError("No valid model found")
        print(f'using model(s): {[model.img_encoder.name for model in self.models]}')

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
        
            for extractor in self.models:
                frame_vecs[extractor.img_encoder.name].append(extractor.get_img_vector(frame.image, as_numpy=True))
            frame_dict = {k: v for k, v in frame.__dict__.items() if k != "image" and k != "guid" and k != "total_time"}
            frame_dict['vec_idx'] = i
            frame_metadata["frames"].append(frame_dict)

        frame_mats = {k: np.vstack(v) for k, v in frame_vecs.items()}
        return frame_metadata, frame_mats

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
        # CSV rows with mod=True should be discarded (taken as "unseen")
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
            if cur_target_frame == len(frame_list):
                break
            ftime = int(frame.time * 1000) 
            if ftime == frame_list[cur_target_frame].curr_time:
                frame_list[cur_target_frame].image = frame.to_image()
                yield frame_list[cur_target_frame]
                cur_target_frame += 1
            fcount += 1


def main(args):
    in_file = args.input_file
    metadata_file = args.csv_file
    featurizer = TrainingDataPreprocessor(args.model_name)
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
                        default=Path(__file__).parent / "vectorized")
    clargs = parser.parse_args()
    main(clargs)
