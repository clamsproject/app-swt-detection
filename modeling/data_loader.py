import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Tuple, Dict, ClassVar

import av
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from modeling import backbones

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)-8s %(thread)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnnotatedImage:
    """
    Object representing a single frame and its metadata
    """
    
    def __init__(self, filename: str, label: str, subtype_label: str, mod: bool = False):
        self.image = None
        self.filename = filename
        self.guid, self.total_time, self.curr_time = self.split_name(filename)
        self.total_time = int(self.total_time)
        self.curr_time = int(self.curr_time)
        self.label = label
        self.subtype_label = subtype_label
        self.mod = mod

    @staticmethod
    def split_name(filename:str) -> List[str]:
        """
        pulls apart the filename into components
        
        :param filename: filename of the format **GUID_TOTAL_CURR**
        :return: a tuple containing all the significant metadata
        """

        split_string = filename.split("_")
        if len(split_string) == 3:
            guid, total, curr = split_string
        elif len(split_string) == 4:
            guid, total, sought, curr = split_string
        curr = curr[:-4]
        return guid, total, curr


class FeatureExtractor(object):
    """
    Main class that provides feature extraction functionality for a given image. Will create two vectors, one from a 
    pretrained CNN model and another from a positional encoding matrix. And then returns a single feature vector based
    on a weighed sum of the two.
    """
    img_encoder: backbones.ExtractorModel
    pos_length: int
    pos_unit: int
    pos_abs_th_front: int
    pos_abs_th_end: int
    pos_vec_coeff: float
    sinusoidal_embeddings: ClassVar[Dict[Tuple[int, int], torch.Tensor]] = {}

    def __init__(self, 
                 img_enc_name: str,
                 pos_length: int = 6000000,
                 pos_unit: int = 60000,
                 pos_abs_th_front: int = 3,
                 pos_abs_th_end: int = 10,
                 pos_vec_coeff: float = 0.5, 
                 **kwargs):  # to catch unexpected arguments
        """
        Initializes the FeatureExtractor object.

        :param img_enc_name: a name of backbone model (e.g. CNN) to use for image vector extraction
        :param pos_length: "width" of positional encoding matrix, actual number of matrix columns is calculated by 
                             pos_length / pos_unit (with default values, that is 100 minutes)
        :param pos_unit: unit of positional encoding in milliseconds (e.g., 60000 for minutes, 1000 for seconds)
        :param pos_abs_th_front: the number of "units" to perform absolute lookup at the front of the video
        :param pos_abs_th_end: the number of "units" to perform absolute lookup at the end of the video
        :param pos_vec_coeff: a value used to regularize the impact of positional encoding
        """
        if img_enc_name is None:
            raise ValueError("A image vector model must be specified")
        else:
            self.img_encoder: backbones.ExtractorModel = backbones.model_map[img_enc_name]()
        self.pos_unit = pos_unit
        self.pos_abs_th_front = pos_abs_th_front
        self.pos_abs_th_end = pos_abs_th_end
        self.pos_vec_coeff = pos_vec_coeff
        position_dim = int(pos_length / self.pos_unit)
        if position_dim % 2 == 1:
            position_dim += 1
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

    def convert_position(self, cur, tot):
        if cur < self.pos_abs_th_front or tot - cur < self.pos_abs_th_end:
            return cur
        else:
            return cur * self.pos_vec_lookup.shape[0] // tot

    def encode_position(self, cur_time, tot_time, img_vec):
        if isinstance(img_vec, np.ndarray):
            img_vec = torch.from_numpy(img_vec)
        img_vec = img_vec.squeeze(0)
        pos_lookup_col = self.convert_position(cur_time, tot_time)
        pos_vec = self.pos_vec_lookup[pos_lookup_col] * self.pos_vec_coeff
        return torch.add(img_vec, pos_vec)

    def feature_vector_dim(self):
        return self.img_encoder.dim

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
        logger.info(f'using model(s): {[model.img_encoder.name for model in self.models]}')

    def process_input(self,
                      input_path: Union[os.PathLike, str],
                      csv_path: Union[os.PathLike, str]):
        """
        Extract the features for every annotated timepoint in a video.

        :param input_path: filename of the input
        :param csv_path: csv file containing timepoint-wise annotations
        """
        if Path(input_path).is_dir():
            logger.info(f'processing dictionary: {input_path}')
        else:
            logger.info(f'processing video: {input_path}')

        frame_metadata = {'frames': []}
        frame_vecs = defaultdict(list)
        for frame in tqdm(self.get_stills(input_path, csv_path)):
            if 'guid' in frame_metadata and frame.guid != frame_metadata['guid']:
                frame_mats = {k: np.vstack(v) for k, v in frame_vecs.items()}
                yield frame_metadata, frame_mats
                frame_metadata = {'frames': []}
                frame_vecs = defaultdict(list)
            if 'guid' not in frame_metadata:
                frame_metadata['guid'] = frame.guid
            if 'duration' not in frame_metadata:
                frame_metadata['duration'] = frame.total_time
            for extractor in self.models:
                frame_vecs[extractor.img_encoder.name].append(extractor.get_img_vector(frame.image, as_numpy=True))
            frame_dict = {k: v for k, v in frame.__dict__.items() if k != "image" and k != "guid" and k != "total_time" and k != "filename"}
            frame_dict['vec_idx'] = len(frame_metadata['frames'])
            frame_metadata["frames"].append(frame_dict)
        frame_mats = {k: np.vstack(v) for k, v in frame_vecs.items()}
        yield frame_metadata, frame_mats

    def get_stills(self, media_path: Union[os.PathLike, str],
                   csv_path: Union[os.PathLike, str]) -> List[AnnotatedImage]:
        """
        Extract stills at given timepoints from a video file

        :param media_path: the filename of the video
        :param csv_path: path to the csv file containing timepoint-wise annotations
        :return: a generator of image objects that contains raw Image array and metadata from the annotations
        """
        with open(csv_path, encoding='utf8') as f:
            reader = csv.reader(f)
            next(reader)
            frame_list = [AnnotatedImage(filename=row[0],
                                         label=row[2],
                                         subtype_label=row[3],
                                         mod=row[4].lower() == 'true') for row in reader if row[1] == 'true']
        # CSV rows with mod=True should be discarded (taken as "unseen")
        # maybe we can throw away the video with the least (88) frames annotation from B2 
        # to make 20/20 split on dense vs sparse annotation

        if Path(media_path).is_dir():
            # Process as directory of images
            for frame in frame_list:
                image_path = Path(media_path) / frame.filename
                if image_path.exists():
                    # see https://stackoverflow.com/a/30376272
                    i = Image.open(image_path)
                    frame.image = i.copy()
                    yield frame
                else:
                    logger.warning(f"Image file not found for annotation: {frame.filename}")

        else:
            # this part is doing the same thing as the get_stills function in getstills.py
            # (copied from https://github.com/WGBH-MLA/keystrokelabeler/blob/df4d2bc936fa3a73cdf3004803a0c35c290caf93/getstills.py#L36 )
            container = av.open(media_path)
            video_stream = next((s for s in container.streams if s.type == 'video'), None)
            if video_stream is None:
                raise Exception("No video stream found in {}".format(media_path))
            fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
            cur_target_frame = 0
            fcount = 0
            for frame in container.decode(video=0):
                if cur_target_frame == len(frame_list):
                    break
                ftime = int(fcount/fps * 1000)
                if ftime == frame_list[cur_target_frame].curr_time:
                    frame_list[cur_target_frame].image = frame.to_image()
                    yield frame_list[cur_target_frame]
                    cur_target_frame += 1
                fcount += 1


def main(args):
    in_file = args.input_data
    metadata_file = args.annotation_csv
    featurizer = TrainingDataPreprocessor(args.model)
    logger.info('extractor ready')

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for feat_metadata, feat_mats in featurizer.process_input(in_file, metadata_file):
        logger.info(f'{feat_metadata["guid"]} extraction complete')
        with open(f"{args.outdir}/{feat_metadata['guid']}.json", 'w', encoding='utf8') as f:
            json.dump(feat_metadata, f)
        for name, vectors in feat_mats.items():
            np.save(f"{args.outdir}/{feat_metadata['guid']}.{name}", vectors)
    logger.info('all extraction complete')

    # featurizer.process_input(in_file, metadata_file, args.outdir)
    logger.info('extraction complete')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for preprocessing a video file and its associated manual SWT "
                                                 "annotations to pre-generate (CNN) image feature vectors with manual"
                                                 "labels attached for later training.")
    parser.add_argument("-i", "--input-data", 
                        help="filepath for the video file or a directory of extracted images to be processed.", 
                        required=True) 
    parser.add_argument("-c", "--annotation-csv",
                        help="filepath for the csv containing timepoints + labels.",
                        required=True)
    parser.add_argument("-m", "--model",
                        type=str,
                        help="name of backbone model to use for feature extraction, all available models are used if "
                             "not specified.",
                        choices=list(backbones.model_map.keys()),
                        default=None)
    parser.add_argument("-o", "--outdir",
                        help="directory to save output files. Output files: 1) json with per-frame metadata including "
                             "the manual labels, 2) numpy file(s) with backbone model name siffix.",
                        default=Path(__file__).parent / "extracted")
    clargs = parser.parse_args()
    main(clargs)
