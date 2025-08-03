import enum
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Dict, ClassVar

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as transform_functions

from modeling import backbones

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)-8s %(thread)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageResizeStrategy(enum.Enum):
    """
    Enum-like class to define different image resizing strategies.
    These strategies correspond to the three pre-processing schemes:
    1. DISTORTED: Directly resize to 224x224.
    2. CROPPED256: Resize shorter side to 256, then center crop to 224x224.
    3. CROPPED224: Resize shorter side to 224, then pad to 224x224.
    """
    DISTORTED = 'distorted'
    CROPPED256 = 'cropped256'
    CROPPED224 = 'cropped224'

    @classmethod
    def from_string(cls, strategy_str):
        """
        Convert a string to an ImageResizeStrategy enum.
        :param strategy_str: The string representation of the resize strategy.
        :return: An instance of ImageResizeStrategy.
        """
        try:
            return cls[strategy_str.upper()]
        except KeyError:
            raise ValueError(f"Unknown resize strategy: {strategy_str}")

    @classmethod
    def get_transform_function(cls, strategy_str):
        """
        Get the appropriate transform function based on the resize strategy.
        :param strategy_str: The string representation of the resize strategy.
        :return: A function that applies the specified resizing strategy.
        """
        strategy = cls.from_string(strategy_str)
        if strategy == cls.DISTORTED:
            return lambda img: transform_functions.resize(img, [224, 224])
        elif strategy == cls.CROPPED256:
            return lambda img: transform_functions.center_crop(transform_functions.resize(img, [256]), [224])
        elif strategy == cls.CROPPED224:
            return lambda img: transform_functions.center_crop(transform_functions.resize(img, [224]), [224])
        else:
            raise ValueError(f"Unknown resize strategy: {strategy_str}")


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
        self.pos_vec_lookup = self.get_sinusoidal_embeddings(position_dim, self.feature_vector_dim())

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

    def get_img_vector(self, img_vec, as_numpy=True):
        if img_vec.ndim == 3:  # when a single image is passed
            img_vec = img_vec.unsqueeze(0)
        if torch.cuda.is_available():
            img_vec = img_vec.to('cuda')
            self.img_encoder.model.to('cuda')
        with torch.no_grad():
            # for huggingface models, forward() will return 
            # (last_hidden_state, global_average_pool) tuple,
            # and we only care about GAP values.
            feature_vec = self.img_encoder.model(img_vec, False, False)[1]
        if as_numpy:
            return feature_vec.cpu().numpy()
        else:
            return feature_vec.cpu()

    def convert_position(self, cur, tot):
        if cur < self.pos_abs_th_front or tot - cur < self.pos_abs_th_end:
            pos = cur
        else:
            pos = cur * self.pos_vec_lookup.shape[0] // tot
        return int(pos)

    def encode_position(self, position: Tensor, img_vec):
        if isinstance(img_vec, np.ndarray):
            img_vec = torch.from_numpy(img_vec)
        if img_vec.ndim == 3:  # when a single image is passed
            img_vec = img_vec.unsqueeze(0)
        pos_lookup_col = [self.convert_position(pos[0], pos[1]) for pos in position]
        pos_vec = self.pos_vec_lookup[pos_lookup_col] * self.pos_vec_coeff
        return torch.add(img_vec, pos_vec)

    def feature_vector_dim(self):
        return self.img_encoder.dim

    def get_full_feature_vectors(self, raw_imgs, positions):
        img_vecs = self.get_img_vector(raw_imgs, as_numpy=False)
        # having first element only [0] will work for train_loader, but get 
        # `IndexError: too many indices for tensor of dimension 2` 
        # for valid_loader
        return self.encode_position(positions, img_vecs)


class TrainingDataPreprocessor(object):
    """
    Handles the creation of HDF5 datasets from image annotations and zip 
    files. 
    """

    def __init__(self, annotation_csv_path, image_zip_path, output_dir):
        """
        Initializes basic paths and read csv that has annotations.
        """
        self.image_zip = image_zip_path
        self.output_dir = output_dir
        self.guid = Path(annotation_csv_path).stem
        self.metadata = pd.read_csv(annotation_csv_path)
        self.metadata['at'] = self.metadata['at'].apply(self._convert_to_milliseconds)

    @staticmethod
    def _convert_to_milliseconds(time_str):
        """Convert time in hh:mm:ss.ms format to milliseconds."""
        hms, ms = time_str.split('.')
        ms = int(ms) if ms else 0
        h, m, s = map(int, hms.split(':'))
        return (h * 3600 + m * 60 + s) * 1000 + ms

    @staticmethod
    def _ensure_group(h5file, group_name):
        """Ensure that the given group_name in h5 file is a Group, not a Dataset or Datatype."""
        if group_name in h5file:
            if not isinstance(h5file[group_name], h5py.Group):
                del h5file[group_name]
                return h5file.create_group(group_name)
            else:
                return h5file[group_name]
        else:
            return h5file.create_group(group_name)

    def create_hdf5_dataset(self):
        """
        Creates a single HDF5 file for the given GUID. Instead of creating 
        three separate HDF5 files per video, this implementation stores the 
        three pre-processing schemes as different groups within a h5 file,
        e.g., 'images_distorted', 'images_cropped256', 'images_cropped224'.
        This still achieves the goal of co-locating all data for a single 
        guid. Each image will get an identifier that encodes all the 
        necessary metadata (timing info, manual label).
        """
        hdf5_path = Path(self.output_dir) / f"{self.guid}.h5"

        # Check if the HDF5 file already exists
        if hdf5_path.exists():
            mode = 'a'  # Append mode
        else:
            mode = 'w'  # Write mode

        with h5py.File(hdf5_path, mode) as hdf5_file:
            zf = zipfile.ZipFile(self.image_zip, 'r')
            for (_, row), zfile in zip(self.metadata.iterrows(), sorted(zf.namelist())):
                # sanity check for timing match 
                guid, total_time, _ = zfile.split('_', maxsplit=2)
                total_time = int(total_time)
                ts = int(zfile.split('_')[-1].split('.')[0])
                if row['at'] != ts:
                    raise ValueError(f"Timestamp mismatch: {row['at']} != {ts} in file {zfile}")
                # prepare single string for label
                subtype_label = row['scene-subtype'] if not pd.isna(row['scene-subtype']) else ''
                concat_label = (row['scene-type'] + subtype_label).strip()
                # then reformat the image file name to data ID with all metadata
                image_id = f"{self.guid}_{ts}_{total_time}_{concat_label}"
                with zf.open(zfile) as img_file:
                    img = Image.open(img_file)
                    for resize_strategy in ImageResizeStrategy:
                        # Create a group for each resize strategy (e.g., 'images_distorted')
                        reshaped_group = self._ensure_group(hdf5_file, f'images_{resize_strategy.value}')

                        # Resize and convert the image to an array
                        img_resized = self._resize_image(img, resize_strategy)
                        img_array = np.array(img_resized)

                        # Avoid overwriting existing datasets, append if not present
                        if image_id not in reshaped_group:
                            reshaped_group.create_dataset(image_id, data=img_array, compression="gzip")

    @staticmethod
    def _resize_image(img, strategy):
        """Resize the image according to the specified strategy enum or string."""
        if isinstance(strategy, ImageResizeStrategy):
            transform_function = ImageResizeStrategy.get_transform_function(strategy.value)
            return transform_function(img)
        elif isinstance(strategy, str):
            strategy_enum = ImageResizeStrategy.from_string(strategy)
            transform_function = ImageResizeStrategy.get_transform_function(strategy_enum.value)
            return transform_function(img)
        else:
            raise ValueError(f"Invalid resize strategy: {strategy}")


def main():
    # read csv directory and zip directory from .env file or os.environ
    try:
        csv_dir = os.getenv('ANNDIR')
        zip_dir = os.getenv('ZIPDIR')
        out_dir = os.getenv('H5_DIR')
    except KeyError as e:
        raise KeyError(f"Environment variable not set, ensure `ANNDIR`, `ZIPDIR`, and `H5_DIR` are defined.")

    import argparse

    parser = argparse.ArgumentParser(description="Create HDF5 datasets from image annotations.")
    parser.add_argument('guids', metavar='GUID', type=str, nargs='*', 
                        help="GUID(s) for the dataset(s). If none are provided, "
                             "all CSV files in ANNDIR will be processed.")
    # then, "view" mode
    parser.add_argument('-v', '--view', action='store_true', help="View the HDF5 dataset.")

    args = parser.parse_args()
    # first make sure output dir exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine the number of workers for the ThreadPoolExecutor
    # Use a sensible default, e.g., number of CPU cores or a fixed number
    num_workers = os.cpu_count() or 4 
    logger.info(f'using {num_workers} workers')

    # ignore `d` batch since it's not used in train/valid 
    from modeling.config import batches
    excluded_guids = batches.excluded_guids
    
    # If no GUIDs are provided, glob CSV files in ANNDIR and extract GUIDs from their stems
    if not args.guids:
        csv_files = Path(csv_dir).glob('*.csv')
        args.guids = [f.stem for f in csv_files]

    # Use ThreadPoolExecutor for parallel processing of GUIDs
    futures = []

    for i, guid in enumerate(args.guids, 1):
        if guid in excluded_guids:
            print(f"Ignoring '{guid}' dataset as it is not used in training/validation.")
            continue
        csv_path = Path(csv_dir) / f"{guid}.csv"
        image_zip_path = Path(zip_dir) / f"{guid}.zip"
        if not csv_path.exists() or not image_zip_path.exists():
            print(f"Ignoring '{guid}' annotation as image zip file is not found.")
            continue

        if args.view:
            h5fname = Path(out_dir) / f"{guid}.h5"
            if not h5fname.exists():
                raise FileNotFoundError(f"HDF5 file {h5fname} does not exist. Please create it first.")
            with h5py.File(h5fname, 'r') as hdf5_file:
                print(f"\n--- Viewing HDF5 for GUID: {guid} ---")
                print("\nLabels Group:")
                for strategy in ImageResizeStrategy:
                    print(f"\nImages Group ({strategy.value}):")
                    if f'images_{strategy.value}' in hdf5_file:
                        for key in hdf5_file[f'images_{strategy.value}'].keys():
                            img_data = hdf5_file[f'images_{strategy.value}'][key][()]
                            print(f"  {key}: shape {img_data.shape}")
                    else:
                        print(f"  No images found for strategy {strategy.value}.")
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures.append(executor.submit(
                    TrainingDataPreprocessor(csv_path, image_zip_path, out_dir).create_hdf5_dataset
                ))
    
    # Wait for all tasks to complete and handle potential exceptions
    for future in as_completed(futures):
        try:
            future.result()  # This will re-raise any exception that occurred in the thread
        except Exception as exc:
            logger.error(f'HDF5 creation generated an exception: {exc}')


if __name__ == "__main__":
    main()
