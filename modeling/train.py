import argparse
import collections
import copy
import hashlib
import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Union, List, Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader

from modeling import data_loader, gridsearch, FRAME_TYPES, get_prebinned_labelset
from modeling.config import bins
from modeling.config.batches import guids_for_fixed_validation_set

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)-8s %(thread)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / f"results-{platform.node().split('.')[0]}"
# at 200 
# convnext_large (v1 & v2) will use ~8GB of vram 
# this is roughly the safe maximum for desktop GPUs with 12-16GB of vram
BATCH_SIZE = 200  
    # # these sizes are for A6000 GPU
    # if 'tiny' in img_enc_name:
    #     batch_size = 1600
    # elif 'base' in img_enc_name:
    #     batch_size = 1600
    # elif 'large' in img_enc_name:
    #     batch_size = 800


class SWTH5Dataset(Dataset):
    """
    A PyTorch Dataset for loading image data from a collection of HDF5 files.

    This dataset is designed to work with HDF5 files where each file contains
    images from a single source (e.g., a video), and the images are organized
    into groups based on a resizing strategy.
    """

    def __init__(self, 
                 h5_dir: str,
                 guids: List[str], 
                 backbone_model_name: str,
                 resize_strategy: str, 
                 prebin: Dict = None, 
                 evalmode=False):
        """
        Initializes the dataset. 

        :param guids: A list of paths to the HDF5 files.
        :param resize_strategy: The resizing strategy to use (e.g., 'distorted',
                                'cropped256', 'cropped224'). This corresponds to
                                the group name in the HDF5 files.
        :param prebin: prebin strategy to use. 
        """
        self.guids = guids
        self.image_group_name = f'images_{resize_strategy}'
        self.h5_dir = Path(h5_dir)
        self.index_map = []
        self.total_images = 0
        self.prebin = prebin
        self.img_enc_name = backbone_model_name
        self.feat_extr = data_loader.FeatureExtractor(img_enc_name=self.img_enc_name)
        self.feat_dim = self.feat_extr.feature_vector_dim()
        if evalmode:
            self.feat_extr.img_encoder.model.eval()

        # Create a mapping from a global index to a file and an in-file index
        for i, guid in enumerate(self.guids):
            file_path = self.h5_dir / f"{guid}.h5"
            if not file_path.exists():
                continue
            with h5py.File(file_path, 'r') as f:
                if self.image_group_name not in f:
                    logger.warning(f"Image group '{self.image_group_name}' not found in {file_path}. Skipping file.")
                    continue
                num_images = len(f[self.image_group_name])
                image_keys = sorted(list(f[self.image_group_name].keys()))
                for i in range(num_images):
                    self.index_map.append((guid, image_keys[i]))
                self.total_images += num_images

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return self.total_images

    def __getitem__(self, idx):
        # This is the standard method, kept for compatibility.
        # It can simply call the plural version for a single item.
        return self.__getitems__([idx])[0]

    def __getitems__(self, idxs):
        """
        Retrieves an item from the dataset at the specified index.

        :param idxs: The index of the item to retrieve.
        :return: A tuple containing the image and its corresponding label.
        """
        # map from guid (file_id) to (datum_id, image_id) tuples
        guid_to_idx = collections.defaultdict(list)
        for zero_idx, idx in enumerate(idxs):
            guid, image_key = self.index_map[idx]
            guid_to_idx[guid].append((zero_idx, image_key))
        # three tensor placeholders initiated with the exact "idxs" length
        images = torch.empty((len(idxs), 3, 224, 224))
        positions = torch.empty((len(idxs), 2))
        labels = torch.empty((len(idxs),))
        for guid, keys in guid_to_idx.items():
            file_path = self.h5_dir / f"{guid}.h5"
            with h5py.File(file_path, 'r') as f:
                for idx, image_key in keys:
                    # retrieve reshaped image vector and put in collection tensor
                    images[idx] = torch.from_numpy(f[self.image_group_name][image_key][()])

                    parts = image_key.split('_')
                    
                    # Create position tensor
                    cur_time = int(parts[1])
                    tot_time = int(parts[2])
                    positions[idx] = torch.tensor([cur_time, tot_time])

                    # Integer-encode the label
                    label = pretraining_bin(parts[3][0], self.prebin)
                    labels[idx] = label
        features = self.feat_extr.get_full_feature_vectors(images, positions)
        return list(zip(features, labels.to(torch.long)))


def get_guids(data_dir):
    guids = set()
    # iterate through *.json or *.csv 
    for suffix in 'json csv zip h5'.split():
        for j in Path(data_dir).glob(f'*.{suffix}'):
            guid = j.with_suffix("").name
            guids.add(guid)
    return sorted(list(guids))


def pretraining_bin(label, specs):
    if specs is None or len(specs) == 0:
        return int_encode(label)
    for i, ptbin in enumerate(specs.values()):
        if label and label in ptbin:
            return i
    return len(specs.keys())


def load_config(config):
    if config is None:
        return None
    with open(config) as f:
        try:
            return(yaml.safe_load(f))
        except yaml.scanner.ScannerError:
            logger.error("Invalid config file. Using full label set.")
            return None
                

def int_encode(label):
    if not isinstance(label, str):
        return label
    if label in FRAME_TYPES:
        return FRAME_TYPES.index(label)
    else:
        return len(FRAME_TYPES)


def get_net(in_dim, n_labels, num_layers, dropout=0.0):
    dropouts = [dropout] * (num_layers - 1) if isinstance(dropout, (int, float)) else dropout
    if len(dropouts) + 1 != num_layers:
        raise ValueError("length of dropout must be equal to num_layers - 1")
    net = nn.Sequential()
    for i in range(1, num_layers):
        neurons = max(128 // i, n_labels)
        net.add_module(f"dropout{i}", nn.Dropout(p=dropouts[i - 1]))
        net.add_module(f"fc{i}", nn.Linear(in_dim, neurons))
        net.add_module(f"relu{i}", nn.ReLU())
        in_dim = neurons
    net.add_module("fc_out", nn.Linear(neurons, n_labels))
    # no softmax here since we're using CE loss which includes it
    # net.add_module(Softmax(dim=1))
    return net


def train(indir, outdir, config_file, configs, train_id=time.strftime("%Y%m%d-%H%M%S")):
    os.makedirs(outdir, exist_ok=True)

    # need to implement "whitelist"?
    guids = get_guids(indir)
    configs = load_config(configs) if not isinstance(configs, dict) else configs
    for k, v in configs.items():
        if isinstance(v, list):
            logger.info(f'Using config: {k}=({len(v)}) {v[:5]}...')
        else:
            logger.info(f'Using config: {k}={v}')
    train_all_guids = set(guids) - set(configs['block_guids_train'])
    loss = nn.CrossEntropyLoss(reduction="none")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # the number of labels (after "pre"-binning)
    if configs and 'prebin' in configs and len(configs['prebin']) > 0:
        num_labels = len(configs['prebin'].keys()) + 1
    else:
        num_labels = len(FRAME_TYPES) + 1
    labelset = get_prebinned_labelset(configs)
    logger.info(f'Labels for training: ({num_labels}) {labelset}')

    valid_guids = guids_for_fixed_validation_set
    train_all_guids = list(train_all_guids - set(valid_guids))
    img_enc_name = configs['img_enc_name']
    resize_strategy = configs['resize_strategy']
    prebin = configs.get('prebin', None)
    train = SWTH5Dataset(indir, train_all_guids, img_enc_name, resize_strategy, prebin)
    valid = SWTH5Dataset(indir, valid_guids, img_enc_name, resize_strategy, prebin, evalmode=True)
    logger.info(f'Instances for training: {str(len(train))}')
    logger.info(f'Instances for validation: {str(len(valid))}')
    
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    base_fname = f"{outdir}/{train_id}"
    export_model_file = f"{base_fname}.pt"
    t = time.perf_counter()
    model, epoch_losses = train_model(
        get_net(train.feat_dim, num_labels, configs['num_layers'], configs['dropouts']),
        loss, device, train_loader, configs)
    model.to('cpu')  # move to cpu before saving to make device-agnostic when loading later
    torch.save(model.state_dict(), export_model_file)
    p_config = Path(f'{base_fname}.yml')
    train_elapsed = time.perf_counter() - t
    _, _, _, valid_elapsed = validate(model, device, valid_loader, labelset, export_fname=f'{base_fname}.csv')
    vram_usage = log_peak_vram_usage()
    with open(f'{base_fname}.csv', 'a') as export_file:
        export_file.write('\n\n')
        export_file.write(f"peak-vram-usage,{vram_usage}\n")
        export_file.write(f"training-time,{train_elapsed}\n")
        export_file.write(f"validation-time,{valid_elapsed}\n")
        export_file.write('\n\n')
        for epoch, loss in enumerate(epoch_losses, 1):
            export_file.write(f'Epoch {epoch} loss,{loss:.6f}\n')
    export_train_config(config_file, configs, p_config)
    # then unload datasets and release gpu memory 
    del model
    del train
    del valid


def export_train_config(config_file: str, configs: dict, outfile: Union[str, Path]):
    if config_file is None:
        configs_copy = copy.deepcopy(configs)
        with open(outfile, 'w') as fh:
            yaml.dump(configs_copy, fh, default_flow_style=False, sort_keys=False)
    else:
        shutil.copyfile(config_file, outfile)


def export_kfold_results(trial_specs, p_scores, r_scores, f_scores, p_results):
    with open(p_results, 'w') as out:
        max_f1_idx = f_scores.index(max(f_scores))
        min_f1_idx = f_scores.index(min(f_scores))
        out.write(f'Highest f1 @ {max_f1_idx:03d}\n')
        out.write(f'\t{trial_specs[max_f1_idx]}\n')
        out.write(f'\tf-1 = {f_scores[max_f1_idx]}\n')
        out.write(f'\tprecision = {p_scores[max_f1_idx]}\n')
        out.write(f'\trecall = {r_scores[max_f1_idx]}\n')
        out.write(f'Lowest f1 @ {min_f1_idx:03d}\n')
        out.write(f'\t{trial_specs[min_f1_idx]}\n')
        out.write(f'\tf-1 = {f_scores[min_f1_idx]}\n')
        out.write(f'\tprecision = {p_scores[min_f1_idx]}\n')
        out.write(f'\trecall = {r_scores[min_f1_idx]}\n')
        out.write('Mean performance\n')
        out.write(f'\tf-1 = {sum(f_scores) / len(f_scores)}\n')
        out.write(f'\tprecision = {sum(p_scores) / len(p_scores)}\n')
        out.write(f'\trecall = {sum(r_scores) / len(r_scores)}\n')


def log_peak_vram_usage():
    if not torch.cuda.is_available():
        return "No CUDA available"
    
    num_gpus = torch.cuda.device_count()
    vram_info = []
    
    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        gpu_name = torch.cuda.get_device_name(i)
        
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = peak_bytes / (1024 * 1024)
        
        vram_info.append(f"GPU {i} ({gpu_name}): {peak_mb:.2f}MB peak")
    
    return "; ".join(vram_info)


def train_model(model, loss_fn, device, train_loader, configs):
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    since = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []  # List to store epoch losses

    for num_epoch in tqdm(range(configs['num_epochs'])):

        running_loss = 0.0

        model.train()
        for num_batch, (feats, labels) in enumerate(train_loader):
            feats = feats.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                outputs = model(feats)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                loss.sum().backward()
                optimizer.step()

            running_loss += loss.sum().item()
            if num_batch % 100 == 0:
                logger.debug(f'Batch {num_batch} of {len(train_loader)}')
                logger.debug(f'Loss: {loss.mean().item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)

        logger.debug(f'Loss: {epoch_loss:.4f} after {num_epoch+1} epochs')
    time_elapsed = time.perf_counter() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model, epoch_losses  # Return the model and recorded losses


def normalize_config_and_get_train_id_suffix_from_config(config):
    """ 
    This will alter config dict, so except right before calling a worker, use a copy of the config
    """
    backbonename = config['img_enc_name']
    if len(config['prebin']) == 0:  # empty binning = no binning
        config.pop('prebin')
        prebin_name = 'noprebin'
    elif isinstance(config['prebin'], str):
        prebin_name = config['prebin']
        config['prebin'] = bins.binning_schemes[prebin_name]
    else:
        # "regular" fully-custom binning config via a proper dict - can't set a name for this
        prebin_name = 'custom'
    positionalencoding = "pos" + ("F" if config["pos_vec_coeff"] == 0 else "T")
    resize_strategy = config['resize_strategy']
    suffix = '.'.join(filter(None, [backbonename, resize_strategy, prebin_name, positionalencoding]))
    return suffix if suffix else 'noconfig'


def check_results_exist(outdir, train_id_prefix, config):
    """Check if results already exist for a given configuration."""
    # Find all .yml config files that match the timestamp pattern
    suffix = normalize_config_and_get_train_id_suffix_from_config(config.copy())  # just to normalize the config
    existing_config_files = list(Path(outdir).glob(f"{train_id_prefix}.*.{suffix}.yml"))
    
    # Compare each existing config with the current one
    for existing_config_file in existing_config_files:
        print(f'Checking existing config: {existing_config_file}')
        try:
            with open(existing_config_file, 'r') as f:
                existing_config = yaml.safe_load(f)
            
            # Compare configurations
            if configs_match(config, existing_config):
                # Check if the corresponding CSV results file exists
                csv_file = existing_config_file.with_suffix('.csv')
                if csv_file.exists():
                    return True
        except (yaml.YAMLError, FileNotFoundError, KeyError):
            # Skip files that can't be read or parsed
            continue
    
    return False


def configs_match(config1, config2):
    """Compare two configuration dictionaries for equivalence."""
    for key in gridsearch.clss_param_keys:
        if not (key in config1 and key in config2 and config1[key] == config2[key]):
            return False
    return True


if __name__ == "__main__":
    # imports only used in actual training runs (not in app.py)
    import h5py
    from tqdm import tqdm
    
    from .validate import validate

    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="root directory containing the vectors and labels to train on")
    parser.add_argument("-c", "--config", metavar='FILE', help="the YAML model config file", default=None)
    parser.add_argument("-o", "--outdir", metavar='DIR', help="the results directory", default=RESULTS_DIR)
    parser.add_argument("-p", "--train-id-prefix", metavar='PREFIX', help="prefix for training IDs (default: current timestamp)", default=None)
    args = parser.parse_args()

    if args.config:
        configs = [load_config(args.config)]
    else:
        import modeling.gridsearch
        configs = list(modeling.gridsearch.get_classifier_training_grids())

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    print(f'training with {str(len(configs))} different configurations')
    
    # Set train_id_prefix from argument or use current timestamp as fallback
    train_id_prefix = args.train_id_prefix or time.strftime("%Y%m%d-%H%M%S")
    print(f"Using train ID prefix: {train_id_prefix}")
    
    import multiprocessing

    def train_worker(config, gpu_id, args, train_id_prefix):
        # Generate a unique hash for the config to ensure unique train_id
        config_str = str(sorted(config.items())) # Convert config to a sorted string for consistent hashing
        config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:12] # Take first 12 chars
        
        # Combine prefix, hash, and normalized config suffix
        train_id = '.'.join([train_id_prefix, config_hash, normalize_config_and_get_train_id_suffix_from_config(config)])
        
        logger.info(f"Starting training with ID: {train_id} on GPU {gpu_id}")
        with torch.cuda.device(gpu_id):
            # Call the train function with the correct parameters
            train(indir=args.indir, outdir=args.outdir, config_file=args.config, configs=config, train_id=train_id)

    num_gpus = torch.cuda.device_count() or 1
    print(f"Using {num_gpus} parallel workers (GPUs detected: {torch.cuda.device_count()})")
    processes = []
    skipped_count = 0
    
    for i, config in enumerate(configs):
        # Check if results already exist for this configuration
        if check_results_exist(args.outdir, train_id_prefix, config):
            skipped_count += 1
            suffix = normalize_config_and_get_train_id_suffix_from_config(config.copy())
            print(f"Skipping {i+1} of {len(configs)} (results exist): {suffix}")
            continue

        gpu_id = i % num_gpus
        p = multiprocessing.Process(target=train_worker, args=(config, gpu_id, args, train_id_prefix))
        print(f"Training {i+1} of {len(configs)} configuration on GPU {gpu_id}")
        p.start()
        processes.append(p)
        if len(processes) >= num_gpus:
            for proc in processes:
                proc.join()
            processes = []
    for proc in processes:
        proc.join()
        
    if skipped_count > 0:
        print(f"Skipped {skipped_count} configurations with existing results")
