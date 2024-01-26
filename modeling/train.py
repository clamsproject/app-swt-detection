import argparse
import csv
import json
import logging
import platform
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, IO
import copy

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchmetrics import functional as metrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm

import modeling
from modeling import data_loader

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)-8s %(thread)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# full typology from https://github.com/clamsproject/app-swt-detection/issues/1
FRAME_TYPES = ["B", "S", "S:H", "S:C", "S:D", "S:B", "S:G", "W", "L", "O",
               "M", "I", "N", "E", "P", "Y", "K", "G", "T", "F", "C", "R"]
RESULTS_DIR = Path(__file__).parent / f"results-{platform.node().split('.')[0]}"


class SWTDataset(Dataset):
    def __init__(self, backbone_model_name, labels, vectors):
        self.img_enc_name = backbone_model_name
        self.feat_dim = vectors[0].shape[0] if len(vectors) > 0 else None
        self.labels = labels
        self.vectors = vectors

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.vectors[i], self.labels[i]
    
    def has_data(self):
        return 0 < len(self.vectors) == len(self.labels)


def get_guids(data_dir):
    guids = []
    for j in Path(data_dir).glob('*.json'):
        guid = j.with_suffix("").name
        guids.append(guid)
    return guids


def pre_bin(label, specs):
    if specs is None or "pre" not in specs["bins"]:
        return int_encode(label)
    for i, bin in enumerate(specs["bins"]["pre"].values()):
        if label and label in bin:
            return i
    return len(specs["bins"]["pre"].keys())


def post_bin(label, specs):
    if specs is None:
        return int_encode(label)
    # If no post binning method, just return the label
    if "post" not in specs["bins"]:
        return label
    # If there was no pre-binning, use default int encoding
    if type(label) != str and "pre" not in specs["bins"]:
        if label >= len(FRAME_TYPES):
            return len(FRAME_TYPES)
        label_name = FRAME_TYPES[label]
    # Otherwise, get label name from pre-binning
    else:
        pre_bins = specs["bins"]["pre"].keys()
        if label >= len(pre_bins):
            return len(pre_bins)
        label_name = list(pre_bins)[label]
    
    for i, post_bin in enumerate(specs["bins"]["post"].values()):
        if label_name in post_bin:
            return i
    return len(specs["bins"]["post"].keys())


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


def prepare_datasets(indir, train_guids, validation_guids, configs):
    """
    Given a directory of pre-computed dense feature vectors, 
    prepare the training and validation datasets. The preparation incluses
    1. positional encodings are applied.
    2. 'gold' labels are attached to each vector.
    3. split of vectors into training and validation sets (at video-level, meaning all frames from a video are either in training or validation set).
    returns training dataset, validation dataset, and the number of labels (after "pre"-binning)
    """
    train_vectors = []
    train_labels = []
    valid_vectors = []
    valid_labels = []
    if configs and 'bins' in configs and 'pre' in configs['bins']:
        pre_bin_size = len(configs['bins']['pre'].keys()) + 1
    else:
        pre_bin_size = len(FRAME_TYPES) + 1
    train_vimg = valid_vimg = 0

    extractor = data_loader.FeatureExtractor(
        img_enc_name=configs.get('img_enc_name'),
        pos_enc_name=configs.get('pos_enc_name'),
        pos_unit=configs['pos_unit'] if configs and 'pos_unit' in configs else 3600000,
        pos_enc_dim=configs['pos_enc_dim'] if 'pos_enc_dim' in configs else 512,
        max_input_length=configs.get('max_input_length')
    )
        
    for j in Path(indir).glob('*.json'):
        guid = j.with_suffix("").name
        feature_vecs = np.load(Path(indir) / f"{guid}.{configs['img_enc_name']}.npy")
        labels = json.load(open(Path(indir) / f"{guid}.json"))
        total_video_len = labels['duration']
        for i, vec in enumerate(feature_vecs):
            if not labels['frames'][i]['mod']:  # "transitional" frames
                pre_binned_label = pre_bin(labels['frames'][i]['label'], configs)
                vector = torch.from_numpy(vec)
                position = labels['frames'][i]['curr_time']
                vector = extractor.encode_position(position, total_video_len, vector)
                if guid in validation_guids:
                    valid_vimg += 1
                    valid_vectors.append(vector)
                    valid_labels.append(pre_binned_label)
                elif guid in train_guids:
                    train_vimg += 1
                    train_vectors.append(vector)
                    train_labels.append(pre_binned_label)
    logger.info(f'train: {len(train_guids)} videos, {train_vimg} images, valid: {len(validation_guids)} videos, {valid_vimg} images')
    train = SWTDataset(configs['img_enc_name'], train_labels, train_vectors)
    valid = SWTDataset(configs['img_enc_name'], valid_labels, valid_vectors)
    return train, valid, pre_bin_size


def k_fold_train(indir, config_file, configs, train_id=time.strftime("%Y%m%d-%H%M%S")):
    # need to implement "whitelist"? 
    guids = get_guids(indir)
    configs = load_config(configs) if not isinstance(configs, dict) else configs
    logger.info(f'Using config: {configs}')
    len_val = len(guids) // configs['num_splits']
    val_set_spec = []
    p_scores = []
    r_scores = []
    f_scores = []
    for i in range(0, configs['num_splits']):
        validation_guids = set(guids[i*len_val:(i+1)*len_val])
        train_guids = set(guids) - validation_guids
        for block in configs['block_guids_valid']:
            validation_guids.discard(block)
        for block in configs['block_guids_train']:
            train_guids.discard(block)
        logger.debug(f'After applied block lists:')
        logger.debug(f'train set: {train_guids}')
        logger.debug(f'dev set: {validation_guids}')
        train, valid, labelset_size = prepare_datasets(indir, train_guids, validation_guids, configs)
        # `train` and `valid` vectors DO contain positional encoding after `split_dataset`
        if not train.has_data() or not valid.has_data():
            logger.info(f"Skipping fold {i} due to lack of data")
            continue
        train_loader = DataLoader(train, batch_size=40, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)
        loss = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Split {i}: training on {len(train_guids)} videos, validating on {validation_guids}')
        export_csv_file = f"{RESULTS_DIR}/{train_id}.kfold_{i:03d}.csv"
        export_model_file = f"{RESULTS_DIR}/{train_id}.kfold_{i:03d}.pt"
        model, p, r, f = train_model(
                get_net(train.feat_dim, labelset_size, configs['num_layers'], configs['dropouts']), 
                loss, device, train_loader, valid_loader, configs, labelset_size,
                export_fname=export_csv_file)
        torch.save(model.state_dict(), export_model_file)
        val_set_spec.append(validation_guids)
        p_scores.append(p)
        r_scores.append(r)
        f_scores.append(f)
    if train_id:
        p = Path(f'{RESULTS_DIR}/{train_id}.kfold_results.txt')
        p.parent.mkdir(parents=True, exist_ok=True)
        export_f = open(p, 'w', encoding='utf8')
    else:
        export_f = sys.stdout
    export_kfold_results(val_set_spec, p_scores, r_scores, f_scores, out=export_f, **configs)
    export_config(config_file, configs, train_id)


def export_config(config_file: str, configs: dict, train_id: str):
    config_path = Path(f"{RESULTS_DIR}", f"{train_id}.kfold_config.yml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_file is None:
        configs_copy = copy.deepcopy(configs)
        with open(config_path, 'w') as fh:
            yaml.dump(configs_copy, fh, default_flow_style=False, sort_keys=False)
    else:
        shutil.copyfile(config_file, config_path)


def export_kfold_results(trial_specs, p_scores, r_scores, f_scores, out=sys.stdout, **train_spec):
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


def get_final_label_names(config):
    base = FRAME_TYPES
    if config and "post" in config["bins"]:
        base = list(config["bins"]["post"].keys())
    elif config and "pre" in config["bins"]:
        base = list(config["bins"]["pre"].keys()) 
    return base + [modeling.negative_label]
    

def train_model(model, loss_fn, device, train_loader, valid_loader, configs, n_labels, export_fname=None):
    since = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for num_epoch in tqdm(range(configs['num_epochs'])):

        running_loss = 0.0

        model.train()
        for num_batch, (feats, labels) in enumerate(train_loader):
            feats.to(device)
            labels.to(device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                outputs = model(feats)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                loss.sum().backward()
                optimizer.step()

            running_loss += loss.sum().item() * feats.size(0)
            if num_batch % 100 == 0:
                logger.debug(f'Batch {num_batch} of {len(train_loader)}')
                logger.debug(f'Loss: {loss.sum().item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        
        model.eval()
        for vfeats, vlabels in valid_loader:
            outputs = model(vfeats)
            _, preds = torch.max(outputs, 1)
            # post-binning
            preds = torch.from_numpy(np.vectorize(post_bin)(preds, configs))
            vlabels = torch.from_numpy(np.vectorize(post_bin)(vlabels, configs))
        p = metrics.precision(preds, vlabels, 'multiclass', num_classes=n_labels, average='macro')
        r = metrics.recall(preds, vlabels, 'multiclass', num_classes=n_labels, average='macro')
        f = metrics.f1_score(preds, vlabels, 'multiclass', num_classes=n_labels, average='macro')
        # m = metrics.confusion_matrix(preds, vlabels, 'multiclass', num_classes=n_labels)

        final_classes = get_final_label_names(configs)

        logger.debug(f'Loss: {epoch_loss:.4f} after {num_epoch+1} epochs')
    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    if not export_fname:
        export_f = sys.stdout
    else:
        path = Path(export_fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_f = open(path, 'w', encoding='utf8')
    export_train_result(out=export_f, predictions=preds, labels=vlabels,
                        labelset=final_classes, img_enc_name=train_loader.dataset.img_enc_name)
    logger.info(f"Exported to {export_f.name}")
            
    return model, p, r, f


def export_train_result(out: IO, predictions: Tensor, labels: Tensor, labelset: List[str], img_enc_name: str):
    """Exports the data into a human-readable format.
    @return: class-based accuracy metrics for each label, organized into a csv.
    """

    label_metrics = defaultdict(dict)

    for i, label in enumerate(labelset):
        pred_labels = torch.where(predictions == i, 1, 0)
        true_labels = torch.where(labels == i, 1, 0)
        binary_acc = BinaryAccuracy()
        binary_prec = BinaryPrecision()
        binary_recall = BinaryRecall()
        binary_f1 = BinaryF1Score()
        label_metrics[label] = {"Model_Name": img_enc_name,
                                "Label": label,
                                "Accuracy": binary_acc(pred_labels, true_labels).item(),
                                "Precision": binary_prec(pred_labels, true_labels).item(),
                                "Recall": binary_recall(pred_labels, true_labels).item(),
                                "F1-Score": binary_f1(pred_labels, true_labels).item()}
        
    writer = csv.DictWriter(out, fieldnames=["Model_Name", "Label", "Accuracy", "Precision", "Recall", "F1-Score"])
    writer.writeheader()
    for label, metrics in label_metrics.items():
        writer.writerow(metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="root directory containing the vectors and labels to train on")
    parser.add_argument("-c", "--config", help="the YAML config file specifying binning strategy", default=None)
    # Added because I wanted to be able to overrule the RESULTS_DIR with a parameter,
    # but commented out because I haven't finished that yet. (MV)
    # parser.add_argument("-r", "--results", metavar="DIR", help="the results directory")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        k_fold_train(
            indir=args.indir, config_file=args.config, configs=config,
            train_id=f'{time.strftime("%Y%m%d-%H%M%S")}.{config["img_enc_name"]}')
    else:
        import modeling.gridsearch
        for config in modeling.gridsearch.configs:
            k_fold_train(
                indir=args.indir, config_file=args.config, configs=config,
                train_id=f'{time.strftime("%Y%m%d-%H%M%S")}.{config["img_enc_name"]}')
