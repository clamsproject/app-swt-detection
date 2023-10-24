import argparse
import csv
import json
import sys
import time
from collections import defaultdict
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import functional as metrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)-8s %(thread)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

feat_dims = {
    "convnext_base": 1024,
    "convnext_tiny": 768,
    "convnext_small": 768,
    "convnext_lg": 1536,
    "densenet121": 1024,
    "efficientnet_small": 1280,
    "efficientnet_med": 1280,
    "efficientnet_large": 1280,
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "vgg16": 4096,
    "bn_vgg16": 4096,
    "vgg19": 4096,
    "bn_vgg19": 4096,
}

# full typology from https://github.com/clamsproject/app-swt-detection/issues/1
FRAME_TYPES = ["B", "S", "S:H", "S:C", "S:D", "S:B", "S:G", "W", "L", "O",
               "M", "I", "N", "E", "P", "Y", "K", "G", "T", "F", "C", "R"]


class SWTDataset(Dataset):
    def __init__(self, feature_model, labels, vectors):
        self.feature_model = feature_model
        self.feat_dim = feat_dims[feature_model]
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
            print("Invalid config file. Using full label set.")
            return None
                

def int_encode(label):
    if not isinstance(label, str):
        return label
    if label in FRAME_TYPES:
        return FRAME_TYPES.index(label)
    else:
        return len(FRAME_TYPES)


def get_net(in_dim, n_labels):
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Linear(128, n_labels),
        # no softmax here since we're using CE loss which includes it
        # nn.Softmax(dim=1)
    )


def split_dataset(indir, train_guids, validation_guids, feature_model, bins):
    train_vectors = []
    train_labels = []
    valid_vectors = []
    valid_labels = []
    if bins and 'bins' in bins and 'pre' in bins['bins']:
        pre_bin_size = len(bins['bins']['pre'].keys()) + 1
    else:
        pre_bin_size = len(FRAME_TYPES) + 1
    train_vnum = train_vimg = valid_vnum = valid_vimg = 0
    for j in Path(indir).glob('*.json'):
        guid = j.with_suffix("").name
        feature_vecs = np.load(Path(indir) / f"{guid}.{feature_model}.npy")
        labels = json.load(open(Path(indir) / f"{guid}.json"))
        if guid in validation_guids:
            valid_vnum += 1
            for i, vec in enumerate(feature_vecs):
                valid_vimg += 1
                valid_labels.append(pre_bin(labels['frames'][i]['label'], bins))
                valid_vectors.append(torch.from_numpy(vec))
        elif guid in train_guids:
            train_vnum += 1
            for i, vec in enumerate(feature_vecs):
                train_vimg += 1
                train_labels.append(pre_bin(labels['frames'][i]['label'], bins))
                train_vectors.append(torch.from_numpy(vec))
    print(f'train: {train_vnum} videos, {train_vimg} images, valid: {valid_vnum} videos, {valid_vimg} images')
    train = SWTDataset(feature_model, train_labels, train_vectors)
    valid = SWTDataset(feature_model, valid_labels, valid_vectors)
    return train, valid, pre_bin_size


def k_fold_train(indir, k_fold, feature_model, bins, block_train=(), block_val=()):
    # need to implement "whitelist"? 
    guids = get_guids(indir)
    bins = load_config(bins)
    len_val = len(guids) // k_fold
    val_set_spec = []
    p_scores = []
    r_scores = []
    f_scores = []
    for i in range(0, k_fold):
        validation_guids = set(guids[i*len_val:(i+1)*len_val])
        train_guids = set(guids) - validation_guids
        for block in block_val:
            validation_guids.discard(block)
        for block in block_train:
            train_guids.discard(block)
        logger.debug(f'After applied block lists:')
        logger.debug(f'train set: {train_guids}')
        logger.debug(f'dev set: {validation_guids}')
        train, valid, labelset_size = split_dataset(indir, train_guids, validation_guids, feature_model, bins)
        if not train.has_data() or not valid.has_data():
            logger.info(f"Skipping fold {i} due to lack of data")
            continue
        train_loader = DataLoader(train, batch_size=40, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)
        loss = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Split {i}: training on {len(train_guids)} videos, validating on {validation_guids}')
        model, p, r, f = train_model(get_net(train.feat_dim, labelset_size), train_loader, valid_loader, loss, device, bins, labelset_size)
        val_set_spec.append(validation_guids)
        p_scores.append(p)
        r_scores.append(r)
        f_scores.append(f)
    print_scores(val_set_spec, p_scores, r_scores, f_scores)


def print_scores(trial_specs, p_scores, r_scores, f_scores, out=sys.stdout):
    max_f1_idx = f_scores.index(max(f_scores))
    min_f1_idx = f_scores.index(min(f_scores))
    out.write(f'Highest f1 @ {trial_specs[max_f1_idx]}\n')
    out.write(f'\tf-1 = {f_scores[max_f1_idx]}\n')
    out.write(f'\tprecision = {p_scores[max_f1_idx]}\n')
    out.write(f'\trecall = {r_scores[max_f1_idx]}\n')
    out.write(f'Lowest f1 @ {trial_specs[min_f1_idx]}\n')
    out.write(f'\tf-1 = {f_scores[min_f1_idx]}\n')
    out.write(f'\tprecision = {p_scores[min_f1_idx]}\n')
    out.write(f'\trecall = {r_scores[min_f1_idx]}\n')
    out.write('Mean performance\n')
    out.write(f'\tf-1 = {sum(f_scores) / len(f_scores)}\n')
    out.write(f'\tprecision = {sum(p_scores) / len(p_scores)}\n')
    out.write(f'\trecall = {sum(r_scores) / len(r_scores)}\n')


def get_valid_classes(config):
    base = FRAME_TYPES
    if config and "post" in config["bins"]:
        base = list(config["bins"]["post"].keys())
    elif config and "pre" in config["bins"]:
        base = list(config["bins"]["pre"].keys()) 
    return base + ["none"]
    

def train_model(model, train_loader, valid_loader, loss_fn, device, bins, n_labels, num_epochs=2, export_fname=None):
    since = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir) / 'best_model_params.pt'

        torch.save(model.state_dict(), best_model_params_path)

        for num_epoch in tqdm(range(num_epochs)):

            running_loss = 0.0

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
            for vfeats, vlabels in valid_loader:
                outputs = model(vfeats)
                _, preds = torch.max(outputs, 1)
                # post-binning
                preds = torch.from_numpy(np.vectorize(post_bin)(preds, bins))
                vlabels = torch.from_numpy(np.vectorize(post_bin)(vlabels, bins))
            p = metrics.precision(preds, vlabels, 'multiclass', num_classes=n_labels, average='macro')
            r = metrics.recall(preds, vlabels, 'multiclass', num_classes=n_labels, average='macro')
            f = metrics.f1_score(preds, vlabels, 'multiclass', num_classes=n_labels, average='macro')
            # m = metrics.confusion_matrix(preds, vlabels, 'multiclass', num_classes=n_labels)

            valid_classes = get_valid_classes(bins)

            logger.debug(f'Loss: {epoch_loss:.4f} after {num_epoch+1} epochs')
        time_elapsed = time.time() - since
        logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if export_fname is None:
            export_f = sys.stdout
        else:
            p = Path(export_fname)
            p.parent.mkdir(parents=True, exist_ok=True)
            export_f = open(p.parent / f'{timestamp}.{p.name}', 'w', encoding='utf8')
        export_result(out=export_f, predictions=preds, labels=vlabels, labelset=valid_classes, model_name=train_loader.dataset.feature_model)
        logger.info(f"Exported to {export_f.name}")
                
        model.load_state_dict(torch.load(best_model_params_path))
    return model, p, r, f


def export_result(out, predictions, labels, labelset, model_name):
    """Exports the data into a human readable format.
    
    @param: predictions - a list of predicted labels across validation instances
    @param: labels      - the list of potential labels
    @param: fname       - name of export file

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
        label_metrics[label] = {"Model_Name": model_name,
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
    parser.add_argument("featuremodel", help="feature vectors to use for training", choices=['vgg16', 'resnet50'], default='vgg16')
    parser.add_argument("k_fold", help="k (interger), the number of distinct dev splits to evaluate on", default=10)
    parser.add_argument("-b", "--bins", help="The YAML config file specifying binning strategy", default=None)
    args = parser.parse_args()
    args.block_train = []
    args.block_valid = []
    k_fold_train(indir=args.indir, k_fold=int(args.k_fold), feature_model=args.featuremodel, block_train=args.block_train, block_val=args.block_valid, bins=args.bins)
