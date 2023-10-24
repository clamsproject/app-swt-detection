import argparse
import csv
import json
import sys
import time
from collections import defaultdict
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
    'vgg16': 4096,
    'resnet50': 2048,
}


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


def int_encode(label):
    slate = ["S"]
    chyron = ["I", "N", "Y"]
    credit = ["C"]
    if label in slate:
        return 0
    elif label in chyron:
        return 1
    elif label in credit:
        return 2
    else:
        return 3


def get_net(in_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
        # no softmax here since we're using CE loss which includes it
        # nn.Softmax(dim=1)
    )


def split_dataset(indir, train_guids, validation_guids, feature_model):
    train_vectors = []
    train_labels = []
    valid_vectors = []
    valid_labels = []
    for j in Path(indir).glob('*.json'):
        guid = j.with_suffix("").name
        feature_vecs = np.load(Path(indir) / f"{guid}.{feature_model}.npy")
        labels = json.load(open(Path(indir) / f"{guid}.json"))
        if guid in validation_guids:
            for i, vec in enumerate(feature_vecs):
                valid_labels.append(int_encode(labels['frames'][i]['label']))
                valid_vectors.append(torch.from_numpy(vec))   
        elif guid in train_guids:
            for i, vec in enumerate(feature_vecs):
                train_labels.append(int_encode(labels['frames'][i]['label']))
                train_vectors.append(torch.from_numpy(vec))
    train = SWTDataset(feature_model, train_labels, train_vectors)
    valid = SWTDataset(feature_model, valid_labels, valid_vectors)
    return train, valid


def k_fold_train(indir, k_fold, feature_model, block_train=(), block_val=()):
    # need to implement "whitelist"? 
    guids = get_guids(indir)
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
        train, valid = split_dataset(indir, train_guids, validation_guids, feature_model)
        if not train.has_data() or not valid.has_data():
            logger.info(f"Skipping fold {i} due to lack of data")
            continue
        train_loader = DataLoader(train, batch_size=40, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)
        loss = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Split {i}: training on {len(train_guids)} videos, validating on {validation_guids}')
        model, p, r, f = train_model(get_net(train.feat_dim), train_loader, valid_loader, loss, device)
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
    out.write(f'\tf-1 = {sum(f_scores)/len(f_scores)}\n')
    out.write(f'\tprecision = {sum(p_scores)/len(p_scores)}\n')
    out.write(f'\trecall = {sum(r_scores)/len(r_scores)}\n')


def train_model(model, train_loader, valid_loader, loss_fn, device, num_epochs=2, export_fname=None):
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
            p = metrics.precision(preds, vlabels, 'multiclass', num_classes=4, average='macro')
            r = metrics.recall(preds, vlabels, 'multiclass', num_classes=4, average='macro')
            f = metrics.f1_score(preds, vlabels, 'multiclass', num_classes=4, average='macro')
            # m = metrics.confusion_matrix(preds, vlabels, 'multiclass', num_classes=4)

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
        export_result(out=export_f, predictions=preds, labels=vlabels, model_name=train_loader.dataset.feature_model)
        logger.info(f"Exported to {export_f.name}")
                
        model.load_state_dict(torch.load(best_model_params_path))
    return model, p, r, f


def export_result(out, predictions, labels, model_name):
    """Exports the data into a human readable format.
    
    @param: predictions - a list of predicted labels across validation instances
    @param: labels      - the list of potential labels
    @param: fname       - name of export file

    @return: class-based accuracy metrics for each label, organized into a csv.
    """
    
    label_metrics = defaultdict(dict)
    
    for i, label in enumerate(["slate", "chyron", "credit", "none"]):
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
    args = parser.parse_args()
    args.allow_guids = []
    args.block_guids = []
    k_fold_train(args.indir, int(args.k_fold), args.featuremodel, args.allow_guids, args.block_guids)
