import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import functional as metrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm

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


class SWTDataset(Dataset):
    def __init__(self, feature_model, labels, vectors, allow_guids=[]):
        self.feature_model = feature_model
        self.feat_dim = feat_dims[feature_model]
        self.labels = labels
        self.vectors = vectors

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.vectors[i], self.labels[i]


def get_guids(dir, block_guids=[]):
    # TODO (krim @ 10/10/23): implement whitelisting
    guids = []
    for j in Path(dir).glob('*.json'):
        guid = j.with_suffix("").name
        if guid not in block_guids:
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


def split_dataset(indir, validation_guids, feature_model):
    train_vectors = []
    train_labels = []
    valid_vectors = []
    valid_labels = []
    train_vnum = train_vimg = valid_vnum = valid_vimg = 0
    for j in Path(indir).glob('*.json'):
        guid = j.with_suffix("").name
        feature_vecs = np.load(Path(indir) / f"{guid}.{feature_model}.npy")
        labels = json.load(open(Path(indir) / f"{guid}.json"))
        if guid in validation_guids:
            valid_vnum += 1
            for i, vec in enumerate(feature_vecs):
                valid_vimg += 1
                valid_labels.append(int_encode(labels['frames'][i]['label']))
                valid_vectors.append(torch.from_numpy(vec))
        else:
            train_vnum += 1
            for i, vec in enumerate(feature_vecs):
                train_vimg += 1
                train_labels.append(int_encode(labels['frames'][i]['label']))
                train_vectors.append(torch.from_numpy(vec))
    print(f'train: {train_vnum} videos, {train_vimg} images, valid: {valid_vnum} videos, {valid_vimg} images')
    train = SWTDataset(feature_model, train_labels, train_vectors)
    valid = SWTDataset(feature_model, valid_labels, valid_vectors)
    return train, valid


def k_fold_train(indir, k_fold, feature_model, whitelist, blacklist):
    guids = get_guids(indir, blacklist)
    val_set_spec = []
    p_scores = []
    r_scores = []
    f_scores = []
    for i in range(k_fold):
        validation_guids = {guids[i]}
        train, valid = split_dataset(indir, validation_guids, feature_model)
        train_loader = DataLoader(train, batch_size=40, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)
        print(len(train), len(valid))
        loss = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'training on {len(guids) - len(validation_guids)} videos, validating on {validation_guids}')
        model, p, r, f = train_model(get_net(train.feat_dim), train_loader, valid_loader, loss, device)
        val_set_spec.append(validation_guids)
        p_scores.append(p)
        r_scores.append(r)
        f_scores.append(f)
    print_scores(val_set_spec, p_scores, r_scores, f_scores)


def print_scores(trial_specs, p_scores, r_scores, f_scores):
    max_f1_idx = f_scores.index(max(f_scores))
    min_f1_idx = f_scores.index(min(f_scores))
    print(f"Highest f1 @ {trial_specs[max_f1_idx]}")
    print(f'\tf-1 = {f_scores[max_f1_idx]}')
    print(f'\tprecision = {p_scores[max_f1_idx]}')
    print(f'\trecall = {r_scores[max_f1_idx]}')
    print(f"Lowest f1 @ {trial_specs[min_f1_idx]}")
    print(f'\tf-1 = {f_scores[min_f1_idx]}')
    print(f'\tprecision = {p_scores[min_f1_idx]}')
    print(f'\trecall = {r_scores[min_f1_idx]}')
    print("Mean performance")
    print(f'\tf-1 = {sum(f_scores) / len(f_scores)}')
    print(f'\tprecision = {sum(p_scores) / len(p_scores)}')
    print(f'\trecall = {sum(r_scores) / len(r_scores)}')


def train_model(model, train_loader, valid_loader, loss_fn, device, num_epochs=2):
    since = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir) / 'best_model_params.pt'

        torch.save(model.state_dict(), best_model_params_path)

        for num_epoch in tqdm(range(num_epochs)):
            # print(f'Epoch {epoch}/{num_epochs - 1}')
            # print('-' * 10)

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
                    print(f'Batch {num_batch} of {len(train_loader)}')
                    print(f'Loss: {loss.sum().item():.4f}')

            epoch_loss = running_loss / len(train_loader)
            for vfeats, vlabels in valid_loader:
                outputs = model(vfeats)
                _, preds = torch.max(outputs, 1)
            p = metrics.precision(preds, vlabels, 'multiclass', num_classes=4, average='macro')
            r = metrics.recall(preds, vlabels, 'multiclass', num_classes=4, average='macro')
            f = metrics.f1_score(preds, vlabels, 'multiclass', num_classes=4, average='macro')
            m = metrics.confusion_matrix(preds, vlabels, 'multiclass', num_classes=4)

            print(f'Loss: {epoch_loss:.4f} after {num_epoch + 1} epochs')
            print(m)
            print("slate, chyron, credit, none")
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        export = True  # TODO: deancahill 10/11/23 put this var in the run configuration
        if export:
            print("Exporting Data")
            export_data(predictions=preds, labels=vlabels, fname="results/oct11_results.csv",
                        model_name=train_loader.dataset.feature_model)

        model.load_state_dict(torch.load(best_model_params_path))
        print()
    return model, p, r, f


def export_data(predictions, labels, fname, model_name="vgg16"):
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

    with open(fname, 'a', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=["Model_Name", "Label", "Accuracy", "Precision", "Recall", "F1-Score"])
        writer.writeheader()
        for label, metrics in label_metrics.items():
            writer.writerow(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="root directory containing the vectors and labels to train on")
    parser.add_argument("featuremodel", help="feature vectors to use for training", choices=['vgg16', 'resnet50'],
                        default='vgg16')
    parser.add_argument("k_fold", help="the number of distinct dev sets to evaluate on", default=10)
    args = parser.parse_args()
    args.allow_guids = []
    args.block_guids = []
    k_fold_train(args.indir, int(args.k_fold), args.featuremodel, args.allow_guids, args.block_guids)
