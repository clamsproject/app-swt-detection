from tqdm import tqdm
import argparse
import csv
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from torchmetrics import functional as metrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from collections import defaultdict, Counter


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SWTDataset(Dataset):
    def __init__(self, in_dir, feature_model, allow_guids=[], block_guids=[]):
        self.feature_model = feature_model
        self.feat_dim = 4096 if feature_model == 'vgg16' else 2048  # for now, there are only two models
        self.labels = []
        self.vectors = []
        if not allow_guids:
            # TODO (krim @ 10/10/23): implement whitelisting
            for j in Path(in_dir).glob('*.json'):
                guid = j.with_suffix("").name
                if guid not in block_guids:
                    feature_vecs = np.load(Path(in_dir) / f"{guid}.{feature_model}.npy")
                    labels = json.load(open(Path(in_dir) / f"{guid}.json"))
                    for i, vec in enumerate(feature_vecs):
                        l = int_encode(labels['frames'][i]['label'])
                        self.labels.append(int_encode(labels['frames'][i]['label']))
                        self.vectors.append(torch.from_numpy(vec))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return self.vectors[i], self.labels[i]
    
    
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


def get_net(dim=4096):
    return nn.Sequential(
        nn.Linear(dim, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
        # nn.Softmax(dim=1)
    )


def train(dataset):
    train, valid = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train, batch_size=40, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)
    print(len(train), len(valid))
    loss = nn.CrossEntropyLoss(reduction="none")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(get_net(dataset.feat_dim), train_loader, valid_loader, loss, device)

def train_model(model, train_loader, valid_loader, loss_fn, device, num_epochs=25):
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

            print(f'Loss: {epoch_loss:.4f} after {num_epoch+1} epochs')
            print(f'Precision: {p:.4f} after {num_epoch+1} epochs')
            print(f'Recall: {r:.4f} after {num_epoch+1} epochs')
            print(f'F-1: {f:.4f} after {num_epoch+1} epochs')
            print(m)
            print("slate, chyron, credit, none")
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        export = True #TODO: deancahill 10/11/23 put this var in the run configuration
        if export:
            print("Exporting Data")
            export_data(predictions=preds, labels=vlabels, fname="results/oct11_results.csv", model_name=train_loader.dataset.dataset.feature_model)
                
        model.load_state_dict(torch.load(best_model_params_path))
    return model

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
        for label, metrics in label_metrics.items():
            writer.writerow(metrics)


            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="root directory containing the vectors and labels to train on")
    parser.add_argument("featuremodel", help="feature vectors to use for training", choices=['vgg16', 'resnet50'], default='vgg16')
    args = parser.parse_args()
    dataset = SWTDataset(args.indir, args.featuremodel)
    train(dataset)

