from tqdm import tqdm
import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from torchmetrics import functional as metrics

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SWTDataset(Dataset):
    def __init__(self, feat_dim, labels, vectors, allow_guids=[]):
        self.feat_dim = feat_dim 
        self.labels = labels
        self.vectors = vectors
        # TODO (krim @ 10/10/23): implement whitelisting

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return self.vectors[i], self.labels[i]


def get_guids(dir, block_guids=[]):
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


def get_net(dim=4096):
    return nn.Sequential(
        nn.Linear(dim, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
        # nn.Softmax(dim=1)
    )


def split_dataset(indir, valid_guid, feature_model):
    feat_dim = 4096 if feature_model == 'vgg16' else 2048  # for now, there are only two models
    train_vectors = []
    train_labels = []
    valid_vectors = []
    valid_labels = []
    for j in Path(indir).glob('*.json'):
        guid = j.with_suffix("").name
        feature_vecs = np.load(Path(indir) / f"{guid}.{feature_model}.npy")
        labels = json.load(open(Path(indir) / f"{guid}.json"))
        if guid == valid_guid:
            for i, vec in enumerate(feature_vecs):
                l = int_encode(labels['frames'][i]['label'])
                valid_labels.append(int_encode(labels['frames'][i]['label']))
                valid_vectors.append(torch.from_numpy(vec))   
        else:
             for i, vec in enumerate(feature_vecs):
                l = int_encode(labels['frames'][i]['label'])
                train_labels.append(int_encode(labels['frames'][i]['label']))
                train_vectors.append(torch.from_numpy(vec))
    train = SWTDataset(feat_dim, train_labels, train_vectors)
    valid = SWTDataset(feat_dim, valid_labels, valid_vectors)
    return train, valid


def train(indir, k_fold, featuremodel):
    guids = get_guids(indir)
    p_scores = []
    r_scores = []
    f_scores = []
    for i in range(k_fold):
        valid_guid = guids[i]
        train, valid = split_dataset(indir, valid_guid, featuremodel)
        train_loader = DataLoader(train, batch_size=40, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)
        print(len(train), len(valid))
        loss = nn.CrossEntropyLoss(reduction="none")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, p, r, f = train_model(get_net(train.feat_dim), train_loader, valid_loader, loss, device, valid_guid)
        p_scores.append(p)
        r_scores.append(r)
        f_scores.append(f)
    print_scores(p_scores, r_scores, f_scores)
    

def print_scores(p_scores, r_scores, f_scores):
    max = f_scores.index(max(f_scores))
    min = f_scores.index(min(f_scores))
    print("Highest p/r/f is")
    print(f'\tprecision = {p_scores[max]}')
    print(f'\trecall = {r_scores[max]}')
    print(f'\tf-1 = {f_scores[max]}')
    print("Lowest p/r/f is")
    print(f'\tprecision = {p_scores[max]}')
    print(f'\trecall = {r_scores[max]}')
    print(f'\tf-1 = {f_scores[max]}')
    print("Mean p/r/f is")
    print(f'\tprecision = {sum(p_scores)/len(p_scores)}')
    print(f'\trecall = {sum(r_scores)/len(r_scores)}')
    print(f'\tf-1 = {sum(f_scores)/len(f_scores)}')


def train_model(model, train_loader, valid_loader, loss_fn, device, valid_guid, num_epochs=25):
    print(f'Using the guid {valid_guid} to evaluate')

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
            print(m)
            print("slate, chyron, credit, none")
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        model.load_state_dict(torch.load(best_model_params_path))
        print()
    return model, p, r, f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="root directory containing the vectors and labels to train on")
    parser.add_argument("featuremodel", help="feature vectors to use for training", choices=['vgg16', 'resnet50'], default='vgg16')
    parser.add_argument("k_fold", help="the number of distinct dev sets to evaluate on", default=10)
    args = parser.parse_args()
    train(args.indir, args.k_fold, args.featuremodel)