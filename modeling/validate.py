import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import IO, List

import torch
from torch import Tensor
from torchmetrics import functional as metrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score


def validate(model, valid_loader, labelset, export_fname=None):
    model.eval()
    # valid_loader is currently expected to be a single batch
    vfeats, vlabels = next(iter(valid_loader))
    outputs = model(vfeats)
    _, preds = torch.max(outputs, 1)
    p = metrics.precision(preds, vlabels, 'multiclass', num_classes=len(labelset), average='macro')
    r = metrics.recall(preds, vlabels, 'multiclass', num_classes=len(labelset), average='macro')
    f = metrics.f1_score(preds, vlabels, 'multiclass', num_classes=len(labelset), average='macro')
    # m = metrics.confusion_matrix(preds, vlabels, 'multiclass', num_classes=len(labelset))

    if not export_fname:
        export_f = sys.stdout
    else:
        path = Path(export_fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_f = open(path, 'w', encoding='utf8')
    export_validation_results(out=export_f, preds=preds, golds=vlabels,
                              labelset=labelset, img_enc_name=valid_loader.dataset.img_enc_name)
    logging.info(f"Exported to {export_f.name}")
    return p, r, f


def export_validation_results(out: IO, preds: Tensor, golds: Tensor, labelset: List[str], img_enc_name: str):
    """Exports the data into a human-readable format.
    """

    label_metrics = defaultdict(dict)

    for i, label in enumerate(labelset):
        pred_labels = torch.where(preds == i, 1, 0)
        true_labels = torch.where(golds == i, 1, 0)
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
