import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import IO, List

import torch
from torch import Tensor
from torchmetrics.functional import accuracy, precision, recall, f1_score, confusion_matrix


def validate(model, valid_loader, labelset, export_fname=None):
    model.eval()
    # valid_loader is currently expected to be a single batch
    vfeats, vlabels = next(iter(valid_loader))
    outputs = model(vfeats)
    _, preds = torch.max(outputs, 1)

    if not export_fname:
        export_f = sys.stdout
    else:
        path = Path(export_fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_f = open(path, 'w', encoding='utf8')
    p, r, f = export_validation_results(out=export_f, preds=preds, golds=vlabels,
                                        labelset=labelset, img_enc_name=valid_loader.dataset.img_enc_name)
    logging.info(f"Exported to {export_f.name}")
    return p, r, f


def export_validation_results(out: IO, preds: Tensor, golds: Tensor, labelset: List[str], img_enc_name: str):
    """Exports the data into a human-readable format.
    """

    label_metrics = defaultdict(dict)
    a_avg = accuracy(preds, golds, task='multiclass', num_classes=len(labelset), average='micro')
    p_avg = precision(preds, golds, task='multiclass', num_classes=len(labelset), average='micro')
    r_avg = recall(preds, golds, task='multiclass', num_classes=len(labelset), average='micro')
    f_avg = f1_score(preds, golds, task='multiclass', num_classes=len(labelset), average='micro')
    a = accuracy(preds, golds, task='multiclass', num_classes=len(labelset), average='none')
    p = precision(preds, golds, task='multiclass', num_classes=len(labelset), average='none')
    r = recall(preds, golds, task='multiclass', num_classes=len(labelset), average='none')
    f = f1_score(preds, golds, task='multiclass', num_classes=len(labelset), average='none')
    m = confusion_matrix(preds, golds, task='multiclass', num_classes=len(labelset))

    for i, label in enumerate(labelset):
        label_metrics[label] = {"Model_Name": img_enc_name,
                                "Label": label,
                                "Accuracy": a[i].item(),
                                "Precision": p[i].item(),
                                "Recall": r[i].item(),
                                "F1-Score": f[i].item()}
    writer = csv.DictWriter(out, fieldnames=["Model_Name", "Label", "Accuracy", "Precision", "Recall", "F1-Score"], lineterminator='\n')
    writer.writeheader()
    writer.writerow({"Model_Name": img_enc_name,
                     "Label": "Overall",
                     "Accuracy": a_avg.item(),
                     "Precision": p_avg.item(),
                     "Recall": r_avg.item(),
                     "F1-Score": f_avg.item()})
    for label, metrics in label_metrics.items():
        writer.writerow(metrics)
    out.write('\n\n')
    out.write("Confusion Matrix (cols = preds, rows = golds)\n")
    col_sums = torch.sum(m, dim=0)
    row_sums = torch.sum(m, dim=1)
    # longest_label_len with minimum 5 digits 
    lll = max(5, max(len(label) for label in labelset))
    out.write(f'{"":<{lll}},')
    for label in labelset:
        out.write(f'{label:>{lll}},')
    out.write(f'{"+":>{lll}}\n')
    for i, label in enumerate(labelset):
        out.write(f'{label:>{lll}},')
        for j in range(len(labelset)):
            out.write(f'{m[i][j]:>{lll}},')
        out.write(f'{row_sums[i]:>{lll}}\n')
    out.write(f'{"+":>{lll}},')
    for col_sum in col_sums:
        out.write(f'{col_sum:>{lll}},')
    if torch.sum(col_sums) == torch.sum(row_sums):
        out.write(f'{torch.sum(col_sums):>{lll}}\n')
    else:
        out.write(f'{"!!!":>{lll}}\n')
    return p_avg, r_avg, f_avg
