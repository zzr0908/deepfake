import torch
from sklearn.metrics import roc_auc_score
import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    auc = roc_auc_score(labels, predictions, average='macro', multi_class='ovo')
    return {"auc": auc, "accuracy": (predictions == labels).mean()}


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
