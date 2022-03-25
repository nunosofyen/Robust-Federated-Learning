import numpy as np
import os
from sklearn import metrics
import pandas as pd
import csv
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score, accuracy_score

def write_pre(pre, path):
    # pd.DataFrame(pre).to_csv(path, header=None, index=None)
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(pre)

def scoring(truth, pred):
    truth = truth.argmax(axis=1)
    pred = pred.argmax(axis=1)
    uar = recall_score(truth, pred, average='macro')
    acc = accuracy_score(truth, pred)
    confusion_mat = confusion_matrix(truth, pred, labels=list(range(7)))
    return confusion_mat, uar, acc
