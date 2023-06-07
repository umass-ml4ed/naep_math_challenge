import os
import torch
import torch.nn.functional as F
from collections import defaultdict
import csv
from collections import defaultdict
import json
import pandas as pd
import collections
from sklearn.metrics import cohen_kappa_score
import numpy as np
import json
import utils.var as var
def safe_makedirs(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass

def OLL2_loss(num_classes, dist_matrix, labels, logits):
    probas = F.softmax(logits, dim=1)
    true_labels = [num_classes * [labels[k].item()] for k in range(len(labels))]
    label_ids = len(labels) * [[k for k in range(num_classes)]]
    distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in
                 range(len(labels))]
    distances_tensor = torch.tensor(distances, device='cuda:0', requires_grad=True)
    err = -torch.log(1 - probas) * abs(distances_tensor) ** 2
    loss = torch.sum(err, axis=1).mean()
    return loss

def select_best_result(metric):
    best_epoch = defaultdict(int)
    best = defaultdict(int)
    for epoch, value in metric.items():
        for key in value.keys():
            if 'eval' in key and 'eval_kappa' not in key and 'kappa' in key:
                key_all = key
                #key = key.split('_')
                key = key.replace('eval_','')
                key = key.replace('_kappa','')
                if best[key] < value[key_all]:
                    best[key] = value[key_all]
                    best_epoch[key] = epoch

    best_test = defaultdict(int)
    for key, e in best_epoch.items():
        best_test[key] = metric[e][f"test_{key}_kappa"]
    result = {'eval':best, 'test': best_test}
    return {'eval':best, 'test': best_test}
def itemwise_avg_kappa(df, start=1, epoch=None, alias='avg'):
    predict_name = df.columns
    metric = {}
    if epoch is None:
        epoch = 0
        for p in predict_name:
            if 'predict' in p:
                i = p.split('predict')
                i = i[1]
                if i != '' and int(i) > epoch:
                    epoch = int(i)
    epoch += 1
    column = ['predict'+str(i) for i in range(start,epoch)]
    df['avg'] = df[column].mean(axis=1)
    df['avg0'] = df['avg']
    qdf_temp = []
    all_qdf = []
    for key, qdf in df.groupby('qid'):
        qdf['avg'] = qdf['avg'].apply(round)
        all_qdf.append(qdf)
        if '2017' in key or '2019' in key:
            qdf_temp.append(qdf)
        key_n = var.QUESTION_TO_NAME[key]
        predictions = np.array(qdf['avg'].tolist())
        labels = np.array(qdf['label_str'].astype(int).tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        metric[f'{alias}_{key_n}_kappa'] = kappa
    if len(qdf_temp) > 0:
        qdf = pd.concat(qdf_temp)
        qdf['avg'] = qdf['avg'].apply(round)
        predictions = np.array(qdf['avg'].tolist())
        labels = np.array(qdf['label_str'].astype(int).tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        metric[f'{alias}_slope_kappa'] = kappa
        all_qdf.append(qdf)
    return pd.concat(all_qdf), metric






