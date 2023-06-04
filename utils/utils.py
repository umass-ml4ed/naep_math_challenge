import os
import torch
import torch.nn.functional as F
from collections import defaultdict
import json
#import utils.var as var
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

# with open('../saved_models/epochmetrics.json', 'r') as f:
#     metric = json.load(f)
# result = select_best_result(metric)
# for key, value in result.items():
#     for k, v in value.items():
#         print(f'{key} {k}: {v}')
#     print('===================')






