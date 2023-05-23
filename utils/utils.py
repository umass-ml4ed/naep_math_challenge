import os
import torch
import torch.nn.functional as F
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


