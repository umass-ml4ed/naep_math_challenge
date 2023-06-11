import json
import ast
from model.posprocessing import ThresholdOptimizerNew
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
path = 'saved_models/bert_all_multiHead_l0_fold_9_8_loss0_0604_10pm/'
train_path = 'data/train_0604.csv'
path_val = path + 'val_predict.csv'
train = pd.read_csv(train_path)
val = pd.read_csv(path_val)
key = 'VH134067'
val = val[val['qid'] == key]
val_ids = val['id'].tolist()
val_info = train[train['id'].isin(val_ids)]
X = [ast.literal_eval(i.replace(' ',',').replace(',,',',')) for i in val['d'].tolist()]
X = np.array([x[0:2] for x in X])
y = np.array(val['label_str'].astype(int)) -1
sensitive_features = val_info['srace10'].astype(str).tolist()

#X                  = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],[10], [11], [12], [13], [14], [15], [16], [17], [18], [19]]
#y                  = [ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  1 ,  0 ,  0 ,  0 ,1 ,  0 ,  1 ,  1 ,  0,   0 ,  1 ,  1 ,  1 ,  0 ]
#sensitive_features = ["aa", "b", "c", "aa", "c", "aa", "b", "b", "c", "b","ab", "b", "c", "ab", "c", "ab", "b", "b", "c", "b"]
unmitigated_lr = LogisticRegression().fit(X, y)
unmitigated_lr = X[:,1]
postprocess_est = ThresholdOptimizerNew(
                   estimator=unmitigated_lr,
                   constraints="false_negative_rate_parity",
                   objective="balanced_accuracy_score",
                   prefit=True,
                   predict_method='predict_proba')
postprocess_est.fit(X, y, sensitive_features=sensitive_features)
z_before = unmitigated_lr.predict(X)
acc = sum(z_before==y)/len(y)
z = postprocess_est.predict(X, sensitive_features=sensitive_features)
acc_2 = sum(z==y)/len(y)
print(f"{acc} and {acc_2}")
print('done')