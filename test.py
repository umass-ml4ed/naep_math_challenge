from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression
import numpy as np
X                  = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],[10], [11], [12], [13], [14], [15], [16], [17], [18], [19]]
y                  = [ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  1 ,  0 ,  0 ,  0 ,1 ,  0 ,  1 ,  1 ,  0,   0 ,  1 ,  1 ,  1 ,  0 ]
sensitive_features = ["aa", "b", "c", "aa", "c", "aa", "b", "b", "c", "b","ab", "b", "c", "ab", "c", "ab", "b", "b", "c", "b"]
unmitigated_lr = LogisticRegression().fit(X, y)
postprocess_est = ThresholdOptimizer(
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