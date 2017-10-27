from representation_data import *
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import pandas as pd
import matplotlib.pyplot as plt

Mt = 4000

# ratio spam : count of word s, if s > 10, we put s in dataset
rs = 10
# ration  ham : count of word h, if h > 10, we put h in dataset
rh = 0.2

X_train, X_test, y_train, y_test, feature_names = get_data(option=1, rs=rs, rh=rh, random_state_=1)

mlp = MLPClassifier(solver='lbfgs', random_state=1, max_iter=100)
mlp.fit(X_train, y_train)

y_pred_class = mlp.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

y_pred_prob = mlp.predict_proba(X_test)[:,1]
print metrics.roc_auc_score(y_test, y_pred_prob)

# learning_curve
cv = ShuffleSplit(n_splits=100, test_size=0.4, random_state=1)
print "Learning curve"
train_size, train_scores, test_scores = learning_curve(mlp, X_train, y
