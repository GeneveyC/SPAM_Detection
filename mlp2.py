from representation_data import *
from learning_curve import *

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit

import pandas as pd
import matplotlib.pyplot as plt

def test(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linsoace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

Mt = 4000

# ratio spam : count of word s, if s > 10, we put s in dataset
rs = 10
# ration  ham : count of word h, if h > 10, we put h in dataset
rh = 0.2

X_train, X_test, y_train, y_test, feature_names = get_data(option=1, rs=rs, rh=rh, random_state_=1)

#mlp = MLPClassifier(solver='lbfgs', random_state=1, max_iter=100)
#mlp.fit(X_train, y_train)

#y_pred_class = mlp.predict(X_test)
#print metrics.accuracy_score(y_test, y_pred_class)

#y_pred_prob = mlp.predict_proba(X_test)[:,1]
#print metrics.roc_auc_score(y_test, y_pred_prob)

title = "Learning Curves (MLP)"
cv = ShuffleSplit(n_splits=100, test_size=0.4, random_state=1)

estimator = MLPClassifier(solver='lbfgs', random_state=1, max_iter=100)
print estimator
plot_learning_curve(estimator, title, X_train, y_train, ylim=None, cv=cv,n_jobs=1)
plt.show()
