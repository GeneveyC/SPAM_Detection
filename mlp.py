from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# url of dataset
url = "http://www.irisa.fr/dyliss/public/fcoste/data/pub/sms.tsv"

sms = pd.read_table(url, header=None, names=["label","message"])
sms["label_num"] = sms.label.map({"ham":0, "spam":1})
X = sms.message
y = sms.label_num
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

# instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)

# transform testing data into a document-term matrix
X_test_dtm = vect.transform(X_test)

# Create an neural network with activation relu
clf = MLPClassifier(solver='lbfgs', random_state=1)

#clf.fit(X_train_dtm, y_train)
#y_pred_class = clf.predict(X_test_dtm)

# calculate accuracy of class predictions
#print metrics.accuracy_score(y_test, y_pred_class)


# calculate predicted probabilities for X_test_dtm
#y_pred_prob = clf.predict_proba(X_test_dtm)[:,1]

# calculate AUC
#print metrics.roc_auc_score(y_test, y_pred_prob)

# learning curve
cv = ShuffleSplit(n_splits=100, test_size=0.4, random_state=1)
print("Learning curve")
train_sizes, train_scores, test_scores = learning_curve(clf, X_train_dtm.toarray(), y_train.as_matrix(), cv=None)

plt.figure()
plt.title("Learning Curves (MLP)")
plt.xlabel("Training examples")
plt.ylabel("Score")

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()
