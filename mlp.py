from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pandas as pd

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
clf = MLPClassifier(solver='lbfgs', random_state=1, max_iter=100)
clf.fit(X_train_dtm, y_train)
y_pred_class = clf.predict(X_test_dtm)

# calculate accuracy of class predictions
print metrics.accuracy_score(y_test, y_pred_class)

