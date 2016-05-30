import pickle
import numpy
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from email_preprocess import *

##Deviding entire data into testing and training sets ######
### test_size is the percentage of events assigned to the test set (remainder go into training)


features_train, features_test, labels_train, labels_test = preprocess()

####Using classifiers to predict our accuracy #######
from sklearn.metrics import accuracy_score
features_train = features_train[:len(features_train)/3]
labels_train = labels_train[:len(labels_train)/3]

###NAIVE BAYES######
"""Uncomment to try
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)
"""

###SVM #####
"""Uncomment to try
from sklearn.svm import SVC
clf = SVC(C=1.0, kernel="rbf")
clf.fit(features, labels)
"""

###DECISION TREE ###
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred=clf.predict(features_test)
acc=accuracy_score(pred,labels_test)
print 'acc : '+str(acc)