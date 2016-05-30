#!/usr/bin/python

import pickle
import numpy
from email_preprocess import *
from time import time #funny :D

##Deviding entire data into testing and training sets ######
### test_size is the percentage of events assigned to the test set (remainder go into training)

features_train, features_test, labels_train, labels_test = preprocess()

####Using classifiers to predict our accuracy #######

from sklearn.metrics import accuracy_score
features_train = features_train[:len(features_train)]
labels_train = labels_train[:len(labels_train)]

###NAIVE BAYES######

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

###SVM #####

"""Uncomment to try
from sklearn.svm import SVC
clf = SVC()
"""

###DECISION TREE ###
"""Uncomment to try
from sklearn import tree
clf = tree.DecisionTreeClassifier()
"""

t0 = time()
clf = clf.fit(features_train, labels_train)
pred=clf.predict(features_test)
acc=accuracy_score(pred,labels_test)
print 'acc : '+str(acc)+'	Time : '+str(time() - t0)

#########################################