import pickle
import numpy
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

##Read content of spambase.data ###
data = open('spambase.data', "r")
data=str(data.read())
data=data.split()

###Creating two seperate lists one containing emaildata and another whether that email was spam or not ####
email_data=[]
label_data=[]

for email in data:
	e=[float(i) for i in email.split(',')]
	l=len(e)
	email_data.append(e[:l-1])
	label_data.append(e[l-1])
##Deviding entire data into testing and training sets ######
### test_size is the percentage of events assigned to the test set (remainder go into training)


features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(email_data,label_data,test_size=0.1, random_state=42)


####Using classifiers to predict our accuracy #######
from sklearn.metrics import accuracy_score



### feature selection, because text is super high dimensional and
### can be really computationally chewy as a result
selector = SelectPercentile(f_classif, percentile=60)
selector.fit(features_train, labels_train)
features_train_transformed = selector.transform(features_train)
features_test_transformed  = selector.transform(features_test)


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
clf = tree.DecisionTreeClassifier(random_state=20, min_samples_split=150)
clf = clf.fit(features_train_transformed, labels_train)
pred=clf.predict(features_test_transformed)
acc=accuracy_score(pred,labels_test)
print 'acc : '+str(acc)