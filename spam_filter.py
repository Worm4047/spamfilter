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
	l=len(email)
	email_data.append(email[:l-1])
	label_data.append(email[l-1])

##Deviding entire data into testing and training sets ######
### test_size is the percentage of events assigned to the test set (remainder go into training)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(email_data,label_data,test_size=0.1, random_state=42)


