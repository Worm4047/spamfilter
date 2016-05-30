#Spam Filter.####

##1. Dataset and goal of project

####Goal ####

The main purpose of project is develop the machine learning algorithm to detect spam Emails from dataset.
The "spam" concept is diverse: advertisements for products/web
        sites, make money fast schemes, chain letters, pornography...
	Our collection of spam e-mails came from our postmaster and 
	individuals who had filed spam.  Our collection of non-spam 
	e-mails came from filed work and personal e-mails, and hence
	the word 'george' and the area code '650' are indicators of 
	non-spam.  These are useful when constructing a personalized 
	spam filter.  One would either have to blind such non-spam 
	indicators or get a very wide collection of non-spam to 
	generate a general purpose spam filter.

####Dataset used 
Spambase dataset that has been divided into email contents(email_words.pkl) and category of email(email_category.pkl) dataset. 

####Steps to use
	All you have to do is set your email content in one file and name it as - email_words.pkl
	set your email category(spam/not spam) using binary number to signify each in file - email_category.pkl
	and you are good to go.

	Note : To install required module paste the following commands into terminal - pip install -r requirements.txt

####Example of current data 
There's example of one email_words.pkl: 

"aS' sbaile2 nonprivilegedpst 1 txu energi trade compani 2 bp capit energi fund lp may be subject to mutual termin 2 nobl gas market inc 3 puget sound energi inc 4 virginia power energi market inc 5 t boon picken may be subject to mutual termin 5 neumin product co 6 sodra skogsagarna ek for probabl an ectric counterparti 6 texaco natur gas inc may be book incorrect for texaco inc financi trade 7 ace capit re oversea ltd 8 nevada power compani 9 prior energi corpor 10 select energi inc origin messag from tweed sheila sent thursday januari 31 2002 310 pm to   subject pleas send me the name of the 10 counterparti that we are evalu thank' "

email_category contains whether a particular email is spam or not 

	aI0 = spam

	aI1 = not a spam

##3. Picking an algorithm
I tried the Naive Bayes, SVM and Decision Trees algorithms. 

####All results of examination I included in the following table


		**Naive Bayes**		 Acccuracy	:	0.971937808115    	Time : 1.82899999619

		**Decision Trees**	 Acccuracy  : 	0.985589685248		Time : 61.3320000172

		**SVM**				 Accuracy   :   0.800568828214		Time : 301.3390120329

####Chosen algorithm
Based on best performance level I'd pick Decision Tree as a final algorithm.
But considering the time taken i'd rather stick with Niave Bayes .
You'd want your spam filter to be fast wouldn't you ?

##4. Tune the algorithm
####Reasons for algorithm tuning
The main reason is get better results from algorithm. Parameters of ML classifiers have a big influence in output results. 
The purpose of tuning is to find best sets of parameters for particular dataset.

####GridSearchCV
I apply GridSearchCV to tune the following parameters

	|Parameter          	Settings for investigation
	|min_samples_split	 	[1-59]                    
	|random_state	     	[1-50]                    


As a result i obtained best performance when random_state=15, min_samples_split=3
but i got even better performance with naive bayes so i used default settings  . 


####Outcome
The spamfilter filters emails with an accuracy of 97.2% which quite a significant number granted the type of data that was available for testing and training.