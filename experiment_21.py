#!/usr/bin/python

"""
Experiment 21: Gender classification of English, Dutch, Italian, Spanish tweets using TFIDF.
- one line, all tweets
- cleaning: remove html 
			remove non-letter
			all small letters
			use stopwords
- classifier: SVC - linear, C=1
			  LinearSVC - loss='l2', penalty='l2', dual=False, tol=1e-3
"""

print(__doc__)


import os
import pandas as pd
import numpy as np
import re
import timeit

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sklearn import preprocessing

languages = ["english", "dutch", "italian", "spanish"]
datafiles = ["summary-english-truth.txt", \
			 "summary-dutch-truth.txt", \
			 "summary-italian-truth.txt", \
			 "summary-spanish-truth.txt"]

tasks = ["gender"]

languages = ["spanish"]
datafiles = ["summary-spanish-truth.txt"]

tasks = ["age"]



# SETTINGS:
n_words = 10000
n_folds = 10

def review_to_words(raw_review, language):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words(language))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


for language, datafile in zip(languages, datafiles):
	for task in tasks:
		start = timeit.default_timer()
		train = pd.read_csv(datafile, header=0, delimiter="\t", quoting=1)
		num_text = train["text"].size
		
		clean_train_reviews = []
		print "Processing text for %s." % language
		print "Looping through all text.\n" 

		for i in xrange( 0, num_text):
			clean_train_reviews.append( review_to_words( train["text"][i], language ) )
		
		print "Creating the bag of words.\n"

		vectorizer = TfidfVectorizer(analyzer = "word",\
									 tokenizer = None,      \
									 preprocessor = None,   \
									 stop_words = None,     \
									 max_features = n_words) 

		train_X = vectorizer.fit_transform(clean_train_reviews)
		train_X = train_X.toarray()
		train_y = train[task]
		train_y = np.array(train_y)
# 		import pdb; pdb.set_trace()		
# 		le = preprocessing.LabelEncoder()
# 		le.fit(list(set(train_y)))
# 		train_y = le.transform(train_y)
		print "Shape: ", train_X.shape

		print "Performing cross validation."
		print "*"*80
		print "Results for %s classification for %s." % (task, language)
# 		clf = svm.SVR(kernel='linear', C=1)
		clf = svm.SVC(kernel='linear', C=1)		
		scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
		print "#"*80
		print "SVR(kernel='linear', C=1) scores: "
		print "mean: ", scores.mean()
		print scores
		print "#"*80

# 		clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)   
# 		scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
# 		print "#"*80
# 		print "LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3) scores:"
# 		print "mean: ", scores.mean()
# 		print scores
# 		print "#"*80

		stop = timeit.default_timer()
		time_elapsed = stop - start
		print "Time elapsed: %s seconds" % time_elapsed