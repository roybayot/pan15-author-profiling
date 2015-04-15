#!/usr/bin/python

"""
Experiment 15: Personality classification of English tweets based on ngrams of POS tags.
- one line, all tweets
- cleaning
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

from treetagger import TreeTagger

from sklearn import preprocessing

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
    
    
languages = ["english", "dutch", "italian", "spanish"]
datafiles = ["summary-english-truth.txt", \
			 "summary-dutch-truth.txt", \
			 "summary-italian-truth.txt", \
			 "summary-spanish-truth.txt"]
personality_dimensions = ["extroverted", "stable", "agreeable", "open", "conscientious"]

ngrams_range = [(1,1), (2,2), (3,3), (1,2), (1,3)]

for lang, datafile in zip(languages, datafiles):
	for dimension in personality_dimensions:
		for current_ngram in ngrams_range:
			start = timeit.default_timer()
			train = pd.read_csv(datafile, header=0, delimiter="\t", quoting=1)
			num_text = train["text"].size

			clean_train_reviews = []
			print "Processing text for %s." % lang
			print "Looping through all text.\n" 

			for i in xrange( 0, num_text):
				clean_train_reviews.append( review_to_words( train["text"][i], lang ) )

			tt = TreeTagger(encoding='latin-1',language=lang)
			train_reviews_pos_tags = []
			for line in clean_train_reviews:
				a = tt.tag(line)
				a = [col[1] for col in a]
				pos_line = " ".join(a)
				train_reviews_pos_tags.append(pos_line)

			bigram_vectorizer = CountVectorizer(ngram_range=current_ngram, \
												min_df=1)
									
			train_X = bigram_vectorizer.fit_transform(train_reviews_pos_tags).toarray()
			
			train_y = train[dimension]
			train_y = np.array(train_y)
			le = preprocessing.LabelEncoder()
			le.fit(list(set(train_y)))
			train_y = le.transform(train_y)
			print "Shape: ", train_X.shape

			print "Performing cross validation."
			print "*"*80
			print "Results for %s classification for %s for ngram range %s." % (dimension, lang, current_ngram)
# 			clf = svm.SVR(kernel='linear', C=1)
			clf = svm.SVC(kernel='linear', C=1)		
			scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
			print "#"*80
			print "SVR(kernel='linear', C=1) scores: "
			print "mean: ", scores.mean()
			print scores
			print "#"*80

# 			clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)   
# 			scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
# 			print "#"*80
# 			print "LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3) scores:"
# 			print "mean: ", scores.mean()
# 			print scores
# 			print "#"*80

			stop = timeit.default_timer()
			time_elapsed = stop - start
			print "Time elapsed: %s seconds" % time_elapsed