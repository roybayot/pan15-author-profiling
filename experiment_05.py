#!/usr/bin/python

"""
Experiment 05: Gender classification of English tweets based on ngrams of POS tags.
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
# SETTINGS:
n_words = 10000
n_folds = 10
tt = TreeTagger(encoding='latin-1',language='english')

		
def review_to_words(raw_review):
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
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


# MAIN SCRIPT
train = pd.read_csv("summary-english-truth.txt", header=0, delimiter="\t", quoting=1)
train_reviews = train["text"]
num_text = train_reviews.size

start = timeit.default_timer()

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
print "Looping through all text.\n" 

for i in xrange( 0, num_text):
   # Call our function for each one, and add the result to the list of
   # clean reviews
   clean_train_reviews.append( review_to_words( train["text"][i] ) )

train_reviews_pos_tags = []
for line in clean_train_reviews:
	a = tt.tag(line)
	a = [col[1] for col in a]
	pos_line = " ".join(a)
	train_reviews_pos_tags.append(pos_line)


bigram_vectorizer = CountVectorizer(ngram_range=(1, 3), \
									min_df=1)
									
train_X = bigram_vectorizer.fit_transform(train_reviews_pos_tags).toarray()
train_y = train["gender"]
print "Shape: ", train_X.shape

print "cross validation"
# clf = svm.SVC(kernel='linear', C=1).fit(train_X, train_y)
print "#"*80
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
print "SVC(kernel='linear', C=1) scores: "
print "mean: ", scores.mean()
print scores
print "#"*80

clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)   
scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
print "LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3) scores:"
print "mean: ", scores.mean()
print scores
print "#"*80

stop = timeit.default_timer()
time_elapsed = stop - start
print time_elapsed

