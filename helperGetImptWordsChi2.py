#!/usr/bin/python
"""
Retrieving most important words from chi-squared.
""" 

from __future__ import print_function

import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import re

from time import time
from optparse import OptionParser
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn import svm

# SETTINGS:
n_words = 10000
n_folds = 10
n_top_words = 50

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.7f\t%-15s\t\t%.7f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

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
num_text = train["text"].size
# start = timeit.default_timer()

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
print("Looping through all text.\n")

for i in xrange( 0, num_text):
   # Call our function for each one, and add the result to the list of
   # clean reviews
   clean_train_reviews.append( review_to_words( train["text"][i] ) )

print("Creating the bag of words...\n")

vectorizer = TfidfVectorizer(analyzer = "word",\
						tokenizer = None,      \
						preprocessor = None,   \
						stop_words = None,     \
						max_features = n_words) 

train_X = vectorizer.fit_transform(clean_train_reviews)
train_X = train_X.toarray()
train_y = train["gender"]
# train_y = np.array(train_y)


	
print("Extracting %d best features by a chi-squared test" % n_top_words)
t0 = time()
ch2 = SelectKBest(chi2, n_top_words)
# for a in train_y:
# 	print(a)
# import pdb; pdb.set_trace()
train_X = ch2.fit_transform(train_X, train_y.values)
# test_X = ch2.transform(test_X)
print("done in %fs" % (time() - t0))
print()

print("cross validation")

str = "#"*80

print(str)
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
print("SVC(kernel='linear', C=1) scores: ")
print("mean: %f" % scores.mean())
print(scores)
print(str)


# clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
# print('_' * 80)
# print("Training: ")
# scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
# print("LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3) scores:")
# print("mean: %f" % scores.mean())
# print(scores)
# print(clf)
# t0 = time()

# clf = LinearSVC(loss='l2', penalty='l1', dual=False, tol=1e-3)
# print(str)
# print("Training: ")
# scores = cross_validation.cross_val_score( clf, train_X, train_y, cv=n_folds)
# print("LinearSVC(loss='l2', penalty='l1', dual=False, tol=1e-3) scores:")
# print("mean: %f" % scores.mean())
# print(scores)
# print(clf)
# t0 = time()


show_most_informative_features(vectorizer, clf, n=n_top_words)
