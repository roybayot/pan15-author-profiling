#!/usr/bin/python

"""
Experiment 21: Gender classification of English, Dutch, Italian, Spanish tweets 
			   using function words.
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