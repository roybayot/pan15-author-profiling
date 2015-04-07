#!/usr/bin/python


import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import re
import StringIO
import pydot


from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from IPython.display import Image



num_features = 10000

train = pd.read_csv("summary-english-truth.txt", header=0, delimiter="\t", quoting=1)


num_text = train["text"].size
print(num_text)
all_train_data_X = train["text"]
all_train_data_y = train["gender"]

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

def get_n_important_features(n, feature_names):
	if n > len(feature_names):
		print "invalid number"
		sys.exit()
		
	important_feature_list = []
	feature_importances = list(clf.feature_importances_)    
	for i in range(n):
		a = max(feature_importances)
		b = feature_importances.index(a)
		c = feature_names[b]
# 		import pdb; pdb.set_trace()
		d = min(feature_importances)
		important_feature_list.append(c)
		feature_importances[b]=d	
	return important_feature_list
	
def showFeatures(features):
	n=1
	for i in features:
		print n,":", i
		n=n+1
	
clean_train_reviews = []
print("Looping through all text.\n")

for i in xrange( 0, len(all_train_data_X)):
   clean_train_reviews.append( review_to_words( all_train_data_X[i] ) )


print("Count Vectorizer")

vectorizer = CountVectorizer(analyzer = "word",   \
							 tokenizer = None,    \
							 preprocessor = None, \
							 stop_words = None,   \
							 max_features = num_features)  

all_train_data_X = vectorizer.fit_transform(clean_train_reviews)
feature_names = vectorizer.get_feature_names()
all_train_data_X = all_train_data_X.toarray()

clf = tree.DecisionTreeClassifier(criterion="entropy")
featureList = feature_names
X=all_train_data_X
y=all_train_data_y
clf.fit(X,y)

dot_data = StringIO.StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("TweetGenderDecisionTree.pdf")
Image(graph.create_png())

imptFeatures = get_n_important_features(50, feature_names)
showFeatures(imptFeatures)
