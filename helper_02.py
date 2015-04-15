#!/usr/bin/python


import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import re
import StringIO
import pydot
import timeit


from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from IPython.display import Image


languages = ["english", "dutch", "italian", "spanish"]

datafiles = [
			 "summary-english-truth.txt", \
			 "summary-dutch-truth.txt", \
			 "summary-italian-truth.txt", \
			 "summary-spanish-truth.txt"]

# tasks = ["gender", "age"]
tasks = ["extroverted", "stable", "agreeable", "open", "conscientious"]
num_features = 10000


def get_n_important_features(clf, n, feature_names):
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
		array_name = "x["+ str(b) + "]="+ c
		important_feature_list.append(array_name)
		feature_importances[b]=d	
	return important_feature_list

def showFeatures(features):
	n=1
	for i in features:
		print n,":", i
		n=n+1

def review_to_words(raw_review, language):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 

    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             

    # 4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words(language))                  

    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   

    # 6. Join the words back into one string separated by space, and return the result.
    return( " ".join( meaningful_words ))
    
    
def main():
	for lang, fileName in zip(languages, datafiles):	
		for task in tasks:
			print lang, task, ':'
			
			train = pd.read_csv(fileName, header=0, delimiter="\t", quoting=1)
			all_train_data_X = train["text"]	
		
			clean_train_reviews = []
			print("Looping through all text.\n")

			for i in xrange( 0, len(all_train_data_X)):
   				clean_train_reviews.append(review_to_words( all_train_data_X[i], lang ))


			print("Tfidf")

			vectorizer = TfidfVectorizer(analyzer = "word",   \
										 tokenizer = None,    \
										 preprocessor = None, \
							 			 stop_words = None,   \
							 			 max_features = num_features)  

			all_train_data_X = vectorizer.fit_transform(clean_train_reviews)
			feature_names = vectorizer.get_feature_names()
			all_train_data_X = all_train_data_X.toarray()
			
			all_train_data_y = train[task]
			all_train_data_y = np.array(all_train_data_y)			
			
			print "Starting decision tree."

			clf = tree.DecisionTreeClassifier(criterion="entropy")
			featureList = feature_names
			X=all_train_data_X
			y=all_train_data_y
			clf.fit(X,y)

			dot_data = StringIO.StringIO()
			tree.export_graphviz(clf, out_file=dot_data)
			graph = pydot.graph_from_dot_data(dot_data.getvalue())
			output_file_name = lang + "_"+ task + "_decision_tree.pdf"
			graph.write_pdf(output_file_name)
			Image(graph.create_png())

			imptFeatures = get_n_important_features(clf, 50, feature_names)
			
			print lang, task, ":"
			showFeatures(imptFeatures)

			output_file_name = lang + "_"+ task + "_function_words.txt"
			f = open(output_file_name, "w")
			f.write("\n".join(map(lambda x: str(x), imptFeatures)) + "\n")
			f.close()


if __name__ == "__main__":
	main()