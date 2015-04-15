#!/usr/bin/python

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

from treetagger import TreeTagger

files_for_tasks = {
'english':{
			'age':	[
			 			'english_age_X_tfidf.csv', 
						'english_age_X_function_words.csv', 
						'english_age_X_stylistic_features.csv', 
						'english_age_X_unigrams.csv', 
						'english_age_X_bigrams.csv'			 
					],
			'gender': [
			 			'english_gender_X_tfidf.csv', 
						'english_gender_X_function_words.csv', 
						'english_gender_X_stylistic_features.csv', 
						'english_gender_X_unigrams.csv', 
						'english_gender_X_bigrams.csv'
					  ],
			'extroverted':[
					 			'english_age_X_tfidf.csv', 
								'english_extroverted_X_function_words.csv', 
								'english_age_X_stylistic_features.csv', 
								'english_age_X_unigrams.csv', 
								'english_age_X_bigrams.csv'			 
			
							], 
			'stable':[
					 			'english_age_X_tfidf.csv', 
								'english_stable_X_function_words.csv', 
								'english_age_X_stylistic_features.csv', 
								'english_age_X_unigrams.csv', 
								'english_age_X_bigrams.csv'			 
			
							], 
			'agreeable':[
					 			'english_age_X_tfidf.csv', 
								'english_agreeable_X_function_words.csv', 
								'english_age_X_stylistic_features.csv', 
								'english_age_X_unigrams.csv', 
								'english_age_X_bigrams.csv'			 
			
							], 
			'open':[
					 			'english_age_X_tfidf.csv', 
								'english_open_X_function_words.csv', 
								'english_age_X_stylistic_features.csv', 
								'english_age_X_unigrams.csv', 
								'english_age_X_bigrams.csv'			 			
							], 
			'conscientious':[
					 			'english_age_X_tfidf.csv', 
								'english_conscientious_X_function_words.csv', 
								'english_age_X_stylistic_features.csv', 
								'english_age_X_unigrams.csv', 
								'english_age_X_bigrams.csv'			 				
							]					  
			},
'dutch':{
			'age': [
			 			'dutch_age_X_tfidf.csv', 
						'dutch_age_X_function_words.csv', 
						'dutch_age_X_stylistic_features.csv', 
						'dutch_age_X_unigrams.csv', 
						'dutch_age_X_bigrams.csv'			
					],
			'gender':[
			 			'dutch_gender_X_tfidf.csv', 
						'dutch_gender_X_function_words.csv', 
						'dutch_gender_X_stylistic_features.csv', 
						'dutch_gender_X_unigrams.csv', 
						'dutch_gender_X_bigrams.csv'
			
					 ],
			'extroverted':[
				 			'dutch_age_X_tfidf.csv', 
							'dutch_extroverted_X_function_words.csv', 
							'dutch_age_X_stylistic_features.csv', 
							'dutch_age_X_unigrams.csv', 
							'dutch_age_X_bigrams.csv'					
							], 
			'stable':[
				 			'dutch_age_X_tfidf.csv', 
							'dutch_stable_X_function_words.csv', 
							'dutch_age_X_stylistic_features.csv', 
							'dutch_age_X_unigrams.csv', 
							'dutch_age_X_bigrams.csv'						
							], 
			'agreeable':[
				 			'dutch_age_X_tfidf.csv', 
							'dutch_agreeable_X_function_words.csv', 
							'dutch_age_X_stylistic_features.csv', 
							'dutch_age_X_unigrams.csv', 
							'dutch_age_X_bigrams.csv'						
							], 
			'open':[
				 			'dutch_age_X_tfidf.csv', 
							'dutch_open_X_function_words.csv', 
							'dutch_age_X_stylistic_features.csv', 
							'dutch_age_X_unigrams.csv', 
							'dutch_age_X_bigrams.csv'						
							], 
			'conscientious':[
				 			'dutch_age_X_tfidf.csv', 
							'dutch_conscientious_X_function_words.csv', 
							'dutch_age_X_stylistic_features.csv', 
							'dutch_age_X_unigrams.csv', 
							'dutch_age_X_bigrams.csv'							
							]					  
					 
			},
'italian':{
			'age':[
			 			'italian_age_X_tfidf.csv', 
						'italian_age_X_function_words.csv', 
						'italian_age_X_stylistic_features.csv', 
						'italian_age_X_unigrams.csv', 
						'italian_age_X_bigrams.csv'
				  ],
			'gender':[
			 			'italian_gender_X_tfidf.csv', 
						'italian_gender_X_function_words.csv', 
						'italian_gender_X_stylistic_features.csv', 
						'italian_gender_X_unigrams.csv', 
						'italian_gender_X_bigrams.csv'
					 ],
			'extroverted':[
				 			'italian_age_X_tfidf.csv', 
							'italian_extroverted_X_function_words.csv', 
							'italian_age_X_stylistic_features.csv', 
							'italian_age_X_unigrams.csv', 
							'italian_age_X_bigrams.csv'
							], 
			'stable':[
				 			'italian_age_X_tfidf.csv', 
							'italian_stable_X_function_words.csv', 
							'italian_age_X_stylistic_features.csv', 
							'italian_age_X_unigrams.csv', 
							'italian_age_X_bigrams.csv'
							], 
			'agreeable':[
				 			'italian_age_X_tfidf.csv', 
							'italian_agreeable_X_function_words.csv', 
							'italian_age_X_stylistic_features.csv', 
							'italian_age_X_unigrams.csv', 
							'italian_age_X_bigrams.csv'

							], 
			'open':[
				 			'italian_age_X_tfidf.csv', 
							'italian_open_X_function_words.csv', 
							'italian_age_X_stylistic_features.csv', 
							'italian_age_X_unigrams.csv', 
							'italian_age_X_bigrams.csv'			
							], 
			'conscientious':[
				 			'italian_age_X_tfidf.csv', 
							'italian_conscientious_X_function_words.csv', 
							'italian_age_X_stylistic_features.csv', 
							'italian_age_X_unigrams.csv', 
							'italian_age_X_bigrams.csv'							
							]					  

			},
'spanish':{
			'age':[
			 			'spanish_age_X_tfidf.csv', 
						'spanish_age_X_function_words.csv', 
						'spanish_age_X_stylistic_features.csv', 
						'spanish_age_X_unigrams.csv', 
						'spanish_age_X_bigrams.csv'
				  ],
			'gender':[
			 			'spanish_gender_X_tfidf.csv', 
						'spanish_gender_X_function_words.csv', 
						'spanish_gender_X_stylistic_features.csv', 
						'spanish_gender_X_unigrams.csv', 
						'spanish_gender_X_bigrams.csv'
					 ],
			'extroverted':[
				 			'spanish_age_X_tfidf.csv', 
							'spanish_extroverted_X_function_words.csv', 
							'spanish_age_X_stylistic_features.csv', 
							'spanish_age_X_unigrams.csv', 
							'spanish_age_X_bigrams.csv'			
							], 
			'stable':[
				 			'spanish_age_X_tfidf.csv', 
							'spanish_stable_X_function_words.csv', 
							'spanish_age_X_stylistic_features.csv', 
							'spanish_age_X_unigrams.csv', 
							'spanish_age_X_bigrams.csv'						
							], 
			'agreeable':[
				 			'spanish_age_X_tfidf.csv', 
							'spanish_agreeable_X_function_words.csv', 
							'spanish_age_X_stylistic_features.csv', 
							'spanish_age_X_unigrams.csv', 
							'spanish_age_X_bigrams.csv'						
							], 
			'open':[
				 			'spanish_age_X_tfidf.csv', 
							'spanish_agreeable_X_function_words.csv', 
							'spanish_age_X_stylistic_features.csv', 
							'spanish_age_X_unigrams.csv', 
							'spanish_age_X_bigrams.csv'			
							], 
			'conscientious':[
				 			'spanish_age_X_tfidf.csv', 
							'spanish_conscientious_X_function_words.csv', 
							'spanish_age_X_stylistic_features.csv', 
							'spanish_age_X_unigrams.csv', 
							'spanish_age_X_bigrams.csv'							
							]					  					 
			}
}


languages = ["english", "dutch", "italian", "spanish"]
datafiles = ["summary-english-truth.txt", \
			 "summary-dutch-truth.txt", \
			 "summary-italian-truth.txt", \
			 "summary-spanish-truth.txt"]
tasks = ["extroverted", "stable", "agreeable", "open", "conscientious"]

# languages = ["english"]
# datafiles = ["summary-english-truth.txt"]
# tasks = ["extroverted"]

n_folds = 10


results = []

for task in tasks:
	for lang, datafile in zip(languages,datafiles):
		start = timeit.default_timer()
# 		language_key = lang
		files = files_for_tasks[lang][task]
		X = []
		for file in files:
			X.append(pd.read_csv(file, header=None))
		X = np.concatenate(X, axis=1)
	
		train = pd.read_csv(datafile, header=0, delimiter="\t", quoting=1)
		y = train[task]
		
		y = np.array(y)
# 		le = preprocessing.LabelEncoder()
# 		le.fit(list(set(y)))
# 		y = le.transform(y)

		clf = svm.SVR(kernel='linear', C=1)		
		scores = cross_validation.cross_val_score( clf, X, y, cv=n_folds,scoring='mean_squared_error')
		print "#"*80
		print "Results for ", lang, task, "regression using all features."
		print "SVR(kernel='linear', C=1) scores: "
		print "mean: ", scores.mean()
		print scores
		print "#"*80

		stop = timeit.default_timer()
		time_elapsed = stop - start
		print "Time elapsed: %s seconds" % time_elapsed
		results.append({lang: {task: scores.mean()}})

		

		