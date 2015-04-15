#!/usr/bin/python

"""
This script file generates the feature files for GENDER classification using 
tfidf, function words, stylistic features, POS tags for unigrams and bigrams 
for english, dutch, italian, and spanish.
"""

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

languages = ["english", "dutch", "italian", "spanish"]

datafiles = ["summary-english-truth.txt", \
			 "summary-dutch-truth.txt", \
			 "summary-italian-truth.txt", \
			 "summary-spanish-truth.txt"]

# tasks = ["age"]
tasks = ["extroverted", "stable", "agreeable", "open", "conscientious"]

function_words_dict = {
'english':{
			'age':	[ 
						"co",
						"wanna",
						"us",
						"haha",
						"username",
						"fitbit",
						"et",
						"bowl",
						"academia",
						"bitch",
						"happened",
						"even",
						"year",
						"reach",
						"free",
						"times",
						"speech",
						"top",
						"add",
						"social",
						"think",
						"nothing",
						"financial",
						"pop",
						"inspiring",
						"lil",
						"complicated",
						"aa"
						],
			'gender': [ 
						"close",
						"love",
						"mention",
						"co",
						"wife",
						"lanka",	
						"believe",
						"video",
						"cute",
						"phone",
						"le",
						"day",
						"urban",
						"round",
						"thank",
						"bird",
						"wouldn",
						"aa"
						],
			'extroverted':[
							"co",
							"username",
							"million",
							"liked",
							"facebook",
							"last",
							"better",
							"de",
							"music",
							"around",
							"let",
							"book",
							"happy",
							"friends",
							"used",
							"inside",
							"really",
							"di",
							"work",
							"google",
							"opinion",
							"phd",
							"racist",
							"things",
							"forget",
							"via",
							"need",
							"nice",
							"http",
							"application",
							"slides",
							"sign",
							"sun",
							"sell",
							"years",
							"latest",
							"starbucks",
							"jullie",
							"interesante",
							"minute",
							"screen",
							"model",
							"shirt",
							"ziglar",
							], 
			'stable':[
							"like",
							"re",
							"god",
							"computer",
							"cause",
							"android",
							"follow",
							"waiting",
							"well",
							"school",
							"ever",
							"rock",
							"part",
							"photo",
							"want",
							"years",
							"mind",
							"need",
							"bring",
							"original",
							"says",
							"back",
							"colleagues",
							"last",
							"finally",
							"bu",
							"according",
							"experience",
							"work",
							"real",
							"sour",
							"sometimes",
							"many",
							"savigny",
							"play",
							"st",
							"silly",
							"similar",
							"birthday",
							"dz",
							"holds",
							"today",
							"gerrard",
							"middle",
							"song",
							"ve"			
			], 
			'agreeable':[
							"https",
							"birthday",
							"made",
							"google",
							"important",
							"need",
							"church",
							"oh",
							"haha",
							"early",
							"hearts",
							"personal",
							"one",
							"eat",
							"girl",
							"go",
							"mo",
							"ly",
							"facebook",
							"amazing",
							"keeping",
							"speak",
							"iv",
							"secret",
							"room",
							"fate",
							"sit",
							"married",
							"background",
							"sharedleadership",
							"ward",
							"anyone",
							"dream",
							"succes",
							"needs",
							"views",
							"annoyed",
							"habit",
							"walk"			
			], 
			'open':[
							"love",
							"time",
							"years",
							"http",
							"goes",
							"dreams",
							"birthday",
							"high",
							"win",
							"world",
							"wanna",
							"digital",
							"replies",
							"would",
							"women",
							"ready",
							"get",
							"wall",
							"point",
							"lot",
							"project",
							"mean",
							"meet",
							"right",
							"people",
							"page",
							"season",
							"bit",
							"fall",
							"qenbj",
							"er",
							"looks",
							"year",
							"go",
							"want",
							"midnight",
							"username",
							"attention",
							"cold",
							"like",
							"little",
							"psd",			
			], 
			'conscientious':[
							"awesome",
							"party",
							"maybe",
							"crazy",
							"ff",
							"using",
							"thanks",
							"little",
							"new",
							"could",
							"tears",
							"long",
							"thirty",
							"saying",
							"system",
							"find",
							"wtf",
							"one",
							"someone",
							"reason",
							"john",
							"lasting",
							"re",
							"five",
							"reat",
							"http",
							"via",
							"thrones",
							"words",
							"furious",
							"sjgy",
							"bout",
							"thank",
							"mini",
							"qw",
							"central",
							"looks",
							"playing"			
			]
			},
'dutch':{
			'age': [
						"zit",
						"heel",
						"best",
						"geeft",
						"idee",
						"nooit",
						"weer",
						"binnen",
						"goed",
						"avond",
						"bijwerken",
						"dag",
						"laatste",
						"man",
						"voelt",
						"hart",
						"toekomst",
						"boeit",
						"dh",
						"feestje",
						"ging",
						"meisje",
						"morgen",
						"muzikanten",
						"onderweg",
						"onderzoeksjournalistiek",
						"onzin",
						"proficiat",
						"ten",
						"verdient",
						"verzuurde",
						"werkt"			
			],
			'gender':[
						"username",
						"goed",
						"bent",
						"saai",			
					],
			'extroverted':[
							"dingen",
							"blijft",
							"bijna",
							"mr",
							"zeker",
							"vallen",
							"doet",
							"xkwktrd",
							"zoek"			
			], 
			'stable':[
							"username",
							"snel",
							"misschien",
							"ergens",
							"blijft",
							"namelijk",
							"jaar",
							"vrijdag",
							"terwijl",
							"hashtag",
							"interviewee",			
			], 
			'agreeable':[
							"rt",
							"terug",
							"snel",
							"bedankt",
							"smh",
							"terwijl",
							"the",
							"heerlijk",
							"hallo"			
			], 
			'open':[
							"hahaha",
							"week",
							"tijd",
							"username",
							"we",
							"kaviaarbehandeling",
							"jeeeej",
							"can"			
			], 
			'conscientious':[
							"mag",
							"fietsen",
							"mn",
							"dacht",
							"zet",
							"moddermanstraat"			
			]
			},
'italian':{
			'age':[
						"domani",
						"fa",
						"poi",
						"pezzo",
						"immagini",
						"quel",
						"ultimo",
						"binari",
						"bravo",
						"foto",
						"is",
						"sentito",
						"stato",
						"pi",
						"seguire",
						"borgo",
						"elected",
						"federico",
						"riusciamo",
						"super",
						"tassoni",
						"agendadigitale",
						"casalinga",
						"cc",
						"de",
						"dio",
						"eccomi",
						"esempio",
						"novit",
						"oscena",
						"pard",
						"piazza",
						"preso",
						"pu",
						"rispetto",
						"yg"		
				],
			'gender':[
						"co",
						"campagna",
						"ottimo",
						"conoscessi",
						"voci"			
					],
			'extroverted':[
							"design",
							"hotel",
							"ore",
							"dopo",
							"oppure",
							"ariosto",
							"scaccia",
							"son",
							"date"			
			], 
			'stable':[
							"co",
							"design",
							"sostenibile",
							"andare",
							"me",
							"esempio",
							"at",
							"buone",
							"semplicissima",
							"incapace",
							"tv"			
			], 
			'agreeable':[
							"bologna",
							"via",
							"twitter",
							"style",
							"co",
							"sento",
							"monti",
							"disegni"			
			], 
			'open':[
							"qualcosa",
							"anni",
							"bel",
							"ricerca",
							"sangue",
							"zagaria",
							"sento",
							"striati"			
			], 
			'conscientious':[
							"design",
							"ore",
							"username",
							"anni",
							"sembra",
							"oppure",
							"massimo",
							"purtroppo",
							"confermo"			
			]			
			},
'spanish':{
			'age':[
						"http",
						"ma",
						"dijo",
						"momento",
						"cil",
						"as",
						"buenos",
						"mala",
						"bieber",
						"falta",
						"buscan",
						"facebook",
						"info",
						"todas",
						"favor",
						"cula",
						"nom",
						"ofpbmahc"		
					],
			'gender':[
						"vida",
						"alguien",
						"corrupci",
						"ciudades",
						"si",
						"temprano",
						"puro",
						"meta",
						"foto",
						"dio"			
						],
			'extroverted':[
						"xico",
						"alguien",
						"escribir",
						"tambi",
						"nueva",
						"pe",
						"gusto",
						"http",
						"comen",
						"mujeres",
						"fico",
						"toda",
						"quiero",
						"sue",
						"aunque",
						"ahora",
						"chistes",
						"mano",
						"ser",
						"luz",
						"verdad",
						"dar",
						"hoy",
						"cticas",
						"che",
						"suicidio",
						"portugal",
						"recuerdo",
						"responsabilidad",			
			], 
			'stable':[
						"amigos",
						"is",
						"quiero",
						"ja",
						"despertar",
						"noches",
						"buenos",
						"ah",
						"mayor",
						"quieres",
						"bado",
						"iphone",
						"est",
						"culo",
						"sesi",
						"cient",
						"pel",
						"you",
						"sab",
						"internet",
						"torno",
						"tardando",
						"podemos",
						"tampoco",
						"nnjutigybf",
						"corriendo",
						"va",
						"acompa",
						"hacer",
						"papaya",
						"vas",
						"bonitas",			
			], 
			'agreeable':[
						"sabes",
						"cc",
						"dif",
						"quedan",
						"username",
						"despedida",
						"estudiar",
						"vez",
						"pesar",
						"vamos",
						"esperar",
						"tambi",
						"solo",
						"sociales",
						"hacen",
						"luego",
						"ngelamaria",
						"fin",
						"acordaba",
						"terror",
						"ja",
						"bellas",
						"firmad",
						"fr",			
			], 
			'open':[
						"puta",
						"jajaja",
						"interesante",
						"luego",
						"espa",
						"esperar",
						"dia",
						"acuerdo",
						"grande",
						"ma",
						"amigo",
						"siempre",
						"sonrisa",
						"haber",
						"pista",
						"buenos",
						"penlties",
						"aburrida",
						"burra",
						"venes",
						"pelotita",
						"crisis",
						"youtube",
						"social",
						"hombres",
						"plana",
						"serie",			
			], 
			'conscientious':[
						"siempre",
						"fer",
						"cc",
						"rtela",
						"tico",
						"corrupci",
						"solo",
						"momento",
						"mundo",
						"mal",
						"empleo",
						"do",
						"pone",
						"va",
						"transici",
						"veces",
						"pa",
						"escuchar",
						"mayor",
						"meses",
						"puede",
						"ciento",
						"andar",
						"article",
						"gt",
						"moralmente",
						"preguntar",
						"online"			
			]
			}
}


stylistic_features = [ 
				"#",
				"@username",
				"http://",
				":)",
				";)",
				"o_O",
				"!",
				"!!",
				"!!!",
				":("
				]

n_words = 10000
				
def getCount(onePattern, inputString):
	return inputString.count(onePattern)
	

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
    
    

def getFeatureVecFromTFIDF(fileName, lang):
	train = pd.read_csv(fileName, header=0, delimiter="\t", quoting=1)
	num_text = train["text"].size

	clean_train_reviews = []
	print "Looping through all text.\n" 

	for i in xrange( 0, num_text):
		clean_train_reviews.append( review_to_words( train["text"][i], lang ) )

	vectorizer = TfidfVectorizer(analyzer = "word",\
								 tokenizer = None,      \
								 preprocessor = None,   \
								 stop_words = None,     \
								 max_features = n_words) 

	X = vectorizer.fit_transform(clean_train_reviews)
	X = X.toarray()
	return X
	
def	getFeatureVecFromFunctionWords(fileName, test_patterns):
	train = pd.read_csv(fileName, header=0, delimiter="\t", quoting=1)
	train_reviews = train["text"]
	
	X = []
	for line in train_reviews:
		vector_for_one_entry = []
		for pattern in test_patterns:
			count = getCount(pattern, line)
			vector_for_one_entry.append(count)
		X.append(vector_for_one_entry)
	X = np.array(X)
	return X

	
def	getFeatureVecFromStylisticFeatures(fileName, stylistic_features):
	return getFeatureVecFromFunctionWords(fileName, stylistic_features)

def	getFeatureVecFromPOS(fileName, lang, n_gram_range):
	train = pd.read_csv(fileName, header=0, delimiter="\t", quoting=1)
	num_text = train["text"].size

	clean_train_reviews = []
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

	bigram_vectorizer = CountVectorizer(ngram_range=n_gram_range, min_df=1)
	X = bigram_vectorizer.fit_transform(train_reviews_pos_tags).toarray()
	return X

def main():
	for lang, fileName in zip(languages, datafiles):	
		print lang
		for task in tasks:
			print task

# 			print "TFIDF"
# 			X = getFeatureVecFromTFIDF(fileName, lang)
# 			output_file_name = lang + "_" + task + "_X_tfidf.csv"
# 			np.savetxt(output_file_name, X, delimiter=",")

			print "Function words"		
			X = getFeatureVecFromFunctionWords(fileName, function_words_dict[lang][task])
			output_file_name = lang + "_" + task + "_X_function_words.csv"
			np.savetxt(output_file_name, X, delimiter=",")

# 			print "Stylistic Features"		
# 			X = getFeatureVecFromStylisticFeatures(fileName, stylistic_features)
# 			output_file_name = lang + "_" + task + "_X_stylistic_features.csv"
# 			np.savetxt(output_file_name, X, delimiter=",")		
# 
# 			print "POS Unigrams"		
# 			X = getFeatureVecFromPOS(fileName, lang, (1,1))
# 			output_file_name = lang + "_" + task + "_X_unigrams.csv"
# 			np.savetxt(output_file_name, X, delimiter=",")
# 
# 			print "POS Bigrams"		
# 			X = getFeatureVecFromPOS(fileName, lang, (1,2))
# 			output_file_name = lang + "_" + task + "_X_bigrams.csv"
# 			np.savetxt(output_file_name, X, delimiter=",")	
# 	
	
if __name__ == "__main__":
	main()