#!/usr/bin/python

import sys
import getopt
import bleach
import xml.etree.ElementTree as ET
import os
import re
import csv

reload(sys)
sys.setdefaultencoding("ISO-8859-1")


def trainAll(inputDir):
	pass

def writeModels(models, outputDir):
	pass

def dirExists(inputDir):
	if os.path.exists(inputDir):
		return True
	elif os.access(os.path.dirname(inputDir), os.W_OK):
		print "Cannot access the directory. Check for privileges."
		return False
	else:
		print "Directory does not exist."
		return False

def absoluteFilePaths(directory):
	allPaths = []
	for dirpath,_,filenames in os.walk(directory):
		for f in filenames:
			onePath = os.path.abspath(os.path.join(dirpath, f))
			allPaths.append(onePath)
# 			yield os.path.abspath(os.path.join(dirpath, f))
	return allPaths

def getAllFilenamesWithAbsPath(inputDir):
	if dirExists(inputDir):
		allPaths = absoluteFilePaths(inputDir)
		return allPaths
	else:
		sys.exit()

def isTruthTextFile(f):
	return 'truth.txt' in f
	
def getTruthTextFiles(allPaths):
	return [f for f in allPaths if isTruthTextFile(f)]

def getRelevantDirectories(argv):
   inputDir = ''
   outputDir = ''
   modelDir = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
      print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>'
         print 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputDir = arg
      elif opt in ("-o", "--ofile"):
         outputDir = arg   
   return inputDir, outputDir

def tsv_writer(data, path):
    """
    Write data to a TSV file path
    """
    with open(path, "a") as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(data)


def writeOneSummary(outputFilename, f, allPaths):
	data = ["filename", "gender", \
			"age", "extroverted", \
			"stable", "agreeable", \
			"open", "conscientious", \
			"text"]
	path = outputFilename.strip().split("/")
	outputFilename = path[-1]
	path = '/'.join(path[0:-1])
	print "Output filename: ",outputFilename

	tsv_writer(data, outputFilename)
	gender = {'M': 0, 'F':1}
	ageGroup = {'18-24': 0, \
				'25-34': 1, \
				'35-49': 2, \
				'50-XX': 3, \
				'50-64': 3, \
				'XX-XX': None}
	file = open(f, 'r')
	
	for line in file:
		a = line.strip().split(":::")
		fileName 		  = path+ "/" + a[0] + ".xml"
# 		print fileName
		thisGender 	 	  = gender[a[1]]
		thisAgeGroup 	  = ageGroup[a[2]]
		thisExtroverted   = float(a[3])
		thisStable 		  = float(a[4])
		thisAgreeable	  = float(a[5])
		thisOpen		  = float(a[6])
		thisConscientious = float(a[7])
		
# 		print "%s %d %d %f %f %f %f %f" % (fileName, thisGender, thisAgeGroup, thisExtroverted, thisStable, thisAgreeable, thisOpen, thisConscientious)
		
		try:
			tree = ET.parse(fileName)
			print "Filename: %s SUCCESS!" % fileName
		
		except:
			e = sys.exc_info()[0]
			print "Filename: %s Error: %s" % (fileName, e)
		else:
# 			print "In Else"
			allDocs = tree.getroot().findall("document")
 			allText = ""
			
# 			print "Going in for loop"
			for doc in allDocs:
				clean = bleach.clean(doc.text, tags=[], strip=True)
 				allText = allText + clean
	 		
			# 	clean = clean.encode('utf-8')
			allText = allText.encode('utf-8')
# 			print "Out of loop, writing"								
			data = [fileName, thisGender, thisAgeGroup, thisExtroverted, \
					thisStable, thisAgreeable, thisOpen, thisConscientious, \
					allText]
			tsv_writer(data, outputFilename)
			print "Finish writing one line"

	
	   
def main(argv):
# 	print "Getting relevant directories"
	inputDir, outputDir = getRelevantDirectories(argv)
# 	print "Getting all filenames with absolute paths"
	allPaths = getAllFilenamesWithAbsPath(inputDir)
# 	print "Getting all truth files"
	allTruthText = getTruthTextFiles(allPaths)
	models = {}
	tasks = ["gender", "age", "extroverted", "stable", "agreeable", "open", "conscientious"]
	langs = ["english", "dutch", "spanish", "italian"]
# 	allTruthText = [a for a in allTruthText if "english" in a]
# 	import pdb; pdb.set_trace()
	for f in allTruthText:
		a = f.strip().split("/")
		lang = [ lang for lang in langs if lang in f]
		print "Processing: ", lang[0]
# 		import pdb; pdb.set_trace()
		outputFilename = '/'.join(a[0:-1]) + '/summary-' + lang[0] + '-' + a[-1]
		print "Output filename: ", outputFilename
		print "Writing one summary"
		writeOneSummary(outputFilename, f, allPaths)
# 		descriptors = getDescriptorsForOne(outputFilename)
# 		model_for_one = {}
# 		for task in tasks:
# 			y = getTarget(outputFilename, task)
# 			model_for_one[task] = trainOne(descriptors, y)
# 		models[f] = model_for_one
	models = trainAll(inputDir)
	writeModels(models, outputDir)

if __name__ == "__main__":
   main(sys.argv[1:])