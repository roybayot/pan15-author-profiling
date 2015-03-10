#!/usr/bin/python
# This script is not yet complete. It just uses the majority.

import sys, getopt
import os
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

def getRelevantDirectories(argv):
   inputDir = ''
   outputDir = ''
   modelDir = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","ofile=","mfile="])
   except getopt.GetoptError:
      print './myTestingSoftware.py -i <inputdirectory> -m <modelfile> -o <outputdirectory>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print './myTestingSoftware.py -i <inputdirectory> -m <modelfile> -o <outputdirectory>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputDir = arg
      elif opt in ("-m", "--mfile"):
      	 modelDir = arg
      elif opt in ("-o", "--ofile"):
         outputDir = arg   
   return inputDir, outputDir, modelDir

def getAllModels(modelDir):
	return None

def dirExists(inputDir):
	if os.path.exists(inputDir):
		return True
	elif os.access(os.path.dirname(inputDir), os.W_OK):
		print "Cannot access the directory. Check for privileges."
		return False
	else:
		print "Directory does not exist."
		return False

def isXML(f):
	a = f.strip().split('.')
	if a[1] == 'xml':
		return True
	else:
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

def getAllTestFiles(inputDir):
	if dirExists(inputDir):
		allTestFiles = [ f for f in listdir(inputDir) if isfile(join(inputDir,f)) ]
		allTestFiles = [ f for f in allTestFiles if isXML(f) ]
		return allTestFiles
	else:
		sys.exit()

def getAllXmlFiles(allTestFiles):
	allTestFiles = [ f for f in allTestFiles if isfile(f) ]
	allTestFiles = [ f for f in allTestFiles if isXML(f) ]
	return allTestFiles	

def getLanguage(oneFile):
	tree = ET.parse(oneFile)
	root = tree.getroot()
	a = root.attrib
	return a['lang']
	
def classifyTestFiles(models, inputDir):
	results = {}
	
	models = {'nl': { 'gender'		 : 'male', \
					  'age'			 : 'XX-XX', \
					  'extroverted'	 : '0.2', \
					  'stable'		 : '0.4', \
					  'agreeable'	 : '0.1', \
					  'open'		 : '0.1', \
					  'conscientious': '0.4'
					}, \
			  'en': { 'gender'		 : 'male', \
					  'age'			 : '25-34', \
					  'extroverted'	 : '0.1', \
					  'stable'		 : '0.2', \
					  'agreeable'	 : '0.2', \
					  'open'		 : '0.1', \
					  'conscientious': '0.1' \
			  		}, \
			  'it': { 'gender'		 : 'male', \
					  'age'			 : 'XX-XX', \
					  'extroverted'	 : '0.1', \
					  'stable'		 : '0.1', \
					  'agreeable'	 : '0.1', \
					  'open'		 : '0.1', \
					  'conscientious': '0.1'
					}, \
			  'es': { 'gender'		 : 'male', \
					  'age'			 : '25-34', \
					  'extroverted'	 : '0.2', \
					  'stable'		 : '-0.1', \
					  'agreeable'	 : '0.2', \
					  'open'		 : '0.4', \
					  'conscientious': '0.1' \
					}, \
			  }	
	allTestFiles = getAllFilenamesWithAbsPath(inputDir)
	allTestFiles = getAllXmlFiles(allTestFiles)
	
	for oneFile in allTestFiles:
		lang = getLanguage(oneFile)
		model = models[lang]
		aa = oneFile.strip().split("/")
		aa = aa[-1].strip().split(".")
		thisId					= aa[0]
# 		print oneFile
		thisType				= 'twitter'
		thisLanguage			= lang
		predictedGender 	 	= model['gender']
		predictedAge    	 	= model['age']
		predictedExtroverted 	= model['extroverted']
		predictedStable 		= model['stable']
		predictedAgreeable	  	= model['agreeable']
		predictedOpen		  	= model['open']
		predictedConscientious 	= model['conscientious']

		results[oneFile] =  [ thisId, \
							  thisType, \
							  thisLanguage, \
							  predictedGender, \
							  predictedAge, \
							  predictedExtroverted, \
							  predictedStable, \
							  predictedAgreeable, \
							  predictedOpen, \
							  predictedConscientious
							]

	return results

# def writeOneResult(key, value, outputDir):
# 	pass

def writeOneResult(key, value, outputDir):
	key = key.strip().split("/")
	cwd = os.getcwd()
	path = cwd + "/" + outputDir + "/" + key[-1]
	thisId					= value[0]
	thisType				= value[1]
	thisLanguage			= value[2]
	predictedGender 	 	= value[3]
	predictedAge    	 	= value[4]
	predictedExtroverted 	= value[5]
	predictedStable 		= value[6]
	predictedAgreeable	  	= value[7]
	predictedOpen		  	= value[8]
	predictedConscientious 	= value[9]

	
	text_to_write = """<author id='%s'\n\ttype='%s'\n\tlang='%s'\n\tage_group='%s'\n\tgender='%s'\n\textroverted='%s'\n\tstable='%s'\n\tagreeable='%s'\n\tconscientious='%s'\n\topen='%s'\n/>"""% (thisId, thisType, thisLanguage, predictedAge, predictedGender, \
  		  predictedExtroverted, predictedStable, predictedAgreeable, \
  		  predictedConscientious, predictedOpen)
	# Open a file
	fo = open(path, "w")
	fo.write( text_to_write );
	fo.close()
	
def makeDirectory(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
		else:
			print "\nBE CAREFUL! Directory %s already exists." % path
		
def writeAllResults(results, outputDir):
	if (not dirExists(outputDir)):
		print "Creating new directory."
		makeDirectory(outputDir)
	for key, value in results.iteritems():
#  		print key, ":", value
		writeOneResult(key, value, outputDir)	
		
def main(argv):
	inputDir, outputDir, modelDir = getRelevantDirectories(argv)
# 	print 'Input directory is "',  inputDir
# 	print 'Model directory is "',  modelDir   
# 	print 'Output directory is "', outputDir
	
	models = getAllModels(modelDir)
	results = classifyTestFiles(models, inputDir)
# 	print results
	writeAllResults(results, outputDir)

   
if __name__ == "__main__":
	main(sys.argv[1:])